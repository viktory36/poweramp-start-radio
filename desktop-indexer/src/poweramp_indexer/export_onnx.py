"""ONNX export for MuQ-MuLan and Music Flamingo models.

Exports both models to ONNX format for on-device inference on Android via ONNX Runtime.

MuQ-MuLan: raw waveform [1, 240000] @ 24kHz -> [1, 512] embedding
Flamingo:  mel features [B, 128, 3000] + mask + audio_times -> [B, 750, 3584] hidden states
"""

import logging
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MuLanAudioOnnxWrapper(nn.Module):
    """Wraps MuQ-MuLan audio path for ONNX export.

    Input:  wav [1, 240000] (10s @ 24kHz)
    Output: embedding [1, 512] (L2-normalized)

    Bypasses MuQMuLan.forward() which has Python for-loops (iterates batch
    items and clips) that confuse the ONNX tracer. Instead calls the internal
    MuLan audio tower directly: get_audio_latents() is a clean tensor path.
    """

    def __init__(self, mulan_model):
        super().__init__()
        # mulan_module property unwraps DDP if present, gives us the MuLan model
        # MuLan.get_audio_latents: audio → AudioSpectrogramTransformer → project → L2-norm
        self.mulan_inner = mulan_model.mulan_module

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [1, 240000] raw waveform at 24kHz (10s clip)
        Returns:
            [1, 512] L2-normalized audio embedding
        """
        # Direct path: wav → mel_stft → encoder → linear projection → L2-normalize
        # No Python loops, clean tensor ops only
        return self.mulan_inner.get_audio_latents(wav)


class FlamingoOnnxWrapper(nn.Module):
    """Wraps Music Flamingo encoder + projector for ONNX export.

    Input:  input_features [B, 128, 3000] (Whisper mel spectrogram)
            input_features_mask [B, 3000]
            audio_times [B, 750] (absolute timestamps per post-pool frame)
    Output: hidden [B, 750, 3584] (encoder output projected to LLM space)

    Mel spectrogram computation stays outside ONNX (done in Kotlin on Android).
    """

    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        audio_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_features: [B, 128, 3000] mel spectrogram
            input_features_mask: [B, 3000] attention mask (1 = valid)
            audio_times: [B, 750] absolute timestamps per frame
        Returns:
            [B, 750, 3584] projected hidden states (or [B, 750, 1280] without projector)
        """
        output = self.encoder(
            input_features,
            input_features_mask=input_features_mask,
            audio_times=audio_times,
        )
        hidden = output.last_hidden_state  # [B, 750, 1280]
        if self.projector is not None:
            hidden = self.projector(hidden)  # [B, 750, 3584]
        return hidden


def export_mulan_onnx(output_path: Path, fp16: bool = True, opset: int = 17):
    """Export MuQ-MuLan audio encoder to ONNX.

    Args:
        output_path: Path for the .onnx file
        fp16: Whether to export in FP16 (halves model size)
        opset: ONNX opset version (17+ needed for torch.stft)
    """
    from muq import MuQMuLan

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading MuQ-MuLan model...")
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval()

    if fp16:
        model = model.half()
    model = model.to(device)

    wrapper = MuLanAudioOnnxWrapper(model)
    wrapper.eval()

    # 10s clip at 24kHz
    dummy_wav = torch.randn(1, 240000, device=device)
    if fp16:
        dummy_wav = dummy_wav.half()

    logger.info(f"Exporting MuQ-MuLan to {output_path} (opset {opset}, fp16={fp16}, device={device})...")

    try:
        torch.onnx.export(
            wrapper,
            (dummy_wav,),
            str(output_path),
            input_names=["wav"],
            output_names=["embedding"],
            dynamic_axes={
                "wav": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as e:
        logger.warning(f"Standard ONNX export failed: {e}")
        logger.info("Trying with mel spectrogram as preprocessing step...")
        return _export_mulan_split(model, output_path, fp16, opset, device)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported MuQ-MuLan: {size_mb:.1f} MB")

    del model, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def _find_mel_module(root_module):
    """Find the mel/STFT feature extractor in the module tree.

    Searches by class name (torchaudio MelSpectrogram, Spectrogram, etc.)
    and by module path (muq.muq.modules.features).

    Returns (dotted_name, module) or (None, None) if not found.
    """
    # Strategy 1: look for torchaudio mel/spectrogram transforms
    for name, mod in root_module.named_modules():
        class_name = type(mod).__name__
        if class_name in ('MelSpectrogram', 'MelSTFT', 'Spectrogram'):
            return name, mod

    # Strategy 2: look for muq's feature extractor by module path
    for name, mod in root_module.named_modules():
        mod_path = type(mod).__module__ or ""
        if 'features' in mod_path and hasattr(mod, '__call__'):
            return name, mod

    # Strategy 3: look by attribute name patterns
    for name, mod in root_module.named_modules():
        if any(x in name for x in ['preprocessing', 'feature_extract', 'mel_stft']):
            return name, mod

    return None, None


def _replace_module_by_name(root, dotted_name, replacement):
    """Replace a submodule identified by dotted name (e.g. 'audio.model.feat')."""
    parts = dotted_name.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)


class MuLanPostMelWrapper(nn.Module):
    """Fallback wrapper that takes mel features instead of raw waveform.

    Used when torch.stft fails to export in ONNX. The mel spectrogram
    is computed outside the ONNX model (in Python preprocessing or on Android).

    The mel/STFT module (found by scanning the module tree) is replaced with
    nn.Identity(), so mel features pass through unchanged into the encoder.
    """

    def __init__(self, mulan_inner, mel_module_name):
        super().__init__()
        self.mulan_inner = mulan_inner
        # Replace mel/STFT with identity — mel features become the direct input
        _replace_module_by_name(self.mulan_inner, mel_module_name, nn.Identity())

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_features: Output of the original mel/STFT module
        Returns:
            [1, 512] L2-normalized audio embedding
        """
        return self.mulan_inner.get_audio_latents(mel_features)


def _export_mulan_split(model, output_path: Path, fp16: bool, opset: int, device: str = "cpu"):
    """Fallback: export MuQ-MuLan with mel spectrogram as input.

    If raw waveform export fails (torch.stft ONNX issues), we split the model:
    mel spectrogram computed on the caller side, model takes mel features as input.
    """
    logger.info("Attempting split export (mel features as input)...")

    mulan_inner = model.mulan_module

    # Log module tree for debugging
    logger.info("Module tree (searching for mel/STFT):")
    for name, mod in mulan_inner.named_modules():
        if name.count('.') <= 3:  # don't log too deep
            logger.info(f"  {name}: {type(mod).__name__} ({type(mod).__module__})")

    # Step 1: Find the mel/STFT module
    mel_name, mel_module = _find_mel_module(mulan_inner)
    if mel_module is None:
        raise RuntimeError(
            "Could not find mel/STFT module in MuQ-MuLan model. "
            "Run with --verbose to see the module tree."
        )
    logger.info(f"Found mel module: '{mel_name}' ({type(mel_module).__name__})")

    # Step 2: Capture mel output shape via hook
    mel_output = {}

    def hook_fn(module, input, output):
        mel_output['shape'] = output.shape
        mel_output['tensor'] = output.detach()

    handle = mel_module.register_forward_hook(hook_fn)

    dummy_wav = torch.randn(1, 240000, device=device)
    if fp16:
        dummy_wav = dummy_wav.half()

    with torch.no_grad():
        mulan_inner.get_audio_latents(dummy_wav)
    handle.remove()

    mel_shape = mel_output['shape']
    logger.info(f"Captured mel output shape: {mel_shape}")

    # Extract mel spectrogram parameters for Android replication
    mel_params = _extract_mel_params(mel_module)
    logger.info(f"Mel params: {mel_params}")

    # Step 3: Replace mel with Identity and export
    wrapper = MuLanPostMelWrapper(mulan_inner, mel_name)
    wrapper.eval()

    dummy_mel = mel_output['tensor']

    logger.info(f"Exporting post-mel model to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(output_path),
        input_names=["mel_features"],
        output_names=["embedding"],
        dynamic_axes={
            "mel_features": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Save mel parameters so Android can replicate the mel spectrogram
    mel_params_path = output_path.with_suffix(".mel_params.json")
    import json
    params = {
        "input_type": "mel_features",
        "mel_shape": [int(x) for x in mel_shape],
        "sample_rate": 24000,
        "clip_duration_s": 10,
        **mel_params,
    }
    with open(mel_params_path, 'w') as f:
        json.dump(params, f, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported MuQ-MuLan (post-mel): {size_mb:.1f} MB")
    logger.info(f"Mel parameters saved to: {mel_params_path}")
    return output_path


def _extract_mel_params(preproc_module) -> dict:
    """Extract mel spectrogram parameters from the preprocessing module.

    Walks the module tree to find torchaudio MelSpectrogram / Spectrogram
    and extracts n_fft, hop_length, n_mels, etc. for Android replication.
    """
    params = {}

    for name, mod in preproc_module.named_modules():
        class_name = type(mod).__name__
        # torchaudio.transforms.MelSpectrogram
        if 'MelSpectrogram' in class_name or 'MelScale' in class_name:
            for attr in ['n_mels', 'sample_rate', 'f_min', 'f_max', 'n_stft']:
                if hasattr(mod, attr):
                    val = getattr(mod, attr)
                    params[attr] = val if not isinstance(val, torch.Tensor) else val.item()
        # torchaudio.transforms.Spectrogram
        if 'Spectrogram' in class_name:
            for attr in ['n_fft', 'hop_length', 'win_length', 'power', 'normalized']:
                if hasattr(mod, attr):
                    val = getattr(mod, attr)
                    params[attr] = val if not isinstance(val, torch.Tensor) else val.item()

    # Fallback: check direct attributes on the preprocessing module itself
    for attr in ['n_fft', 'hop_length', 'win_length', 'n_mels', 'sample_rate',
                 'f_min', 'f_max', 'power', 'normalized']:
        if attr not in params and hasattr(preproc_module, attr):
            val = getattr(preproc_module, attr)
            params[attr] = val if not isinstance(val, torch.Tensor) else val.item()

    return params


def export_flamingo_onnx(output_path: Path, fp16: bool = True, opset: int = 17):
    """Export Music Flamingo encoder + projector to ONNX.

    Args:
        output_path: Path for the .onnx file
        fp16: Whether to export in FP16
        opset: ONNX opset version
    """
    from safetensors.torch import load_file
    from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoEncoder

    from .embeddings_flamingo import (
        AudioProjector,
        FRAME_DURATION_S,
        get_flamingo_encoder_path,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_path = get_flamingo_encoder_path()
    logger.info(f"Loading Music Flamingo encoder from {encoder_path}...")

    encoder = MusicFlamingoEncoder.from_pretrained(str(encoder_path))
    encoder.eval()

    # Load projector if available
    projector_path = encoder_path / "projector.safetensors"
    projector = None
    if projector_path.exists():
        logger.info("Loading projector...")
        projector = AudioProjector()
        state_dict = load_file(str(projector_path), device="cpu")
        projector.load_state_dict(state_dict)
        projector.eval()

    if fp16:
        encoder = encoder.half()
        if projector is not None:
            projector = projector.half()

    encoder = encoder.to(device)
    if projector is not None:
        projector = projector.to(device)

    wrapper = FlamingoOnnxWrapper(encoder, projector)
    wrapper.eval()

    # Dummy inputs: single 30s chunk
    batch_size = 1
    dummy_mel = torch.randn(batch_size, 128, 3000, device=device)
    dummy_mask = torch.ones(batch_size, 3000, dtype=torch.long, device=device)
    dummy_times = (torch.arange(750, dtype=torch.float32, device=device).unsqueeze(0)
                   * FRAME_DURATION_S)

    if fp16:
        dummy_mel = dummy_mel.half()
        dummy_times = dummy_times.half()

    logger.info(f"Exporting Flamingo to {output_path} (opset {opset}, fp16={fp16}, device={device})...")

    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_mask, dummy_times),
        str(output_path),
        input_names=["input_features", "input_features_mask", "audio_times"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "input_features_mask": {0: "batch"},
            "audio_times": {0: "batch"},
            "hidden_states": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported Flamingo: {size_mb:.1f} MB")
    return output_path


def verify_mulan_onnx(onnx_path: Path, num_tracks: int = 10, tolerance: float = 1e-3):
    """Verify MuQ-MuLan ONNX export against PyTorch reference.

    Handles both full-model export (input=wav) and split export (input=mel_features).
    For split models, generates mel features via PyTorch preprocessing.
    """
    import onnxruntime as ort
    from muq import MuQMuLan

    logger.info(f"Verifying MuQ-MuLan ONNX: {onnx_path}")

    # Load PyTorch model
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval()

    # Load ONNX model and detect input type
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    input_dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32
    is_split_model = input_name == "mel_features"

    if input_dtype == np.float16:
        model = model.half()

    logger.info(f"  Input: {input_name} ({'split/mel' if is_split_model else 'full/wav'}), "
                f"dtype: {'fp16' if input_dtype == np.float16 else 'fp32'}")

    max_diff = 0.0
    all_cos_sims = []

    for i in range(num_tracks):
        # Random 10s audio clip
        wav = torch.randn(1, 240000)
        if input_dtype == np.float16:
            wav = wav.half()

        with torch.no_grad():
            # PyTorch reference — full path through get_audio_latents
            ref_emb = model.mulan_module.get_audio_latents(wav).cpu().numpy()

            if is_split_model:
                # For split model: compute mel features via PyTorch preprocessing
                mel = model.mulan_module.audio.model.preprocessing(wav)
                onnx_input = mel.cpu().numpy()
            else:
                onnx_input = wav.numpy()

        # ONNX inference
        onnx_emb = sess.run(None, {input_name: onnx_input})[0]

        diff = np.abs(ref_emb - onnx_emb).max()
        cos_sim = np.dot(ref_emb.flatten(), onnx_emb.flatten()) / (
            np.linalg.norm(ref_emb) * np.linalg.norm(onnx_emb) + 1e-10
        )

        max_diff = max(max_diff, diff)
        all_cos_sims.append(cos_sim)
        logger.info(f"  Track {i+1}/{num_tracks}: max_diff={diff:.6f}, cos_sim={cos_sim:.6f}")

    mean_cos_sim = np.mean(all_cos_sims)
    passed = max_diff < tolerance
    logger.info(f"MuQ-MuLan verification: max_diff={max_diff:.6f}, "
                f"mean_cos_sim={mean_cos_sim:.6f}, passed={passed}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed, max_diff, mean_cos_sim


def verify_flamingo_onnx(onnx_path: Path, num_tracks: int = 10, tolerance: float = 1e-3):
    """Verify Flamingo ONNX export against PyTorch reference."""
    import onnxruntime as ort
    from safetensors.torch import load_file
    from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoEncoder

    from .embeddings_flamingo import (
        AudioProjector,
        FRAME_DURATION_S,
        get_flamingo_encoder_path,
    )

    logger.info(f"Verifying Flamingo ONNX: {onnx_path}")

    encoder_path = get_flamingo_encoder_path()
    encoder = MusicFlamingoEncoder.from_pretrained(str(encoder_path))
    encoder.eval()

    projector_path = encoder_path / "projector.safetensors"
    projector = None
    if projector_path.exists():
        projector = AudioProjector()
        state_dict = load_file(str(projector_path), device="cpu")
        projector.load_state_dict(state_dict)
        projector.eval()

    # Check if ONNX model is FP16
    sess = ort.InferenceSession(str(onnx_path))
    input_dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32

    if input_dtype == np.float16:
        encoder = encoder.half()
        if projector is not None:
            projector = projector.half()

    max_diff = 0.0
    all_cos_sims = []

    for i in range(num_tracks):
        # Random mel spectrogram (30s chunk)
        mel = torch.randn(1, 128, 3000)
        mask = torch.ones(1, 3000, dtype=torch.long)
        times = torch.arange(750, dtype=torch.float32).unsqueeze(0) * FRAME_DURATION_S

        if input_dtype == np.float16:
            mel = mel.half()
            times = times.half()

        # PyTorch reference
        with torch.no_grad():
            output = encoder(mel, input_features_mask=mask, audio_times=times)
            hidden = output.last_hidden_state
            if projector is not None:
                hidden = projector(hidden)
            ref = hidden.cpu().numpy()

        # ONNX inference
        feed = {
            "input_features": mel.numpy(),
            "input_features_mask": mask.numpy(),
            "audio_times": times.numpy(),
        }
        onnx_out = sess.run(None, feed)[0]

        diff = np.abs(ref - onnx_out).max()
        # Cosine similarity on mean-pooled output
        ref_pooled = ref.mean(axis=1).flatten()
        onnx_pooled = onnx_out.mean(axis=1).flatten()
        cos_sim = np.dot(ref_pooled, onnx_pooled) / (
            np.linalg.norm(ref_pooled) * np.linalg.norm(onnx_pooled) + 1e-10
        )

        max_diff = max(max_diff, diff)
        all_cos_sims.append(cos_sim)
        logger.info(f"  Track {i+1}/{num_tracks}: max_diff={diff:.6f}, cos_sim={cos_sim:.6f}")

    mean_cos_sim = np.mean(all_cos_sims)
    passed = max_diff < tolerance
    logger.info(f"Flamingo verification: max_diff={max_diff:.6f}, "
                f"mean_cos_sim={mean_cos_sim:.6f}, passed={passed}")

    del encoder, projector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed, max_diff, mean_cos_sim
