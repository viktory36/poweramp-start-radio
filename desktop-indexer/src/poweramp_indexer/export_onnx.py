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

    The MuQ-MuLan model's forward() iterates batch items serially and
    internally splits to 10s clips via _get_all_clips(). For ONNX we
    bypass that and call the audio tower directly with a single 10s clip.
    """

    def __init__(self, mulan_model):
        super().__init__()
        # Extract the audio components from MuQMuLan
        # The model has: audio_encoder (Wav2Vec2ConformerEncoder), audio_projection
        self.mulan = mulan_model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: [1, 240000] raw waveform at 24kHz (10s clip)
        Returns:
            [1, 512] L2-normalized audio embedding
        """
        # MuQMuLan.forward calls _get_embedding_from_data for audio
        # which calls _get_all_clips to split into 10s, then encodes each.
        # We pass a single 10s clip, so this is equivalent.
        audio_embeds = self.mulan(wavs=wav)  # [1, 512], already L2-normalized
        return audio_embeds


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

    logger.info("Loading MuQ-MuLan model...")
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval()

    if fp16:
        model = model.half()

    wrapper = MuLanAudioOnnxWrapper(model)
    wrapper.eval()

    # 10s clip at 24kHz
    dummy_wav = torch.randn(1, 240000)
    if fp16:
        dummy_wav = dummy_wav.half()

    logger.info(f"Exporting MuQ-MuLan to {output_path} (opset {opset}, fp16={fp16})...")

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
        return _export_mulan_split(model, output_path, fp16, opset)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported MuQ-MuLan: {size_mb:.1f} MB")
    return output_path


class MuLanPostMelWrapper(nn.Module):
    """Fallback wrapper that takes mel features instead of raw waveform.

    Used when torch.stft fails to export in ONNX. The mel spectrogram
    is computed outside the ONNX model (in Python preprocessing or on Android).
    """

    def __init__(self, mulan_model):
        super().__init__()
        self.mulan = mulan_model
        # Extract the components we need post-mel
        # MuQ-MuLan pipeline: wav -> MelSTFT -> normalize -> Conv2dSubsampling ->
        #   Wav2Vec2ConformerEncoder -> Transformer -> Linear(768->512) -> L2-norm
        # We'll try to capture everything after mel computation

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_features: Output of MelSTFT, shape depends on model internals
        Returns:
            [1, 512] L2-normalized audio embedding
        """
        # This path is model-architecture-dependent; we access internal modules
        audio_encoder = self.mulan.audio_encoder
        audio_projection = self.mulan.audio_projection

        # Pass through encoder layers (skip mel computation)
        # The exact internal structure varies; this is a best-effort extraction
        hidden = audio_encoder(mel_features)
        if hasattr(hidden, 'last_hidden_state'):
            hidden = hidden.last_hidden_state
        # Pool and project
        pooled = hidden.mean(dim=1)  # [B, hidden_dim]
        projected = audio_projection(pooled)  # [B, 512]
        return F.normalize(projected, p=2, dim=-1)


def _export_mulan_split(model, output_path: Path, fp16: bool, opset: int):
    """Fallback: export MuQ-MuLan with mel spectrogram as input.

    If raw waveform export fails (torch.stft ONNX issues), we split the model:
    mel spectrogram computed on the caller side, model takes mel features as input.
    """
    logger.info("Attempting split export (mel features as input)...")

    # Extract mel computation parameters for the caller
    # Run a dummy forward to capture mel output shape
    dummy_wav = torch.randn(1, 240000)
    if fp16:
        dummy_wav = dummy_wav.half()
        model = model.half()

    # Try to intercept after mel computation using hooks
    mel_output = {}

    def hook_fn(module, input, output):
        mel_output['shape'] = output.shape
        mel_output['tensor'] = output.detach()

    # Find the mel/stft layer
    mel_layer = None
    for name, module in model.named_modules():
        if 'mel' in name.lower() or 'stft' in name.lower():
            mel_layer = module
            break

    if mel_layer is None:
        # Try finding by class name
        for name, module in model.named_modules():
            class_name = type(module).__name__.lower()
            if 'mel' in class_name or 'stft' in class_name or 'spectrogram' in class_name:
                mel_layer = module
                break

    if mel_layer is None:
        raise RuntimeError(
            "Could not find mel/STFT layer in MuQ-MuLan model. "
            "Manual inspection of model architecture needed for ONNX export."
        )

    handle = mel_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(wavs=dummy_wav)
    handle.remove()

    mel_shape = mel_output['shape']
    logger.info(f"Captured mel output shape: {mel_shape}")

    # Now export the post-mel part
    wrapper = MuLanPostMelWrapper(model)
    wrapper.eval()

    dummy_mel = mel_output['tensor']

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

    # Save mel parameters as a companion file so Android can replicate
    mel_params_path = output_path.with_suffix(".mel_params.json")
    import json
    params = {
        "input_type": "mel_features",
        "mel_shape": list(mel_shape),
        "sample_rate": 24000,
        "clip_duration_s": 10,
        "note": "MuQ-MuLan ONNX export with mel preprocessing split out. "
                "Caller must compute mel spectrogram matching the model's MelSTFT layer.",
    }
    with open(mel_params_path, 'w') as f:
        json.dump(params, f, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported MuQ-MuLan (post-mel): {size_mb:.1f} MB")
    logger.info(f"Mel parameters saved to: {mel_params_path}")
    return output_path


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

    wrapper = FlamingoOnnxWrapper(encoder, projector)
    wrapper.eval()

    # Dummy inputs: single 30s chunk
    batch_size = 1
    dummy_mel = torch.randn(batch_size, 128, 3000)
    dummy_mask = torch.ones(batch_size, 3000, dtype=torch.long)
    dummy_times = torch.arange(750, dtype=torch.float32).unsqueeze(0) * FRAME_DURATION_S

    if fp16:
        dummy_mel = dummy_mel.half()
        dummy_times = dummy_times.half()

    logger.info(f"Exporting Flamingo to {output_path} (opset {opset}, fp16={fp16})...")

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

    Generates random audio and compares ONNX Runtime output to PyTorch output.
    """
    import onnxruntime as ort
    from muq import MuQMuLan

    logger.info(f"Verifying MuQ-MuLan ONNX: {onnx_path}")

    # Load PyTorch model
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval()

    # Load ONNX model
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    input_dtype = np.float16 if "float16" in sess.get_inputs()[0].type else np.float32

    if input_dtype == np.float16:
        model = model.half()

    max_diff = 0.0
    all_cos_sims = []

    for i in range(num_tracks):
        # Random 10s audio clip
        wav = torch.randn(1, 240000)
        if input_dtype == np.float16:
            wav = wav.half()

        # PyTorch reference
        with torch.no_grad():
            ref_emb = model(wavs=wav).cpu().numpy()

        # ONNX inference
        onnx_emb = sess.run(None, {input_name: wav.numpy()})[0]

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
