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


def _patch_flamingo_rotary_for_onnx():
    """Monkey-patch Flamingo's apply_rotary_emb to be ONNX-exportable.

    The original code does:
        return torch.cat((t_left, t, t_right), dim=-1).to(ori_dtype)

    This fails in PyTorch's legacy ONNX export because the three slices may have
    different dtypes during FP16 tracing. Fix: cast all parts to the same dtype
    BEFORE the cat operation.
    """
    try:
        import transformers.models.musicflamingo.modeling_musicflamingo as mf

        original_apply_rotary = mf.apply_rotary_emb

        def patched_apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
            if t.ndim == 3:
                seq_len = t.shape[seq_dim]
                freqs = freqs[-seq_len:]

            rot_dim = freqs.shape[-1]
            end_index = start_index + rot_dim

            assert rot_dim <= t.shape[-1], (
                f"feature dimension {t.shape[-1]} is not of sufficient size "
                f"to rotate in all the positions {rot_dim}"
            )

            ori_dtype = t.dtype
            t_left = t[..., :start_index]
            t_mid = t[..., start_index:end_index]
            t_right = t[..., end_index:]

            # Rotate the middle portion
            def rotate_half(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)

            t_mid = (t_mid * freqs.cos() * scale) + (rotate_half(t_mid) * freqs.sin() * scale)

            # Cast all parts to same dtype BEFORE cat (fixes ONNX export)
            t_left = t_left.to(ori_dtype)
            t_mid = t_mid.to(ori_dtype)
            t_right = t_right.to(ori_dtype)
            return torch.cat((t_left, t_mid, t_right), dim=-1)

        mf.apply_rotary_emb = patched_apply_rotary_emb
        logger.info("Patched Flamingo apply_rotary_emb for ONNX compatibility")
        return True
    except Exception as e:
        logger.warning(f"Could not patch Flamingo rotary embeddings: {e}")
        return False


def _convert_onnx_to_fp16(model_path: Path):
    """Post-export FP32→FP16 conversion for ONNX models.

    Exporting a PyTorch FP16 model directly often produces mixed-dtype graphs
    (e.g. LayerNorm weight/bias stay FP32 while activations are FP16), which
    ONNX Runtime rejects. Instead: export FP32, then convert everything to FP16.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    logger.info(f"Converting {model_path.name} to FP16...")
    model = onnx.load(str(model_path))

    # Convert initializers (weights/biases)
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            new = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new)

    # Convert Constant node values
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t).astype(np.float16)
                    new = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new)

    # Update type annotations (FLOAT → FLOAT16 only; INT64 etc stay unchanged)
    for x in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if x.type.tensor_type.elem_type == TensorProto.FLOAT:
            x.type.tensor_type.elem_type = TensorProto.FLOAT16

    onnx.save(model, str(model_path))
    size_mb = model_path.stat().st_size / 1024 / 1024
    logger.info(f"FP16 conversion done: {size_mb:.1f} MB")


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


class MuLanPostMelWrapper(nn.Module):
    """Fallback wrapper that takes normalized mel features as input.

    Used when torch.stft fails to export in ONNX. The mel spectrogram +
    normalization are computed outside ONNX (Python / Android).

    Architecture (from inspecting muq source):
        MuQModel.get_predictions(wav):
          1. mel = preprocessing(wav, ["melspec_2048"])   ← MelSTFT (SKIPPED)
          2. mel = normalize(mel)                         ← mean/std (SKIPPED)
          3. encoder(mel["melspec_2048"])                  ← Conv2d + Conformer
          → returns (logits, hidden_states)
        AudioSpectrogramTransformerPretrained:
          → proj(hidden_states[layer_idx]) + transformer + mean pool
        MuLan:
          → audio_to_latents + L2-norm → 512d

    We monkey-patch get_predictions to skip steps 1-2, accepting normalized
    mel features directly.
    """

    def __init__(self, mulan_inner):
        super().__init__()
        self.mulan_inner = mulan_inner

        # Monkey-patch MuQModel.get_predictions to skip preprocessing
        muq_model = self.mulan_inner.audio.model.model  # MuQModel
        original_encoder = muq_model.encoder

        def patched_get_predictions(x, *, mask=None, attention_mask=None,
                                    return_new_mask=False, is_features_only=False):
            # x is already normalized mel features — skip preprocessing+normalize
            logits, hidden_emb, new_mask = original_encoder(
                x, attention_mask=attention_mask, is_features_only=is_features_only)
            if return_new_mask:
                return logits, hidden_emb, mask if new_mask is None else new_mask
            return logits, hidden_emb

        muq_model.get_predictions = patched_get_predictions

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_features: Normalized mel spectrogram [1, 128, 1000]
                          (after MelSTFT + mean/std normalization)
        Returns:
            [1, 512] L2-normalized audio embedding
        """
        return self.mulan_inner.get_audio_latents(mel_features)


def _export_mulan_split(model, output_path: Path, fp16: bool, opset: int, device: str = "cpu"):
    """Fallback: export MuQ-MuLan with mel spectrogram as input.

    torch.stft can't export to ONNX, so we split the model:
    - Mel spectrogram + normalization computed outside ONNX (Python / Android)
    - ONNX model takes normalized mel features [1, 128, 1000] → [1, 512] embedding

    MuQ-MuLan mel parameters (from torchaudio.MelSpectrogram):
      n_fft=2048, hop_length=240, win_length=2048, n_mels=128, sample_rate=24000
      power=2.0 (power spectrogram), normalized=False, f_min=0, f_max=None (Nyquist)
    Then instance-normalized: (mel - 6.7684) / 18.4179 (from model's stat dict)
    """
    logger.info("Attempting split export (mel features as input)...")

    mulan_inner = model.mulan_module
    muq_model = mulan_inner.audio.model.model  # MuQModel

    # Step 1: Compute mel features via PyTorch for the dummy input
    dummy_wav = torch.randn(1, 240000, device=device)
    if fp16:
        dummy_wav = dummy_wav.half()

    with torch.no_grad():
        mel_dict = muq_model.preprocessing(dummy_wav, features=["melspec_2048"])
        mel_dict = muq_model.normalize(mel_dict)
        dummy_mel = mel_dict["melspec_2048"]  # [1, 128, 1000]

    logger.info(f"Mel feature shape: {dummy_mel.shape}")

    # Extract mel parameters and normalization stats
    mel_stft = muq_model.preprocessor_melspec_2048
    torchaudio_mel = mel_stft.mel_stft  # torchaudio.transforms.MelSpectrogram
    mel_params = {}
    for attr in ['n_fft', 'hop_length', 'win_length', 'n_mels', 'sample_rate',
                 'f_min', 'f_max', 'power', 'normalized']:
        if hasattr(torchaudio_mel, attr):
            val = getattr(torchaudio_mel, attr)
            mel_params[attr] = float(val) if isinstance(val, (int, float)) else val

    # Normalization stats
    mel_params['norm_mean'] = float(muq_model.stat['melspec_2048_mean'])
    mel_params['norm_std'] = float(muq_model.stat['melspec_2048_std'])
    mel_params['is_db'] = mel_stft.is_db if hasattr(mel_stft, 'is_db') else False
    logger.info(f"Mel params: {mel_params}")

    # Step 2: Monkey-patch and export
    wrapper = MuLanPostMelWrapper(mulan_inner)
    wrapper.eval()

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
        "mel_shape": [int(x) for x in dummy_mel.shape],
        "sample_rate": 24000,
        "clip_duration_s": 10,
        **mel_params,
    }
    with open(mel_params_path, 'w') as f:
        json.dump(params, f, indent=2)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Exported MuQ-MuLan (post-mel): {size_mb:.1f} MB")
    logger.info(f"Mel parameters saved to: {mel_params_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Patch RoTE to fix torch.cat dtype mismatch in legacy ONNX export
    _patch_flamingo_rotary_for_onnx()

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

    # Export in FP32 to avoid mixed-dtype issues (LayerNorm weight/bias vs activations).
    # The encoder loads with BF16 weights — cast to FP32 for clean tracing.
    # Convert to FP16 post-export for consistent types and smaller model.
    encoder = encoder.float().to(device)
    if projector is not None:
        projector = projector.float().to(device)

    wrapper = FlamingoOnnxWrapper(encoder, projector)
    wrapper.eval()

    # Dummy inputs: single 30s chunk (FP32 for tracing)
    batch_size = 1
    dummy_mel = torch.randn(batch_size, 128, 3000, device=device)
    dummy_mask = torch.ones(batch_size, 3000, dtype=torch.long, device=device)
    dummy_times = (torch.arange(750, dtype=torch.float32, device=device).unsqueeze(0)
                   * FRAME_DURATION_S)

    logger.info(f"Exporting Flamingo to {output_path} (opset {opset}, fp16={fp16}, device={device})...")

    # Try dynamo-based export first (handles RoTE/complex ops that fail with legacy export)
    # Then fall back to legacy TorchScript export
    exported = False

    if hasattr(torch.onnx, 'dynamo_export'):
        try:
            logger.info("Trying dynamo_export (handles complex ops better)...")
            export_output = torch.onnx.dynamo_export(
                wrapper, dummy_mel, dummy_mask, dummy_times
            )
            export_output.save(str(output_path))
            exported = True
            logger.info("dynamo_export succeeded")
        except Exception as e:
            logger.warning(f"dynamo_export failed: {e}, trying legacy export...")

    if not exported:
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
    logger.info(f"Exported Flamingo (FP32): {size_mb:.1f} MB")

    if fp16:
        _convert_onnx_to_fp16(output_path)

    del encoder, projector, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def verify_mulan_onnx(onnx_path: Path, num_tracks: int = 10, tolerance: float = 1e-3):
    """Verify MuQ-MuLan ONNX export against PyTorch reference.

    Handles both full-model export (input=wav) and split export (input=mel_features).
    For split models, generates mel features via PyTorch preprocessing.
    """
    import onnxruntime as ort
    from muq import MuQMuLan

    logger.info(f"Verifying MuQ-MuLan ONNX: {onnx_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model = model.to(device)

    logger.info(f"  Input: {input_name} ({'split/mel' if is_split_model else 'full/wav'}), "
                f"dtype: {'fp16' if input_dtype == np.float16 else 'fp32'}, device: {device}")

    max_diff = 0.0
    all_cos_sims = []

    for i in range(num_tracks):
        # Random 10s audio clip
        wav = torch.randn(1, 240000, device=device)
        if input_dtype == np.float16:
            wav = wav.half()

        with torch.no_grad():
            # PyTorch reference — full path through get_audio_latents
            ref_emb = model.mulan_module.get_audio_latents(wav).cpu().numpy()

            if is_split_model:
                # For split model: compute mel + normalize via PyTorch
                muq_model = model.mulan_module.audio.model.model
                mel_dict = muq_model.preprocessing(wav, features=["melspec_2048"])
                mel_dict = muq_model.normalize(mel_dict)
                onnx_input = mel_dict["melspec_2048"].cpu().numpy()
            else:
                onnx_input = wav.cpu().numpy()

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    encoder = encoder.to(device)
    if projector is not None:
        projector = projector.to(device)

    logger.info(f"  dtype: {'fp16' if input_dtype == np.float16 else 'fp32'}, device: {device}")

    max_diff = 0.0
    all_cos_sims = []

    for i in range(num_tracks):
        # Random mel spectrogram (30s chunk)
        mel = torch.randn(1, 128, 3000, device=device)
        mask = torch.ones(1, 3000, dtype=torch.long, device=device)
        times = (torch.arange(750, dtype=torch.float32, device=device).unsqueeze(0)
                 * FRAME_DURATION_S)

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

        # ONNX inference (CPU)
        feed = {
            "input_features": mel.cpu().numpy(),
            "input_features_mask": mask.cpu().numpy(),
            "audio_times": times.cpu().numpy(),
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
