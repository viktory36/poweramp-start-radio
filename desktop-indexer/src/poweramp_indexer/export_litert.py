"""Export MuQ-MuLan and Music Flamingo models to TFLite via litert-torch.

Converts the post-mel portions of each model to .tflite format for
on-device inference with LiteRT + QNN NPU acceleration.

Usage:
    python -m poweramp_indexer.export_litert mulan
    python -m poweramp_indexer.export_litert flamingo
    python -m poweramp_indexer.export_litert all
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Output directory for .tflite models
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# MuQ-MuLan mel spectrogram parameters
MULAN_MEL_PARAMS = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
    "norm_mean": 6.768444971712967,
    "norm_std": 18.417922652295623,
    "clip_duration_s": 10,
    "mel_frames": 1000,  # 10s * 24000 / 240 = 1000 (after [:-1] trim)
}


class MuLanPostMelWrapper(nn.Module):
    """Wraps the MuQ-MuLan audio pipeline after mel extraction.

    Input:  normalized mel spectrogram [batch, 128, 1000]
            (normalized = (raw_mel_db - 6.7684) / 18.4179)
    Output: L2-normalized embedding [batch, 512]

    Pipeline: Conv2dSubsampling → Conformer → proj(1024→768) →
              Transformer(12 layers) → mean pool → Linear(768→512) → L2-norm
    """

    def __init__(self, mulan_model):
        super().__init__()
        # Extract the components from the loaded model
        mulan_inner = mulan_model.mulan  # MuLanModel

        # Audio tower: AudioSpectrogramTransformerPretrained
        audio_tower = mulan_inner.audio

        # MuQ backbone
        muq = audio_tower.model  # MuQ instance
        muq_model = muq.model   # MuQModel instance

        # Stage 1: Conv2dSubsampling [batch, 128, 1000] → [batch, 250, 1024]
        self.conv = muq_model.conv

        # Stage 2: Conformer encoder (12 layers)
        self.conformer = muq_model.conformer

        # Stage 3: Layer selection index
        self.use_layer_idx = audio_tower.use_layer_idx  # -1 (last layer)

        # Stage 4: Projection 1024 → 768
        self.proj = audio_tower.proj

        # Stage 5: Post-conformer Transformer (12 layers, 768-dim)
        self.transformer = audio_tower.transformer

        # Stage 6: Projection 768 → 512
        self.audio_to_latents = mulan_inner.audio_to_latents

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Normalized mel spectrogram [batch, 128, 1000]
        Returns:
            L2-normalized embedding [batch, 512]
        """
        # Conv2dSubsampling: [batch, 128, 1000] → [batch, 250, 1024]
        x = self.conv(mel)

        # Conformer: [batch, 250, 1024] → hidden_states
        out = self.conformer(x, output_hidden_states=True)
        hidden_states = out["hidden_states"]
        x = hidden_states[self.use_layer_idx]  # last layer: [batch, 250, 1024]

        # Project: [batch, 250, 1024] → [batch, 250, 768]
        x = self.proj(x)

        # Transformer: [batch, 250, 768] → [batch, 250, 768]
        x, _ = self.transformer(x, return_all_layers=True)

        # Mean pool over time: [batch, 250, 768] → [batch, 768]
        x = x.mean(dim=-2)

        # Project to latent: [batch, 768] → [batch, 512]
        x = self.audio_to_latents(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)

        return x


def load_mulan_model():
    """Load the MuQ-MuLan model from HuggingFace cache."""
    from muq import MuQMuLan

    logger.info("Loading MuQ-MuLan model...")
    model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    model.eval()
    logger.info("MuQ-MuLan loaded.")
    return model


def validate_wrapper_vs_pytorch(mulan_model, wrapper, device="cpu"):
    """Validate that the wrapper produces identical output to the full model.

    Uses a synthetic mel spectrogram (not real audio) to test the post-mel
    pathway only. Both should produce identical results since they share
    the same weights.
    """
    wrapper.eval().to(device)

    # Create a fake normalized mel input [1, 128, 1000]
    torch.manual_seed(42)
    fake_mel = torch.randn(1, 128, MULAN_MEL_PARAMS["mel_frames"], device=device)

    with torch.no_grad():
        wrapper_out = wrapper(fake_mel)

    logger.info(f"Wrapper output shape: {wrapper_out.shape}")
    logger.info(f"Wrapper output norm: {torch.norm(wrapper_out).item():.6f}")
    logger.info(f"Wrapper output[:5]: {wrapper_out[0, :5].tolist()}")

    return wrapper_out


def convert_mulan(output_dir: Path):
    """Convert MuQ-MuLan post-mel model to TFLite."""
    import litert_torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full model
    mulan_model = load_mulan_model()

    # Create post-mel wrapper
    logger.info("Creating PostMelWrapper...")
    wrapper = MuLanPostMelWrapper(mulan_model)
    wrapper.eval()

    # Validate wrapper
    logger.info("Validating wrapper output...")
    validate_wrapper_vs_pytorch(mulan_model, wrapper)

    # Free full model to save memory (wrapper has references to submodules)
    del mulan_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sample input for tracing
    sample_mel = torch.randn(1, 128, MULAN_MEL_PARAMS["mel_frames"])

    # Convert to TFLite
    logger.info("Converting MuQ-MuLan to TFLite via litert_torch...")
    t0 = time.perf_counter()
    tflite_model = litert_torch.convert(wrapper, sample_args=(sample_mel,))
    t_convert = time.perf_counter() - t0
    logger.info(f"Conversion took {t_convert:.1f}s")

    # Export
    output_path = output_dir / "mulan_audio.tflite"
    tflite_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Validate TFLite output matches PyTorch wrapper
    logger.info("Validating TFLite output vs PyTorch...")
    with torch.no_grad():
        pytorch_out = wrapper(sample_mel).numpy()

    tflite_out = tflite_model(sample_mel)
    if hasattr(tflite_out, 'numpy'):
        tflite_out = tflite_out.numpy() if hasattr(tflite_out, 'numpy') else np.array(tflite_out)

    cosine_sim = np.dot(pytorch_out.flatten(), tflite_out.flatten()) / (
        np.linalg.norm(pytorch_out) * np.linalg.norm(tflite_out)
    )
    max_diff = np.max(np.abs(pytorch_out - tflite_out))
    logger.info(f"TFLite vs PyTorch: cosine_sim={cosine_sim:.6f}, max_diff={max_diff:.6f}")

    # Save mel params sidecar
    import json
    params_path = output_dir / "mulan_audio.mel_params.json"
    with open(params_path, "w") as f:
        json.dump(MULAN_MEL_PARAMS, f, indent=2)
    logger.info(f"Saved mel params: {params_path}")

    return output_path


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions by 90 degrees."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb_f32(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings using float32 (not float64 like the original).

    TFLite doesn't support float64, so we use float32 for intermediate
    computation. The precision difference is negligible for our use case.
    """
    ori_dtype = t.dtype
    t = t.float()  # float32 instead of float64
    freqs = freqs.float()

    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]  # empty slice (start_index=0)
    t_rot = t[..., :rot_dim]
    t_right = t[..., rot_dim:]

    t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_left, t_rot, t_right), dim=-1).to(ori_dtype)


class FlamingoEncoderWrapper(nn.Module):
    """Wraps Music Flamingo encoder for TFLite export.

    Bypasses create_bidirectional_mask (which uses dynamic assertion ops
    incompatible with TFLite) by passing None as the attention mask.
    This is valid because we always process full 30s chunks with no padding.

    Also replaces float64 rotary embedding computation with float32.

    Input:  mel features [batch, 128, 3000] (from WhisperFeatureExtractor)
            audio_times [batch, 750] (absolute timestamps per post-pool frame)
    Output: hidden states [batch, 750, 1280]
    """

    def __init__(self, encoder):
        super().__init__()
        # Extract submodules to avoid calling encoder.forward() which
        # triggers create_bidirectional_mask
        self.conv1 = encoder.conv1
        self.conv2 = encoder.conv2
        self.embed_positions = encoder.embed_positions
        self.layers = encoder.layers
        self.avg_pooler = encoder.avg_pooler
        self.layer_norm = encoder.layer_norm

        # Pre-compute base RoTE frequencies for batch=1, seq_len=750
        # This avoids caching branches in RotaryEmbedding.get_axial_freqs
        # during torch.export tracing
        with torch.no_grad():
            self.register_buffer(
                "base_freqs",
                encoder.pos_emb.get_axial_freqs(1, 750).detach(),
            )

    def forward(
        self,
        input_features: torch.Tensor,
        audio_times: torch.Tensor,
    ) -> torch.Tensor:
        # Conv front-end: [batch, 128, 3000] → [batch, 1280, 1500]
        inputs_embeds = F.gelu(self.conv1(input_features))
        inputs_embeds = F.gelu(self.conv2(inputs_embeds))
        # [batch, 1280, 1500] → [batch, 1500, 1280]
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        # Add sinusoidal position embeddings
        hidden_states = inputs_embeds + self.embed_positions.weight

        # Transformer stack — no attention mask (full chunk, no padding)
        for layer in self.layers:
            hidden_states = layer(hidden_states, None)[0]

        # AvgPool (time/2): [batch, 1500, 1280] → [batch, 750, 1280]
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states).permute(0, 2, 1)
        hidden_states = self.layer_norm(hidden_states)

        # Rotary Time Embeddings (RoTE)
        # base_freqs: [1, 750, rot_dim] — pre-computed in __init__
        angle = -audio_times * (2.0 * math.pi)  # [batch, 750]
        angle_expanded = angle.unsqueeze(2).expand(
            audio_times.shape[0],
            hidden_states.shape[-2],
            self.base_freqs.shape[-1],
        )
        freqs = self.base_freqs * angle_expanded
        hidden_states = _apply_rotary_emb_f32(freqs, hidden_states)

        return hidden_states


class FlamingoProjectorWrapper(nn.Module):
    """Wraps the AudioProjector MLP for TFLite export.

    Input:  hidden states [batch, 750, 1280]
    Output: projected [batch, 750, 3584]
    """

    def __init__(self, projector):
        super().__init__()
        self.projector = projector

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_states)


def load_flamingo_model():
    """Load Music Flamingo encoder and projector in FP32."""
    from safetensors.torch import load_file
    from transformers.models.musicflamingo.modeling_musicflamingo import (
        MusicFlamingoEncoder,
    )

    from .embeddings_flamingo import (
        AudioProjector,
        get_flamingo_encoder_path,
    )

    encoder_path = get_flamingo_encoder_path()
    logger.info(f"Loading Music Flamingo encoder from '{encoder_path}'...")

    # Force FP32 — the safetensors are stored in BF16 but litert-torch
    # needs FP32 for conversion (it handles precision internally)
    encoder = MusicFlamingoEncoder.from_pretrained(
        str(encoder_path), torch_dtype=torch.float32
    )
    encoder.float()  # Ensure all params are FP32
    encoder.eval()

    projector = None
    projector_path = Path(encoder_path) / "projector.safetensors"
    if projector_path.exists():
        logger.info(f"Loading projector from '{projector_path}'...")
        projector = AudioProjector()
        state_dict = load_file(str(projector_path), device="cpu")
        # Convert BF16/FP16 weights to FP32
        state_dict = {k: v.float() for k, v in state_dict.items()}
        projector.load_state_dict(state_dict)
        projector.eval()

    return encoder, projector


def convert_flamingo(output_dir: Path):
    """Convert Music Flamingo encoder + projector to TFLite."""
    import litert_torch

    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, projector = load_flamingo_model()

    # --- Encoder ---
    logger.info("Converting Flamingo encoder to TFLite...")
    encoder_wrapper = FlamingoEncoderWrapper(encoder)
    encoder_wrapper.eval()

    # Sample inputs (batch=1, 30s chunk)
    # No mask input — wrapper assumes full 30s with no padding
    sample_mel = torch.randn(1, 128, 3000)
    sample_times = torch.arange(750, dtype=torch.float32).unsqueeze(0) * 0.04

    # Validate wrapper output vs original encoder first
    logger.info("Validating wrapper vs original encoder...")
    sample_mask = torch.ones(1, 3000, dtype=torch.long)
    with torch.no_grad():
        orig_out = encoder(
            sample_mel,
            input_features_mask=sample_mask,
            audio_times=sample_times,
        ).last_hidden_state.numpy()
        wrapper_out = encoder_wrapper(sample_mel, sample_times).numpy()
    wrapper_cos = np.dot(orig_out.flatten(), wrapper_out.flatten()) / (
        np.linalg.norm(orig_out) * np.linalg.norm(wrapper_out)
    )
    logger.info(f"Wrapper vs original encoder: cosine_sim={wrapper_cos:.6f}")

    # Free the original encoder to save memory
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    tflite_encoder = litert_torch.convert(
        encoder_wrapper,
        sample_args=(sample_mel, sample_times),
    )
    t_convert = time.perf_counter() - t0
    logger.info(f"Encoder conversion took {t_convert:.1f}s")

    encoder_path = output_dir / "flamingo_encoder.tflite"
    tflite_encoder.export(str(encoder_path))
    size_mb = encoder_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {encoder_path} ({size_mb:.1f} MB)")

    # Validate TFLite output
    with torch.no_grad():
        pytorch_enc_out = encoder_wrapper(sample_mel, sample_times).numpy()
    tflite_enc_out = tflite_encoder(sample_mel, sample_times)
    if hasattr(tflite_enc_out, 'numpy'):
        tflite_enc_out = tflite_enc_out.numpy()
    cos_sim = np.dot(pytorch_enc_out.flatten(), tflite_enc_out.flatten()) / (
        np.linalg.norm(pytorch_enc_out) * np.linalg.norm(tflite_enc_out)
    )
    logger.info(f"Encoder TFLite vs PyTorch: cosine_sim={cos_sim:.6f}")

    # --- Projector ---
    if projector is not None:
        logger.info("Converting Flamingo projector to TFLite...")
        proj_wrapper = FlamingoProjectorWrapper(projector)
        proj_wrapper.eval()

        sample_hidden = torch.randn(1, 750, 1280)

        t0 = time.perf_counter()
        tflite_projector = litert_torch.convert(
            proj_wrapper,
            sample_args=(sample_hidden,),
        )
        t_convert = time.perf_counter() - t0
        logger.info(f"Projector conversion took {t_convert:.1f}s")

        projector_path = output_dir / "flamingo_projector.tflite"
        tflite_projector.export(str(projector_path))
        size_mb = projector_path.stat().st_size / 1024 / 1024
        logger.info(f"Saved: {projector_path} ({size_mb:.1f} MB)")

        # Validate projector
        with torch.no_grad():
            pytorch_proj_out = proj_wrapper(sample_hidden).numpy()
        tflite_proj_out = tflite_projector(sample_hidden)
        if hasattr(tflite_proj_out, 'numpy'):
            tflite_proj_out = tflite_proj_out.numpy()
        cos_sim_proj = np.dot(pytorch_proj_out.flatten(), tflite_proj_out.flatten()) / (
            np.linalg.norm(pytorch_proj_out) * np.linalg.norm(tflite_proj_out)
        )
        logger.info(f"Projector TFLite vs PyTorch: cosine_sim={cos_sim_proj:.6f}")

    return encoder_path


def main():
    parser = argparse.ArgumentParser(description="Export models to TFLite")
    parser.add_argument("model", choices=["mulan", "flamingo", "all"])
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.model in ("mulan", "all"):
        convert_mulan(args.output_dir)

    if args.model in ("flamingo", "all"):
        convert_flamingo(args.output_dir)


if __name__ == "__main__":
    main()
