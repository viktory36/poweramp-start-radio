"""Export CLaMP3 models (MERT, audio encoder, text encoder) to TFLite.

Converts each model component to .tflite format for on-device inference
with LiteRT GPU acceleration.

Usage:
    python -m poweramp_indexer.export_litert mert
    python -m poweramp_indexer.export_litert clamp3_audio
    python -m poweramp_indexer.export_litert clamp3_text
    python -m poweramp_indexer.export_litert tokenizer
    python -m poweramp_indexer.export_litert all
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Output directory for .tflite models
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# MERT constants
MERT_SR = 24000
MERT_WINDOW_SAMPLES = 5 * MERT_SR  # 120,000 samples = 5 seconds


class ChannelNorm(nn.Module):
    """GPU-friendly replacement for GroupNorm(C, C) where num_groups == num_channels.

    PyTorch's GroupNorm generates GATHER_ND ops during TFLite export
    (for indexing the affine params per-group). Since MERT uses
    GroupNorm(512, 512) — one channel per group — we replace it with
    simple per-channel mean/var that exports as GPU-native arithmetic.
    """

    def __init__(self, group_norm: nn.GroupNorm):
        super().__init__()
        self.weight = group_norm.weight  # [C]
        self.bias = group_norm.bias      # [C]
        self.eps = group_norm.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, C, T]
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, correction=0)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.unsqueeze(0).unsqueeze(2) + self.bias.unsqueeze(0).unsqueeze(2)


class MertWrapper(nn.Module):
    """Wraps MERT-v1-95M for TFLite export.

    Input:  raw waveform [1, 120000] (5 seconds at 24kHz)
    Output: mean-pooled features [1, 768] (averaged over all layers and time)

    The wrapper pre-computes the Wav2Vec2 feature extraction (group norm +
    conv feature extraction) and processes through all 12 transformer layers.
    The final output is the mean over all hidden states and time steps.
    """

    def __init__(self, mert_model):
        super().__init__()
        self.mert = mert_model

        # Remove weight norm from positional conv to make it export-compatible
        # (weight_norm uses parametrizations that torch.export can't handle)
        conv = self.mert.encoder.pos_conv_embed.conv
        try:
            nn.utils.parametrize.remove_parametrizations(conv, 'weight')
            logger.info("Removed weight parametrization from pos_conv_embed.conv")
        except Exception:
            try:
                nn.utils.remove_weight_norm(conv)
                logger.info("Removed weight_norm from pos_conv_embed.conv")
            except Exception:
                logger.warning("Could not remove weight_norm — may fail during export")

        # Replace GroupNorm(512,512) with GPU-friendly ChannelNorm
        # (GroupNorm generates GATHER_ND ops that the GPU delegate rejects)
        layer0 = self.mert.feature_extractor.conv_layers[0]
        layer0.layer_norm = ChannelNorm(layer0.layer_norm)
        logger.info("Replaced GroupNorm with ChannelNorm for GPU compatibility")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch, 1, 120000, 1] float32 raw audio at 24kHz
                      (4D for GPU delegate buffer compatibility — GPU delegates
                       require [B,H,W,C] tensor layout for buffer allocation)
        Returns:
            [batch, 768] mean-pooled features
        """
        # Reshape 4D GPU-compatible input to 2D for MERT
        waveform = waveform.reshape(waveform.shape[0], -1)  # [B,1,T,1] → [B,T]
        out = self.mert(waveform, output_hidden_states=True)
        # Accumulate mean over layers without torch.stack.
        # torch.stack generates GATHER_ND which the GPU delegate can't run.
        # Unrolled sum is traced as 13 static additions — all GPU-native.
        num_layers = len(out.hidden_states)
        acc = out.hidden_states[0].mean(dim=1)  # [batch, 768]
        for h in out.hidden_states[1:]:
            acc = acc + h.mean(dim=1)
        return acc / num_layers


def convert_mert(output_dir: Path):
    """Convert MERT-v1-95M to TFLite."""
    import litert_torch

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MERT-v1-95M...")
    from transformers import AutoModel
    mert = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert.eval()

    wrapper = MertWrapper(mert)
    wrapper.eval()

    # Validate wrapper produces reasonable output
    # 4D input: [B, 1, T, 1] — GPU delegate requires [B,H,W,C] for buffer allocation
    logger.info("Validating wrapper...")
    sample_wav = torch.randn(1, 1, MERT_WINDOW_SAMPLES, 1)
    with torch.no_grad():
        ref_out = wrapper(sample_wav)
    logger.info(f"Wrapper output shape: {ref_out.shape}, norm: {ref_out.norm().item():.4f}")

    # Convert
    logger.info("Converting MERT to TFLite...")
    t0 = time.perf_counter()
    tflite_model = litert_torch.convert(wrapper, sample_args=(sample_wav,))
    logger.info(f"Conversion took {time.perf_counter() - t0:.1f}s")

    output_path = output_dir / "mert.tflite"
    tflite_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Validate TFLite output
    logger.info("Validating TFLite vs PyTorch...")
    with torch.no_grad():
        pytorch_out = wrapper(sample_wav).numpy()

    tflite_out = tflite_model(sample_wav)
    if hasattr(tflite_out, 'numpy'):
        tflite_out = tflite_out.numpy()
    else:
        tflite_out = np.array(tflite_out)

    cosine = np.dot(pytorch_out.flatten(), tflite_out.flatten()) / (
        np.linalg.norm(pytorch_out) * np.linalg.norm(tflite_out)
    )
    max_diff = np.max(np.abs(pytorch_out - tflite_out))
    logger.info(f"TFLite vs PyTorch: cosine={cosine:.6f}, max_diff={max_diff:.6f}")

    return output_path


class CLaMP3AudioWrapper(nn.Module):
    """Wraps CLaMP3 audio encoder for TFLite export.

    Input:  audio_inputs [1, 128, 768] (MERT features, zero-padded)
            audio_masks  [1, 128] float32 (1=real, 0=pad)
    Output: [1, 768] projected features (not L2-normalized)

    GPU-safe mask handling: BERT's get_extended_attention_mask uses
    torch.finfo(float32).min (-3.4e38) which overflows to -inf in the GPU
    delegate's FP16 compute. For valid tokens: 0.0 * (-inf) = NaN (IEEE 754).
    We bypass BertModel.forward() and call embeddings + encoder directly,
    computing the additive mask ourselves with -1e4 (fits in FP16 safely).

    ADD with [B, 1, 1, S] broadcasts natively on GPU — no BROADCAST_TO needed.

    This is a single 128-frame window. The segmentation and weighted averaging
    of multi-segment tracks is done on the host (Kotlin/Python) side.
    """

    def __init__(self, encoder):
        super().__init__()
        self.bert_embeddings = encoder.audio_model.embeddings
        self.bert_encoder = encoder.audio_model.encoder
        self.audio_proj = encoder.audio_proj

    def forward(self, audio_inputs: torch.Tensor, audio_masks: torch.Tensor) -> torch.Tensor:
        # Run BERT embeddings (adds position + token_type + LayerNorm)
        hidden_states = self.bert_embeddings(inputs_embeds=audio_inputs)
        # Compute additive attention mask: 0 for valid, -1e4 for padded
        # -1e4 is safely within FP16 range (max ±65504), unlike finfo.min (-3.4e38 → -inf)
        additive_mask = (1.0 - audio_masks.float()) * (-1e4)  # [B, S]
        extended_mask = additive_mask[:, None, None, :]  # [B, 1, 1, S]
        # Run BERT encoder directly (bypassing get_extended_attention_mask)
        encoder_out = self.bert_encoder(hidden_states, attention_mask=extended_mask)
        features = encoder_out['last_hidden_state']
        # Masked mean pooling
        masks = audio_masks.unsqueeze(-1)  # [B, S, 1]
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1)
        return self.audio_proj(pooled)


def convert_clamp3_audio(output_dir: Path):
    """Convert CLaMP3 audio encoder to TFLite."""
    import litert_torch

    from .embeddings_clamp3 import (
        CLAMP3_WEIGHTS_FILENAME, CLaMP3AudioEncoder, MAX_AUDIO_LENGTH, AUDIO_HIDDEN_SIZE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    logger.info("Loading CLaMP3 audio encoder...")
    weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    encoder = CLaMP3AudioEncoder.from_clamp3_checkpoint(weights_path)

    wrapper = CLaMP3AudioWrapper(encoder)
    wrapper.eval()

    # Sample inputs — mask is 2D [B, S] (1=real, 0=pad)
    sample_inputs = torch.randn(1, MAX_AUDIO_LENGTH, AUDIO_HIDDEN_SIZE)
    sample_masks = torch.ones(1, MAX_AUDIO_LENGTH)

    # Validate
    logger.info("Validating wrapper...")
    with torch.no_grad():
        ref_out = wrapper(sample_inputs, sample_masks)
    logger.info(f"Output shape: {ref_out.shape}, norm: {ref_out.norm().item():.4f}")

    # Convert
    logger.info("Converting CLaMP3 audio encoder to TFLite...")
    t0 = time.perf_counter()
    tflite_model = litert_torch.convert(
        wrapper, sample_args=(sample_inputs, sample_masks)
    )
    logger.info(f"Conversion took {time.perf_counter() - t0:.1f}s")

    output_path = output_dir / "clamp3_audio.tflite"
    tflite_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Validate
    logger.info("Validating TFLite vs PyTorch...")
    with torch.no_grad():
        pytorch_out = wrapper(sample_inputs, sample_masks).numpy()

    tflite_out = tflite_model(sample_inputs, sample_masks)
    if hasattr(tflite_out, 'numpy'):
        tflite_out = tflite_out.numpy()
    else:
        tflite_out = np.array(tflite_out)

    cosine = np.dot(pytorch_out.flatten(), tflite_out.flatten()) / (
        np.linalg.norm(pytorch_out) * np.linalg.norm(tflite_out)
    )
    logger.info(f"TFLite vs PyTorch: cosine={cosine:.6f}")

    return output_path


class CLaMP3TextWrapper(nn.Module):
    """Wraps CLaMP3 text encoder for TFLite export.

    Input:  input_ids [1, 128] INT64 (XLM-RoBERTa token IDs)
            attention_mask [1, 128] INT64 (1=real, 0=pad)
    Output: [1, 768] projected features (not L2-normalized)

    GPU-safe mask handling: same as CLaMP3AudioWrapper — bypasses the model's
    get_extended_attention_mask (which uses finfo.min → NaN in FP16) and
    computes a -1e4 additive mask directly.

    Segmentation and weighted averaging for long texts is done on the host side.
    """

    SEQ_LEN = 128  # Static sequence length matching CLaMP3 config

    def __init__(self, text_encoder):
        super().__init__()
        self.roberta_embeddings = text_encoder.text_model.embeddings
        self.roberta_encoder = text_encoder.text_model.encoder
        self.text_proj = text_encoder.text_proj

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Run RoBERTa embeddings (word + position + token_type + LayerNorm)
        hidden_states = self.roberta_embeddings(input_ids=input_ids)
        # Compute GPU-safe additive mask (bypasses finfo.min)
        additive_mask = (1.0 - attention_mask.float()) * (-1e4)  # [B, S]
        extended_mask = additive_mask[:, None, None, :]  # [B, 1, 1, S]
        # Run encoder directly
        encoder_out = self.roberta_encoder(hidden_states, attention_mask=extended_mask)
        features = encoder_out['last_hidden_state']
        # Masked mean pooling
        masks = attention_mask.unsqueeze(-1).float()
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1).clamp(min=1e-10)
        return self.text_proj(pooled)


def convert_clamp3_text(output_dir: Path):
    """Convert CLaMP3 text encoder to TFLite."""
    import litert_torch

    from .embeddings_clamp3 import (
        CLAMP3_WEIGHTS_FILENAME, CLaMP3TextEncoder, MAX_TEXT_LENGTH,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    logger.info("Loading CLaMP3 text encoder...")
    weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    text_encoder = CLaMP3TextEncoder.from_clamp3_checkpoint(weights_path)

    wrapper = CLaMP3TextWrapper(text_encoder)
    wrapper.eval()

    # Sample inputs (padded to MAX_TEXT_LENGTH)
    sample_ids = torch.ones(1, MAX_TEXT_LENGTH, dtype=torch.long)
    sample_ids[0, :5] = torch.tensor([0, 82, 35593, 289, 2])  # BOS + "ethereal" + EOS
    sample_mask = torch.zeros(1, MAX_TEXT_LENGTH, dtype=torch.long)
    sample_mask[0, :5] = 1

    # Validate
    logger.info("Validating wrapper...")
    with torch.no_grad():
        ref_out = wrapper(sample_ids, sample_mask)
    logger.info(f"Output shape: {ref_out.shape}, norm: {ref_out.norm().item():.4f}")

    # Convert
    logger.info("Converting CLaMP3 text encoder to TFLite...")
    t0 = time.perf_counter()
    tflite_model = litert_torch.convert(
        wrapper, sample_args=(sample_ids, sample_mask)
    )
    logger.info(f"Conversion took {time.perf_counter() - t0:.1f}s")

    output_path = output_dir / "clamp3_text.tflite"
    tflite_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Validate
    logger.info("Validating TFLite vs PyTorch...")
    with torch.no_grad():
        pytorch_out = wrapper(sample_ids, sample_mask).numpy()

    tflite_out = tflite_model(sample_ids, sample_mask)
    if hasattr(tflite_out, 'numpy'):
        tflite_out = tflite_out.numpy()
    else:
        tflite_out = np.array(tflite_out)

    cosine = np.dot(pytorch_out.flatten(), tflite_out.flatten()) / (
        np.linalg.norm(pytorch_out) * np.linalg.norm(tflite_out)
    )
    max_diff = np.max(np.abs(pytorch_out - tflite_out))
    logger.info(f"TFLite vs PyTorch: cosine={cosine:.6f}, max_diff={max_diff:.6f}")

    return output_path


def export_tokenizer(output_dir: Path):
    """Export the XLM-RoBERTa tokenizer vocabulary for on-device use.

    Saves a JSON file with the Unigram vocabulary: {piece: [token_id, score]}.
    The score is the log probability used for Viterbi DP segmentation.
    """
    from tokenizers import Tokenizer as HFTokenizer
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading xlm-roberta-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    fast_tok = HFTokenizer.from_pretrained("xlm-roberta-base")
    tok_json = json.loads(fast_tok.to_str())
    unigram_vocab = tok_json["model"]["vocab"]
    logger.info(f"Loaded Unigram model: {len(unigram_vocab)} pieces")

    vocab = {}
    for idx, (piece, score) in enumerate(unigram_vocab):
        vocab[piece] = [idx, score]

    vocab_path = output_dir / "xlm_roberta_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    size_kb = vocab_path.stat().st_size / 1024
    logger.info(f"Saved vocabulary: {vocab_path} ({size_kb:.0f} KB)")

    # Validate
    test_queries = ["ethereal ambient", "psychedelic", "trance", "melancholic"]
    for test in test_queries:
        hf_ids = tokenizer.encode(test)
        hf_tokens = tokenizer.convert_ids_to_tokens(hf_ids)
        logger.info(f"Validate '{test}': {hf_tokens} ids={hf_ids}")

    return vocab_path


def main():
    parser = argparse.ArgumentParser(description="Export CLaMP3 models to TFLite")
    parser.add_argument(
        "model", choices=["mert", "clamp3_audio", "clamp3_text", "tokenizer", "all"]
    )
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.model in ("mert", "all"):
        convert_mert(args.output_dir)

    if args.model in ("clamp3_audio", "all"):
        convert_clamp3_audio(args.output_dir)

    if args.model in ("clamp3_text", "all"):
        convert_clamp3_text(args.output_dir)
        export_tokenizer(args.output_dir)

    if args.model == "tokenizer":
        export_tokenizer(args.output_dir)


if __name__ == "__main__":
    main()
