#!/usr/bin/env python3
"""Convert TFLite FP32 models to GPU-native FP16.

Converts all FP32 weight tensors to FP16 in-place (no DEQUANTIZE ops).
The LiteRT GPU delegate reads FP16 tensors natively — no CPU fallback needed.

This halves model size with essentially zero quality loss on GPU.
Note: these models are GPU-only; CPU inference would need the FP32 originals.

Usage:
    python convert_fp16.py <input.tflite> [output.tflite]

If no output path is given, writes to <input>_fp16.tflite.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as schema_fb


def convert_fp32_to_fp16(input_path: str, output_path: str | None = None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem.replace("_wo_wi8", "") + "_fp16.tflite"
        )
    else:
        output_path = Path(output_path)

    print(f"Loading {input_path} ({input_path.stat().st_size / 1e6:.0f} MB)")
    t0 = time.time()
    file_data = bytearray(input_path.read_bytes())
    print(f"  Read in {time.time() - t0:.1f}s")

    t0 = time.time()
    model_obj = schema_fb.Model.GetRootAs(file_data)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    print(f"  Parsed in {time.time() - t0:.1f}s")

    # Inline all offset-based buffers (large models store data externally)
    inlined = 0
    for buf_idx, buf in enumerate(model.buffers):
        if buf.data is None and buf.offset and buf.size:
            offset = buf.offset
            size = buf.size
            buf.data = np.frombuffer(file_data[offset:offset + size], dtype=np.uint8).copy()
            buf.offset = 0
            buf.size = 0
            inlined += 1
    if inlined:
        print(f"  Inlined {inlined} offset-based buffers")

    del file_data

    subgraph = model.subgraphs[0]
    graph_inputs = set(subgraph.inputs.tolist()) if subgraph.inputs is not None else set()
    graph_outputs = set(subgraph.outputs.tolist()) if subgraph.outputs is not None else set()

    # Pre-pass: identify buffers shared with non-FLOAT32 tensors (must NOT convert)
    buf_types = defaultdict(set)
    for t in subgraph.tensors:
        buf = model.buffers[t.buffer]
        if buf.data is not None and len(buf.data) > 0:
            buf_types[t.buffer].add(t.type)
    mixed_buffers = set()
    for buf_idx, types in buf_types.items():
        if schema_fb.TensorType.FLOAT32 in types and len(types) > 1:
            mixed_buffers.add(buf_idx)
    if mixed_buffers:
        print(f"  Skipping {len(mixed_buffers)} mixed-type shared buffer(s)")

    converted = 0
    total_saved = 0
    converted_buffers = set()

    for tensor_idx, tensor in enumerate(subgraph.tensors):
        if tensor.type != schema_fb.TensorType.FLOAT32:
            continue
        if tensor_idx in graph_inputs or tensor_idx in graph_outputs:
            continue
        if tensor.buffer in mixed_buffers:
            continue
        buf = model.buffers[tensor.buffer]
        if buf.data is None or len(buf.data) == 0:
            continue

        # Shared buffer already converted — just update tensor type
        if tensor.buffer in converted_buffers:
            tensor.type = schema_fb.TensorType.FLOAT16
            converted += 1
            continue

        if len(buf.data) % 4 != 0:
            continue

        # Convert buffer from FP32 to FP16 in-place
        fp32_data = np.frombuffer(buf.data, dtype=np.float32)
        fp16_data = fp32_data.astype(np.float16)
        saved = len(buf.data) - len(fp16_data.tobytes())
        total_saved += saved

        buf.data = np.frombuffer(fp16_data.tobytes(), dtype=np.uint8)
        tensor.type = schema_fb.TensorType.FLOAT16
        converted_buffers.add(tensor.buffer)
        converted += 1

    print(f"  Converted {converted} tensors in-place, saved {total_saved / 1e6:.0f} MB")

    # Pack and write
    print(f"  Packing...")
    t0 = time.time()
    builder = flatbuffers.Builder(1024 * 1024)
    packed = model.Pack(builder)
    builder.Finish(packed, file_identifier=b"TFL3")
    buf = builder.Output()
    print(f"  Packed in {time.time() - t0:.1f}s")

    output_path.write_bytes(bytes(buf))
    out_size = output_path.stat().st_size
    print(f"  Wrote {output_path} ({out_size / 1e6:.0f} MB)")
    print(f"  Reduction: {(1 - out_size / input_path.stat().st_size) * 100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.tflite> [output.tflite]")
        sys.exit(1)
    convert_fp32_to_fp16(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
