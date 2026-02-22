#!/usr/bin/env python3
"""Convert TFLite FP32 models to FP16 weights with DEQUANTIZE ops.

The LiteRT GPU delegate natively handles FP16 data, so FP16 weights with
DEQUANTIZE ops run efficiently on GPU without the crashes seen with INT8
weight-only quantization.

This halves model size with essentially zero quality loss (cosine ~0.999999).

Usage:
    python convert_fp16.py <input.tflite> [output.tflite]

If no output path is given, writes to <input>_fp16.tflite.
"""

import sys
import struct
import time
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

    # Parse into mutable object tree
    t0 = time.time()
    model_obj = schema_fb.Model.GetRootAs(file_data)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    print(f"  Parsed in {time.time() - t0:.1f}s")

    # Phase 1: Inline all offset-based buffers (large models store data externally)
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

    # Free original file data to reduce peak memory
    del file_data

    # Phase 2: Find all FP32 weight tensors and convert to FP16
    subgraph = model.subgraphs[0]
    graph_inputs = set(subgraph.inputs.tolist()) if subgraph.inputs is not None else set()
    graph_outputs = set(subgraph.outputs.tolist()) if subgraph.outputs is not None else set()

    # Ensure DEQUANTIZE opcode exists
    dequant_opcode_idx = None
    for i, code in enumerate(model.operatorCodes):
        if code.builtinCode == 6:  # DEQUANTIZE
            dequant_opcode_idx = i
            break
    if dequant_opcode_idx is None:
        opcode = schema_fb.OperatorCodeT()
        opcode.builtinCode = 6
        opcode.deprecatedBuiltinCode = 6
        opcode.version = 1
        model.operatorCodes.append(opcode)
        dequant_opcode_idx = len(model.operatorCodes) - 1
        print(f"  Added DEQUANTIZE opcode at index {dequant_opcode_idx}")

    converted = 0
    total_saved = 0
    new_tensors = []
    new_ops = []
    tensor_remap = {}  # old_idx -> new_fp32_idx (output of DEQUANTIZE)

    for tensor_idx, tensor in enumerate(subgraph.tensors):
        # Skip non-FP32, activation tensors (empty buffer), graph I/O
        if tensor.type != schema_fb.TensorType.FLOAT32:
            continue
        if tensor_idx in graph_inputs or tensor_idx in graph_outputs:
            continue
        buf = model.buffers[tensor.buffer]
        if buf.data is None or len(buf.data) == 0:
            continue

        # Convert buffer from FP32 to FP16
        fp32_data = np.frombuffer(buf.data, dtype=np.float32)
        fp16_data = fp32_data.astype(np.float16)
        saved = len(buf.data) - len(fp16_data.tobytes())
        total_saved += saved

        buf.data = np.frombuffer(fp16_data.tobytes(), dtype=np.uint8)
        tensor.type = schema_fb.TensorType.FLOAT16

        # Create new FP32 tensor (DEQUANTIZE output, empty buffer — computed at runtime)
        new_buf = schema_fb.BufferT()
        model.buffers.append(new_buf)
        new_buf_idx = len(model.buffers) - 1

        new_tensor = schema_fb.TensorT()
        new_tensor.shape = tensor.shape.copy() if tensor.shape is not None else None
        new_tensor.type = schema_fb.TensorType.FLOAT32
        new_tensor.buffer = new_buf_idx
        new_tensor.name = (tensor.name or b"") + b"_dequantized"
        new_tensor.shapeSignature = (
            tensor.shapeSignature.copy() if tensor.shapeSignature is not None else None
        )
        new_tensors.append(new_tensor)
        new_fp32_idx = len(subgraph.tensors) + len(new_tensors) - 1

        # Create DEQUANTIZE op: FP16 tensor -> new FP32 tensor
        dq_op = schema_fb.OperatorT()
        dq_op.opcodeIndex = dequant_opcode_idx
        dq_op.inputs = np.array([tensor_idx], dtype=np.int32)
        dq_op.outputs = np.array([new_fp32_idx], dtype=np.int32)
        dq_op.builtinOptionsType = schema_fb.BuiltinOptions.DequantizeOptions
        dq_op.builtinOptions = schema_fb.DequantizeOptionsT()
        new_ops.append(dq_op)

        tensor_remap[tensor_idx] = new_fp32_idx
        converted += 1

    print(f"  Converted {converted} tensors, saved {total_saved / 1e6:.0f} MB")

    # Append new tensors to subgraph
    subgraph.tensors.extend(new_tensors)

    # Rewrite operator inputs to use dequantized FP32 tensors
    rewrites = 0
    for op in subgraph.operators:
        if op.inputs is not None:
            for i in range(len(op.inputs)):
                if op.inputs[i] in tensor_remap:
                    op.inputs[i] = tensor_remap[op.inputs[i]]
                    rewrites += 1
    print(f"  Rewired {rewrites} operator inputs")

    # Prepend DEQUANTIZE ops (must run before any op that uses their outputs)
    subgraph.operators = new_ops + list(subgraph.operators)

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
