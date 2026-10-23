#!/usr/bin/env python3
"""Convert the Paradee ONNX model into PDG's compact model format.

Build-time only. Users of pdg never need Python or ONNX Runtime.

    pip install onnx onnxruntime numpy
    python3 tools/convert.py [--onnx FILE] [--out internal/assets/paradee.bin]

The ONNX file is first optimized with ONNX Runtime at the BASIC level:
constant folding removes the int4/int6/int8 weight unpacking and the
tokenizer/vocab tables baked into the graph, while dynamic shapes
(the token axis) are preserved.  The remaining graph is a plain ONNX
program (707 nodes) that PDG's Go executor runs directly.

The output file is:

    "PDNG" u32 version
    u32 header_len, header JSON, raw tensor blob

float16 and small integer tensors are narrowed to float32/int64.  Float64
is preserved: the model's frame/phase grid is computed in double.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper

DEFAULT_URL = "https://mesmer.tools/paradee/onnx/paradee_v2g_noint8.onnx"


def fetch(url, path):
    if os.path.exists(path):
        return path
    import urllib.request

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def optimize(model_path, opt_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = opt_path
    # Session construction is what writes the optimized graph.
    ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    return onnx.load(opt_path)


def dtype_tag(elem):
    if elem == TensorProto.FLOAT:
        return "f32"
    if elem in (TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8, TensorProto.UINT8):
        return "i64"
    if elem == TensorProto.BOOL:
        return "i64"
    if elem == TensorProto.DOUBLE:
        return "f64"
    if elem == TensorProto.FLOAT16:
        return "f32"
    raise ValueError(f"unsupported tensor dtype {elem}")


def narrow(arr):
    if arr.dtype == np.float16:
        return arr.astype(np.float32)
    if arr.dtype == np.bool_:
        return arr.astype(np.int64)
    if arr.dtype in (np.int32, np.int16, np.int8, np.uint8, np.uint32):
        return arr.astype(np.int64)
    return arr


def encode_tensor(arr):
    arr = np.ascontiguousarray(narrow(arr))
    if arr.dtype == np.float32:
        return "f32", arr.size, arr.tobytes()
    if arr.dtype == np.float64:
        return "f64", arr.size, arr.tobytes()
    if arr.dtype == np.int64:
        return "i64", arr.size, arr.tobytes()
    raise ValueError(f"unsupported dtype {arr.dtype}")


def attr_value(attr):
    if attr.type == AttributeProto.INT:
        return int(attr.i)
    if attr.type == AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == AttributeProto.TENSOR:
        t = numpy_helper.to_array(attr.t)
        dt, n, raw = encode_tensor(t)
        dtmap = {"f32": np.float32, "f64": np.float64, "i64": np.int64}
        vals = np.frombuffer(raw, dtype=dtmap[dt]).tolist()
        return {"tensor": {"dtype": dt, "shape": list(t.shape), "data": vals}}
    raise ValueError(f"unsupported attribute type {attr.type} in {attr.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="paradee_v2g_noint8.onnx")
    ap.add_argument("--out", default="internal/assets/paradee.bin")
    args = ap.parse_args()

    src = fetch(DEFAULT_URL, args.onnx)
    opt = src.replace(".onnx", "_pdg_opt.onnx")
    model = optimize(src, opt)
    graph = model.graph

    tensors = []
    blob = bytearray()
    seen = set()

    def add_tensor(name, arr):
        dt, n, raw = encode_tensor(arr)
        if name in seen:
            return
        seen.add(name)
        off = len(blob)
        blob.extend(raw)
        tensors.append(
            {
                "name": name,
                "dtype": dt,
                "shape": list(arr.shape),
                "off": off,
                "n": n,
            }
        )

    for init in graph.initializer:
        add_tensor(init.name, numpy_helper.to_array(init))

    nodes = []
    for node in graph.node:
        attrs = {}
        for a in node.attribute:
            attrs[a.name] = attr_value(a)
        nodes.append(
            {
                "op": node.op_type,
                "name": node.name,
                "in": list(node.input),
                "out": list(node.output),
                "attrs": attrs,
            }
        )

    inputs = []
    for i in graph.input:
        d = i.type.tensor_type
        inputs.append(
            {
                "name": i.name,
                "dtype": dtype_tag(d.elem_type),
                "rank": len(d.shape.dim),
            }
        )
    outputs = [o.name for o in graph.output]

    # v2g's decoder is streamable: everything up to and including the noise
    # reshape ("uY") is sentence-global; the rest is per-subframe (150 samples
    # each) and can run on a time slice.
    producer = {}
    for i, node in enumerate(nodes):
        for out in node["out"]:
            producer[out] = i
    if "uY" not in producer or "yf" not in producer:
        raise SystemExit("streaming split tensors uY/yf not found in graph")
    stream = {
        "split": producer["uY"] + 1,
        "end": producer["yf"] + 1,
        "subframe_samples": 150,
    }
    header = {
        "format": "pdg-model",
        "version": 1,
        "source": os.path.basename(src),
        "sample_rate": 24000,
        "inputs": inputs,
        "outputs": outputs,
        "stream": stream,
        "tensors": tensors,
        "nodes": nodes,
    }
    hb = json.dumps(header, separators=(",", ":")).encode()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"PDNG")
        f.write((1).to_bytes(4, "little"))
        f.write(len(hb).to_bytes(4, "little"))
        f.write(hb)
        f.write(blob)

    print(
        f"{args.out}: {len(nodes)} nodes, {len(tensors)} tensors, "
        f"{len(blob)/1e6:.2f} MB weights, {os.path.getsize(args.out)/1e6:.2f} MB file"
    )
    ops = {}
    for n in nodes:
        ops[n["op"]] = ops.get(n["op"], 0) + 1
    print("ops:", " ".join(f"{k}={v}" for k, v in sorted(ops.items())))


if __name__ == "__main__":
    sys.exit(main())
