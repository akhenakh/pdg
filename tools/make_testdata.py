#!/usr/bin/env python3
"""Regenerate engine test fixtures (deterministic noise) from ONNX Runtime.

The only nondeterministic node in the graph is RandomUniformLike.  For the
fixtures it is replaced by ConstantOfShape(0.5), which gives ORT and pdn
byte-comparable inputs.
"""

import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get("PARADEE_ONNX", "/tmp/v2g_noint8.onnx")
CHECK = "/tmp/paradee-web/paradee-web-v2/check.json"
OUT = os.path.join(ROOT, "internal", "engine", "testdata")


def main():
    ids = json.load(open(CHECK))["ids"][:3]
    m = onnx.load(SRC)
    nodes = []
    for n in m.graph.node:
        if n.op_type == "RandomUniformLike":
            shp = n.output[0] + "_shp"
            nodes.append(onnx.helper.make_node("Shape", [n.input[0]], [shp]))
            c = onnx.helper.make_node("ConstantOfShape", [shp], list(n.output))
            c.attribute.append(
                onnx.helper.make_attribute(
                    "value",
                    onnx.helper.make_tensor("noise", onnx.TensorProto.FLOAT, [1], [0.5]),
                )
            )
            nodes.append(c)
        else:
            nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    patched = os.path.join("/tmp", "paradee_det.onnx")
    onnx.save(m, patched)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(patched, so, providers=["CPUExecutionProvider"])

    os.makedirs(OUT, exist_ok=True)
    cases = []
    for i, sentence in enumerate(ids):
        y = sess.run(
            None,
            {
                "input_ids": np.array([sentence], np.int64),
                "speed": np.ones(1, np.float32),
            },
        )[0][0].astype(np.float32)
        y.tofile(os.path.join(OUT, f"ref{i}.f32"))
        cases.append({"ids": sentence, "samples": int(y.size), "file": f"ref{i}.f32"})
        print(i, len(sentence), "ids ->", y.size, "samples", float(np.abs(y).max()))
    json.dump({"cases": cases}, open(os.path.join(OUT, "cases.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
