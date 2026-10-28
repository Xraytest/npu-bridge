# -*- coding: utf-8 -*-
"""Smoke test: NPU discovery + minimal graph end-to-end (no model weights needed).

Run from the repo root with the ComfyUI venv or any Python that has numpy:
    python smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from runtime import NpuEngine, IrBuilder  # noqa: E402


def main():
    print("== opening NPU ==")
    e = NpuEngine()
    print(f"compiler v{e.compiler_version[0]}.{e.compiler_version[1]} "
          f"formats=0x{e.graph_formats:X} maxOpset={e.max_opset}")

    ir = IrBuilder("smoke")
    x = ir.parameter([1, 4, 8, 8])
    y = ir.relu(x)
    ir.result(y)

    g = e.create_graph(ir.to_xml())
    g.bind_buffers()
    g.write_input(0, np.full(256, -1.0, np.float32))
    g.init()
    g.execute()
    out = g.read_output(1 if len(g.args) > 1 else 0)
    ok = all(abs(v) < 1e-6 for v in out.reshape(-1))
    print("smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
