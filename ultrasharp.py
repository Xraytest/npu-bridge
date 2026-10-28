# -*- coding: utf-8 -*-
"""Full 4x-UltraSharp (RRDBNet) on Intel NPU via npu_bridge.runtime.

Op mapping (verified against npu_driver_compiler 2025.4.0 / opset1 reader):
  Convolution 3x3        -> Convolution (weights Const + bias broadcast Parameter)
  LeakyReLU(0.2)         -> Maximum(x, 0.2*x)   (LeakyReLU op absent from reader)
  dense concat           -> Concat (axis as data attr)
  bilinear x2 upsample   -> per-phase Conv3x3 + DepthToSpace (Interpolate reader
                            is broken); zero-pad border deviates from torch's
                            replicate-edge bilinear by <=1px at image borders.
Weights stay fp16 (driver constant folding keeps fp16 MACs); IO is fp32.
"""
import numpy as np

try:
    from .runtime import NpuEngine, IrBuilder
except ImportError:
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from runtime import NpuEngine, IrBuilder

LRELU = 0.2


class UltraSharpNpu:
    def __init__(self, state_dict, engine=None, tile=64):
        self.sd = state_dict
        self.e = engine or NpuEngine()
        self.tile = tile
        self.graph = None
        self._bias_names = []

    def build(self):
        sd = self.sd
        t = self.tile
        ir = IrBuilder(f"ultrasharp_t{t}")
        cur = ir.parameter([1, 3, t, t])
        cur = self._conv(ir, cur, "model.0.weight", "model.0.bias")
        feat = cur  # head conv, no activation (conv_block act_type=None)
        for rb in range(23):
            block_in = cur
            rdb_in = cur
            for rdb in range(3):
                parts = [rdb_in]
                dense = rdb_in
                for c in range(5):
                    dense = self._conv(
                        ir, dense, f"model.1.sub.{rb}.RDB{rdb+1}.conv{c+1}.0.weight",
                        f"model.1.sub.{rb}.RDB{rdb+1}.conv{c+1}.0.bias")
                    if c < 4:
                        dense = self._lrelu(ir, dense)
                        parts.append(dense)
                        dense = ir.concat(parts, [1, 64 + 32 * (len(parts) - 1), t, t])
                d5 = ir.mul_const(dense, LRELU)
                rdb_in = ir.add(d5, rdb_in, [1, 64, t, t])
            cur = ir.mul_const(rdb_in, LRELU)
            cur = ir.add(cur, block_in, [1, 64, t, t])
        cur = self._conv(ir, cur, "model.1.sub.23.weight", "model.1.sub.23.bias")
        cur = ir.add(cur, feat, [1, 64, t, t])  # ShortcutBlock: x + sub(x)
        for up in range(2):
            cur = self._bilinear_x2(ir, cur)
            key = "model.3" if up == 0 else "model.6"
            cur = self._conv(ir, cur, f"{key}.weight", f"{key}.bias")
            cur = self._lrelu(ir, cur)
        cur = self._conv(ir, cur, "model.8.weight", "model.8.bias")
        cur = self._lrelu(ir, cur)
        cur = self._conv(ir, cur, "model.10.weight", "model.10.bias")
        ir.result(cur)
        g = self.e.create_graph(ir.to_xml(), weights=ir.weights)
        g.bind_buffers()
        bi = 0
        for i, a in enumerate(g.args):
            if a.arg_type == 0 and a.dims[-1] == 1 and a.dims[-2] == 1 and a.name != "input0":
                g.write_input(i, self.sd[self._bias_names[bi]].numpy().astype(np.float32))
                bi += 1
        assert bi == len(self._bias_names), (bi, len(self._bias_names))
        g.init()
        self.graph = g
        return g

    def _out_dims(self, ir, src, oc):
        import re
        xml = ir.layers_by_id[src]
        seg = xml.split("<output>")[1].split("</port>")[0]
        dims = [int(m) for m in re.findall(r"<dim>(\d+)</dim>", seg)]
        return [1, oc, dims[2], dims[3]]

    def _conv(self, ir, src, wname, bname):
        w = self.sd[wname].numpy().astype(np.float16)
        wl = ir.const(w.tobytes(), list(w.shape))
        oc = w.shape[0]
        out_dims = self._out_dims(ir, src, oc)
        x = ir.conv2d(src, wl, out_dims=out_dims, pads=(1, 1))
        if bname:
            bp = ir.parameter([1, oc, 1, 1])
            self._bias_names.append(bname)
            x = ir.add(x, bp, out_dims)
        return x

    def _lrelu(self, ir, src):
        m = ir.mul_const(src, LRELU)
        return ir.maximum(src, m)

    def _bilinear_x2(self, ir, src):
        """Nearest x2 (torch Upsample mode='nearest', what RRDBNet's upconv
        actually uses): 1x1 depthwise conv emitting 4 identical phase channels
        + DepthToSpace. Exact at borders too."""
        import re
        xml = ir.layers_by_id[src]
        seg = xml.split("<output>")[1].split("</port>")[0]
        dims = [int(m) for m in re.findall(r"<dim>(\d+)</dim>", seg)]
        c, h, w = dims[1], dims[2], dims[3]
        Wt = np.ones((c, 4, 1, 1, 1), np.float16)
        wl = ir.const(Wt.tobytes(), [c, 4, 1, 1, 1])
        lid = ir.next_id
        ir.next_id += 1
        xml = (f'    <layer id="{lid}" name="up{lid}" type="GroupConvolution" version="opset1">\n'
               f'      <data strides="1,1" dilations="1,1" pads_begin="0,0" pads_end="0,0"/>\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{ir._dims_xml(dims)}</port>\n'
               f'        <port id="1" precision="FP16">{ir._dims_xml([c, 4, 1, 1, 1])}</port>\n'
               f'      </input>\n'
               f'      <output>\n        {ir._port([1, 4 * c, h, w])}\n      </output>\n'
               f'    </layer>')
        ir.layers.append(xml)
        ir.layers_by_id[lid] = xml
        ir.edges.append((src, 0, lid, 0))
        ir.edges.append((wl, 0, lid, 1))
        return ir.depth_to_space(lid, [1, c, 2 * h, 2 * w], block_size=2)

    def upscale(self, image_nhwc):
        """float32 HWC [0,1] -> 4H x 4W x 3 float32. Tile + blend borders."""
        g = self.graph
        t = self.tile
        h, w, _ = image_nhwc.shape
        ys = list(range(0, h, t))
        xs = list(range(0, w, t))
        out = np.zeros((h * 4, w * 4, 3), np.float32)
        for y in ys:
            for x in xs:
                th = min(t, h - y)
                tw = min(t, w - x)
                tile = np.zeros((t, t, 3), np.float32)
                tile[:th, :tw] = image_nhwc[y:y + th, x:x + tw]
                g.write_input(0, tile.transpose(2, 0, 1).reshape(-1))
                g.execute()
                o = g.read_output([i for i, a in enumerate(g.args) if a.arg_type == 1][0])
                out[y * 4:y * 4 + th * 4, x * 4:x * 4 + tw * 4] = np.clip(
                    o[0, :th * 4, :tw * 4].transpose(1, 2, 0), 0, 1)
        return out



