# npu-bridge

Run ESRGAN-family upscalers (RRDBNet, e.g. **4x-UltraSharp**) on the **Intel AI
Boost NPU** from ComfyUI — with **no OpenVINO runtime dependency**.

A pure-Python (ctypes) Level Zero graph runtime: it emits its own IR, hands it
to the compiler that already ships inside the Intel NPU driver
(`npu_driver_compiler.dll`), and submits the compiled graph to the NPU through
the `ZE_extension_graph` Level Zero extension.

```
PyTorch state_dict
  → own IR emitter (OpenVINO IR v10 XML subset + weights .bin, hand-written)
  → npu_driver_compiler.dll (driver-embedded compiler, via VCL C API / zeGraphCreate2)
  → ELF blob → Level Zero graph API (zeGraphExecute) → Intel AI Boost NPU
```

## Requirements

- Windows 10/11, Intel Core Ultra (Meteor Lake / Arrow Lake / Lunar Lake) with
  Intel AI Boost NPU, driver ≥ 32.0.100.4xxx
- Python 3.10+, `numpy`, `torch` (CPU or CUDA — the NPU bridge itself needs none)
- ComfyUI (for the custom node)
- **No OpenVINO. The driver package provides the compiler.**

## Install

### Custom node (ComfyUI)

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/Xraytest/npu-bridge npu-bridge
```

Then in the node graph:

- `Load Upscale Model (Intel NPU)` — pick a `.pth` from `models/upscale_models`
- `Upscale Image (Intel NPU)` — feed it an `IMAGE`

On first use the graph is compiled by the driver (~10–20 s for 4x-UltraSharp)
and cached in-process. The GPU stays free while the NPU upscales, so the two
can be pipelined.

### Library use

```python
import sys
sys.path.insert(0, r"path\to\npu-bridge")
from runtime import NpuEngine, IrBuilder
from ultrasharp import UltraSharpNpu
import torch

sd = torch.load("4x-UltraSharp.pth", map_location="cpu", weights_only=True)
up = UltraSharpNpu(sd, tile=128)
up.build()

img = ...  # float32 HWC in [0, 1]
out = up.upscale(img)  # 4H x 4W x 3 float32
```

## Verified

| Item | Result |
|---|---|
| NPU discovery | driver 32.0.100.4404, VPU `8086:AD1D`, compiler 7.26 (2025.4.0) |
| Minimal graph (ReLU) | end-to-end numeric PASS (0 error) |
| Conv3x3 / Conv1x1 / additive bias | PASS (≤ 4e-5) |
| Full 4x-UltraSharp (2624 layers, 353 args, 33 MB weights) | vs torch-fp16: rel 0.43% PASS |
| Throughput (Arrow Lake-HX NPU3) | 64 px tile 148 ms · 128 px 399 ms · 256 px 2.8 s |
| ComfyUI end-to-end | FLUX (GPU 512 px) → NPU 4× → 2048 px PNG |

The NPU is ~1.4–2× slower than an RTX 5060 per tile on this workload, but it is
a separate device: the GPU can keep sampling while the NPU upscales.

## Driver quirks this works around

All findings are empirical, on `npu_driver_compiler.dll` 2025.4.0 / UMD
32.0.100.4404:

1. **IR container header** must be the *compiler version* `(7, 26)`; standard
   IR versions 10.x/11.x are rejected ("Unsupported IR API version"). Layers
   use `version="opset1"`.
2. **Const tensors are rank-squeezed** by the reader (`[1,C,1,1] → [1,C]`), so
   additive biases cannot broadcast as Consts — each bias is emitted as a graph
   `Parameter` and bound once from the host.
3. **LeakyReLU is absent** from the reader's opset — emitted as
   `Maximum(x, 0.2·x)`.
4. **The `Interpolate` op is broken** (fp16 scales rejected, pass crash).
   RRDBNet's upsampler is `nn.Upsample(mode='nearest')`, exactly reproduced by
   a 1×1 depthwise conv (4 identical phase channels) + `DepthToSpace`.
5. **DepthToSpace requires `mode="depth_first"`** on this driver
   (`blocks_first` has different semantics here).
6. **GroupConvolution** weight layout is `[G, OC/G, IC/G, kH, kW]` (verified).
7. Fused Convolution bias (classic IR `<blobs>`) is not supported by the reader
   — biases always go through Parameters.
8. Driver logging (`NPU_LOG`) is stripped from the Windows UMD; for debugging,
   call the VCL C API directly (`vclExecutableCreate` +
   `vclLogHandleGetString`) to obtain compiler errors.

## Scope / limitations

- RRDBNet topologies (ESRGAN / 4x-UltraSharp family, `num_grow_ch=32`,
  3 RDBs per block) — other archs need new emit rules.
- Fixed shapes per graph; tiles are re-used across one `UltraSharpNpu`.
- fp16 compute (driver-compiled); IO is fp32 and cast at the boundary.
- FLUX/DiT on NPU is **not** a goal: NPU3 is ~11 TOPS fp16 with no fused
  attention in this opset — the sensible split is GPU for diffusion,
  NPU for upscaling/post-processing (devices are independent, verified).

## Legal

- This repository contains **no Intel binaries**. The runtime loads the driver
  already installed on your machine (`ze_loader.dll`,
  `npu_driver_compiler.dll`, `npu_level_zero_umd.dll`) via its public C
  interfaces.
- Structure/format knowledge was derived from the public spec headers
  ([oneapi-src/level-zero](https://github.com/oneapi-src/level-zero),
  [intel/level-zero-npu-extensions](https://github.com/intel/level-zero-npu-extensions))
  and the open-source
  [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver) (MIT) —
  no code was copied from them.

## License

MIT

---

# npu_bridge — Intel NPU 推理运行时（中文说明）

为 ComfyUI 提供的 Intel AI Boost NPU 直通推理栈。不依赖 OpenVINO 运行时，
通过纯 Python ctypes 直接驱动 Level Zero 图 API。

## 架构

```
PyTorch state_dict
      │  ultrasharp.py（拓扑转换器）
      ▼
自研 IR（OpenVINO IR v10 XML 子集，opset1）── 权重 .bin
      │  runtime.build_ngraph_lite_container()
      ▼
npu_driver_compiler.dll（驱动内置编译器）
      │  zeGraphCreate2 NGRAPH_LITE → vclExecutableCreate
      ▼
ELF blob（\x7fELF）──► zeGraphExecute → NPU 硬件
```

- `runtime.py` — Level Zero 绑定：zeInitDrivers(NPU) / 设备枚举 / 上下文 /
  命令队列 / zeMemAllocHost / zeGraphCreate2 / AppendGraphInitialize+Execute。
  含 `IrBuilder`（IR XML 发射器）与 fp16/fp32 主机缓冲读写。
- `ultrasharp.py` — 4x-UltraSharp（RRDBNet）整体到 NPU 图的转换器 + 分块执行器。
- `comfyui_custom_node/` — ComfyUI 节点，安装方式：
  复制 `comfyui_custom_node` 目录到 `ComfyUI/custom_nodes/`（或按上面 git clone）。

## 已验证

| 项 | 结果 |
|---|---|
| NPU 发现 | driver 32.0.100.4404, VPU 0x8086:0xAD1D, 编译器 7.26/2025.4.0 |
| 最小图（ReLU） | 端到端数值 PASS（0 误差） |
| Conv3x3/Conv1x1/加性偏置 | PASS（≤4e-5） |
| 完整 4x-UltraSharp（2624 层, 353 参数, 33MB 权重） | vs torch-fp16: rel 0.43% PASS |
| 吞吐 | 64px tile 148ms / 128px 399ms / 256px 2.8s |
| ComfyUI 端到端 | FLUX klein(GPU 512px) → NPU 4x → 2048px PNG ✅ |

## 驱动限制与对策（实测记录）

1. **IR 容器头** = 编译器版本 (7,26)，10.x/11.x 会被拒绝
   （"Unsupported IR API version"）。层用 `version="opset1"`。
2. **Const 会被 reader 压维**（[1,C,1,1]→[1,C]），加性偏置无法作 Const
   广播 —— 对策：每个偏置作为图 Parameter 输入，运行前一次性写入。
3. **LeakyReLU 不在 reader 的 opset1 里** —— 用 `Maximum(x, 0.2x)` 等价。
4. **Interpolate 算子损坏**（fp16 scales 判定错误、pass 崩溃）——
   4x-UltraSharp 的上采样实为 `nn.Upsample(mode='nearest')`：
   用 1x1 深度卷积（每通道 4 个相同相位）+ DepthToSpace 精确替代。
5. **DepthToSpace 必须 `mode="depth_first"`**（blocks_first 语义不符）。
6. **GroupConvolution** 权重布局 `[G, OC/G, IC/G, kH, kW]`（实测确认）。
7. Convolution 原生 bias 融合（`<blobs>`）不受支持 —— 偏置一律走 Parameter。
8. 驱动日志（NPU_LOG）在 Windows UMD 中被裁剪，调试需直接调
   `npu_driver_compiler.dll` 的 VCL C 接口（`vclExecutableCreate` +
   `vclLogHandleGetString`）拿编译错误。

## 许可

MIT。本仓库不包含任何 Intel 二进制；运行时加载用户本机已安装的驱动，
通过其公开 C 接口调用。
