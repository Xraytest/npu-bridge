# -*- coding: utf-8 -*-
"""ComfyUI custom nodes: run ESRGAN-family upscalers (RRDBNet) on the Intel NPU.

The NPU executes the whole RRDBNet graph through npu_bridge (zero OpenVINO:
own IR emitter -> driver-embedded compiler -> Level Zero graph execution).
The GPU stays free while upscaling, so it can be pipelined with other work.
"""
import importlib.util
import os

import numpy as np
import torch

import folder_paths

_BRR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "npu_bridge"))

_spec = importlib.util.spec_from_file_location(
    "npu_bridge_runtime", os.path.join(_BRR, "runtime.py"))
runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runtime)

_us_spec = importlib.util.spec_from_file_location(
    "npu_bridge_ultrasharp", os.path.join(_BRR, "ultrasharp.py"))
ultrasharp = importlib.util.module_from_spec(_us_spec)
_us_spec.loader.exec_module(ultrasharp)

_ENGINES = {}
_UPSCALERS = {}


def _get_upscaler(model_path, tile):
    key = (model_path, tile)
    if key not in _UPSCALERS:
        if "default" not in _ENGINES:
            _ENGINES["default"] = runtime.NpuEngine()
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
        up = ultrasharp.UltraSharpNpu(sd, engine=_ENGINES["default"], tile=tile)
        up.build()
        _UPSCALERS[key] = up
    return _UPSCALERS[key]


class NpuUpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "tile_size": ([64, 128, 256],),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load"
    CATEGORY = "model/loaders"

    def load(self, model_name, tile_size):
        path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        up = _get_upscaler(path, tile_size)

        class _NpuModel:
            scale = 4
            tile = tile_size

            def __call__(self, image_bhwc):
                arr = image_bhwc.cpu().numpy()
                out = np.stack([up.upscale(a) for a in arr])
                return torch.from_numpy(out).to(image_bhwc.device, image_bhwc.dtype)

        model = _NpuModel()
        model.upscaler = up
        return (model,)


class NpuImageUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        return (upscale_model(image),)


NODE_CLASS_MAPPINGS = {
    "NpuUpscalerLoader": NpuUpscalerLoader,
    "NpuImageUpscale": NpuImageUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NpuUpscalerLoader": "Load Upscale Model (Intel NPU)",
    "NpuImageUpscale": "Upscale Image (Intel NPU)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

