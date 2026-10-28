# -*- coding: utf-8 -*-
"""
Intel NPU Level Zero graph runtime -- zero-OpenVINO, pure ctypes.

Bindings follow the level-zero spec (ze_api.h) and the intel/linux-npu-driver
UMD sources (MIT): graph extension "ZE_extension_graph", NGRAPH_LITE input
container built by our own emitter, execution via zeCommandList/zeCommandQueue.

IR container layout (umd/level_zero_driver/unit_tests/fixtures/graph_fixture.hpp):
    struct ModelIR {
        ze_graph_compiler_version_info_t version;  // 2 x u16
        uint32_t numInputs;
        struct { uint64_t size; uint8_t data[size]; } buffers[];
    }
buffers = [xml(IR v10, opset11), weights(empty for weightless graphs)]
"""
import ctypes as C
import struct

ze_result_t = C.c_int

ZE_RESULT_SUCCESS = 0
ZE_RESULT_NOT_READY = 1

ZT_DRIVER_PROPERTIES = 0x1
ZT_DEVICE_PROPERTIES_1_2 = 0x00020006
ZT_CONTEXT_DESC = 0xD
ZT_COMMAND_QUEUE_DESC = 0xE
ZT_COMMAND_LIST_DESC = 0xF
ZT_HOST_MEM_ALLOC_DESC = 0x16
ZT_INIT_DRIVER_TYPE_DESC = 0x00020021

ZE_DEVICE_TYPE_VPU = 5
ZE_INIT_DRIVER_TYPE_FLAG_NPU = 1 << 1
ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED = 1 << 0

ZE_EXTENSION_GRAPH = b"ZE_extension_graph"

# graph ext structure types
GX_DEVICE_GRAPH_PROPERTIES = 0x1
GX_DEVICE_GRAPH_PROPERTIES_2 = 0xF
GX_GRAPH_DESC_2 = 0xE
GX_GRAPH_PROPERTIES = 0x3
GX_GRAPH_PROPERTIES_2 = 0x10
GX_GRAPH_PROPERTIES_3 = 0x11
GX_GRAPH_ARGUMENT_PROPERTIES = 0x4

GX_FORMAT_NATIVE = 0x1
GX_FORMAT_NGRAPH_LITE = 0x2

GX_ARG_TYPE_INPUT = 0
GX_ARG_TYPE_OUTPUT = 1

# precision enum values (ze_graph_argument_precision_t)
PREC_NAMES = {
    0x01: "FP32", 0x02: "FP16", 0x09: "BF16", 0x13: "F8E4M3",
    0x0A: "U32", 0x04: "U8", 0x03: "U16", 0x0B: "U4",
    0x05: "I32", 0x07: "I8", 0x06: "I16", 0x0C: "I4",
    0x08: "BIN", 0x0E: "BOOL", 0x0F: "FP64", 0x10: "U64", 0x11: "I64",
}
PREC_BYTES = {"FP16": 2, "FP32": 4, "BF16": 2, "U8": 1, "I8": 1, "I32": 4, "U32": 4}


def _err(r, what):
    if r != ZE_RESULT_SUCCESS:
        raise RuntimeError(f"{what} failed: 0x{r & 0xFFFFFFFF:08X}")
    return r


class GraphCompilerVersion(C.Structure):
    _fields_ = [("major", C.c_uint16), ("minor", C.c_uint16)]


class DeviceGraphProperties(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("graphExtensionVersion", C.c_uint32),
        ("compilerVersion", GraphCompilerVersion),
        ("graphFormatsSupported", C.c_uint32),
        ("maxOVOpsetVersionSupported", C.c_uint32),
    ]


class GraphVersionInfo(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32), ("patch", C.c_uint32)]


class DeviceGraphProperties2(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("graphExtensionVersion", C.c_uint32),
        ("compilerVersion", GraphCompilerVersion),
        ("graphFormatsSupported", C.c_uint32),
        ("maxOVOpsetVersionSupported", C.c_uint32),
        ("elfVersion", GraphVersionInfo),
        ("runtimeVersion", GraphVersionInfo),
    ]


class GraphProperties(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("numGraphArgs", C.c_uint32)]


class GraphProperties2(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("numGraphArgs", C.c_uint32),
        ("initStageRequired", C.c_uint32),
    ]


class GraphArgumentProperties(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("name", C.c_char * 256),
        ("type", C.c_int),
        ("dims", C.c_uint32 * 5),
        ("networkPrecision", C.c_int),
        ("networkLayout", C.c_int),
        ("devicePrecision", C.c_int),
        ("deviceLayout", C.c_int),
    ]


class GraphDesc2(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("format", C.c_uint32),
        ("inputSize", C.c_size_t),
        ("pInput", C.c_void_p),
        ("pBuildFlags", C.c_char_p),
        ("flags", C.c_uint32),
    ]


class ContextDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]


class QueueDesc(C.Structure):
    _fields_ = [
        ("stype", C.c_uint32), ("pNext", C.c_void_p),
        ("ordinal", C.c_uint32), ("index", C.c_uint32), ("flags", C.c_uint32),
        ("mode", C.c_int), ("priority", C.c_int),
    ]


class ListDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p),
                ("commandQueueGroupOrdinal", C.c_uint32)]


class HostMemAllocDesc(C.Structure):
    _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]


GRAPH_FN_NAMES = [
    "Create", "Destroy", "GetProperties", "GetArgumentProperties", "SetArgumentValue",
    "AppendGraphInitialize", "AppendGraphExecute", "GetNativeBinary", "DeviceGetGraphProperties",
    "GraphGetArgumentMetadata", "GetArgumentProperties2", "GetArgumentProperties3",
    "QueryNetworkCreate", "QueryNetworkDestroy", "QueryNetworkGetSupportedLayers",
    "BuildLogGetString", "Create2", "QueryNetworkCreate2", "QueryContextMemory",
    "DeviceGetGraphProperties2", "GetNativeBinary2", "GetProperties2", "GraphInitialize",
]


class NpuTensorInfo:
    def __init__(self, name, arg_type, dims, precision):
        self.name = name
        self.arg_type = arg_type  # 0 input, 1 output
        dims = list(dims)
        while len(dims) > 1 and dims[-1] == 0:
            dims.pop()  # drop zero padding of the fixed 5-dim array
        self.dims = dims
        self.precision = precision
        p = PREC_NAMES.get(precision, f"?{precision}")
        self.precision_name = p
        nbytes = PREC_BYTES.get(p, 1)
        n = 1
        for d in self.dims:
            n *= d
        self.numel = n
        self.nbytes = n * nbytes

    def __repr__(self):
        return (f"<{['in','out'][self.arg_type]} {self.name!r} "
                f"{self.dims} {self.precision_name} ({self.nbytes} B)>")


def build_ngraph_lite_container(xml_bytes, weights_bytes, ir_header=(7, 26)):
    """IR container header: the COMPILER version (e.g. 7.26) is the accepted
    'IR API version'; 10.x/11.x are explicitly rejected by the driver compiler
    (verified against npu_driver_compiler.dll 2025.4.0). Layers use opset1.
    """
    parts = [struct.pack("<HH", *ir_header), struct.pack("<I", 2)]
    for blob in (xml_bytes, weights_bytes):
        parts.append(struct.pack("<Q", len(blob)))
        parts.append(blob)
    return b"".join(parts)


# ---------------------------------------------------------------- IR emitter
class IrBuilder:
    """Emits OpenVINO IR v10 XML (opset11) for weightless graphs. Zero OpenVINO.

    Only the ops the NPU compiler reliably lowers for a first e2e proof:
    Parameter -> eltwise (ReLU/Clamp) -> Result. Convolution with weights
    follows once the runtime loop is proven.
    """

    def __init__(self, name):
        self.name = name
        self.layers = []       # ordered xml
        self.layers_by_id = {} # id -> xml (ids may be created out of order)
        self.edges = []        # (from_layer, from_port, to_layer, to_port)
        self.weights = b""     # concatenated Const payloads (the .bin)
        self.next_id = 0
        self.inputs = []       # layer ids of Parameters
        self.outputs = []      # layer ids of Results

    def _fmt(self, dims):
        return ",".join(str(d) for d in dims)

    def _port(self, dims, precision="FP16", indent="        "):
        dims_xml = "".join(f"\n{indent}  <dim>{d}</dim>" for d in dims)
        return (f'<port id="0" precision="{precision}">{dims_xml}\n{indent}</port>')

    @staticmethod
    def _dims_xml(dims):
        return "".join(f"<dim>{d}</dim>" for d in dims)

    def _layer(self, lid, name, ltype, dims, prec, data_xml="", with_output=True):
        d = f"\n      <data {data_xml}/>" if data_xml else ""
        io_ports = f'      <input>\n        {self._port(dims, prec, indent="        ")}\n      </input>\n'
        out = f'      <output>\n        {self._port(dims, prec, indent="        ")}\n      </output>\n' if with_output else ""
        return (f'    <layer id="{lid}" name="{name}" type="{ltype}" version="opset1">\n'
                f'{d}{io_ports}{out}    </layer>')

    def parameter(self, dims, precision="FP16"):
        lid = self.next_id
        self.next_id += 1
        et = {"FP16": "f16", "FP32": "f32", "U8": "u8", "I32": "i32"}[precision]
        xml = (f'    <layer id="{lid}" name="input{lid}" type="Parameter" version="opset1">\n'
               f'      <data element_type="{et}" shape="{self._fmt(dims)}"/>\n'
               f'      <output>\n        {self._port(dims, precision)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.inputs.append(lid)
        return lid

    def relu(self, src):
        lid = self.next_id
        self.next_id += 1
        dims = self._dims_of(src)
        prec = self._prec_of(src)
        xml = (f'    <layer id="{lid}" name="relu{lid}" type="ReLU" version="opset1">\n'
               f'      <input>\n        {self._port(dims, prec)}\n      </input>\n'
               f'      <output>\n        {self._port(dims, prec)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        return lid

    def clamp(self, src, lo, hi):
        lid = self.next_id
        self.next_id += 1
        dims = self._dims_of(src)
        prec = self._prec_of(src)
        xml = (f'    <layer id="{lid}" name="clamp{lid}" type="Clamp" version="opset1">\n'
               f'      <data min="{lo}" max="{hi}"/>\n'
               f'      <input>\n        {self._port(dims, prec)}\n      </input>\n'
               f'      <output>\n        {self._port(dims, prec)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        return lid

    def const(self, data_bytes, dims, precision="FP16", offset=None):
        """Weights live in the container's second buffer (the .bin). When
        offset is None the data is auto-appended to the builder's blob
        (ir.weights) at the next free offset."""
        if offset is None:
            offset = len(self.weights)
            self.weights += data_bytes
        lid = self.next_id
        self.next_id += 1
        et = {"FP16": "f16", "FP32": "f32", "I32": "i32", "U8": "u8"}[precision]
        xml = (f'    <layer id="{lid}" name="const{lid}" type="Const" version="opset1">\n'
               f'      <data element_type="{et}" '
               f'shape="{self._fmt(dims)}" offset="{offset}" size="{len(data_bytes)}"/>\n'
               f'      <output>\n        {self._port(dims, precision)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        return lid

    def add(self, a, b, out_dims):
        """Elementwise add of two layers (same shape or broadcastable)."""
        lid = self.next_id
        self.next_id += 1
        da, db = self._dims_of(a), self._dims_of(b)
        xml = (f'    <layer id="{lid}" name="add{lid}" type="Add" version="opset1">\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{self._dims_xml(da)}</port>\n'
               f'        <port id="1" precision="FP16">{self._dims_xml(db)}</port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(out_dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((a, 0, lid, 0))
        self.edges.append((b, 0, lid, 1))
        return lid

    def mul_const(self, src, scalar):
        """src * scalar via Const(broadcast 1,1,1,1) + Multiply."""
        import struct as _s
        b = _s.pack("<e", float(scalar))
        clid = self.const(b, [1, 1, 1, 1])
        lid = self.next_id
        self.next_id += 1
        dims = self._dims_of(src)
        xml = (f'    <layer id="{lid}" name="mul{lid}" type="Multiply" version="opset1">\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{self._dims_xml(dims)}</port>\n'
               f'        <port id="1" precision="FP16"><dim>1</dim><dim>1</dim><dim>1</dim><dim>1</dim></port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        self.edges.append((clid, 0, lid, 1))
        return lid

    def maximum(self, a, b, out_dims=None):
        lid = self.next_id
        self.next_id += 1
        dims = out_dims or self._dims_of(a)
        xml = (f'    <layer id="{lid}" name="max{lid}" type="Maximum" version="opset1">\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{self._dims_xml(self._dims_of(a))}</port>\n'
               f'        <port id="1" precision="FP16">{self._dims_xml(self._dims_of(b))}</port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((a, 0, lid, 0))
        self.edges.append((b, 0, lid, 1))
        return lid

    def leaky_relu(self, src, slope=0.2):
        """LeakyReLU as Maximum(x, slope*x) -- LeakyReLU op is not in this
        compiler's opset reader; Mul+Max is proven equivalent."""
        m = self.mul_const(src, slope)
        return self.maximum(src, m)

    def conv2d(self, src, w_lid, out_dims, strides=(1, 1), pads=(0, 0), dilations=(1, 1)):
        """src: data layer id; w_lid: Const layer id; out_dims: computed output shape."""
        lid = self.next_id
        self.next_id += 1
        dims = out_dims
        prec = self._prec_of(src)
        st = ",".join(map(str, strides))
        pb = ",".join(map(str, pads))
        pe = ",".join(map(str, pads))
        dl = ",".join(map(str, dilations))
        xml = (f'    <layer id="{lid}" name="conv{lid}" type="Convolution" version="opset1">\n'
               f'      <data strides="{st}" dilations="{dl}" pads_begin="{pb}" pads_end="{pe}"/>\n'
               f'      <input>\n'
               f'        <port id="0" precision="{prec}">{self._dims_xml(self._dims_of(src))}</port>\n'
               f'        <port id="1" precision="{prec}">{self._dims_xml(self._dims_of(w_lid))}</port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(dims, prec)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        self.edges.append((w_lid, 0, lid, 1))
        return lid

    def interpolate_x2(self, src, out_dims):
        """Bilinear x2 upsample (Interpolate opset1, scales anchor)."""
        lid = self.next_id
        self.next_id += 1
        import struct as _s
        # scales const: [1.0, 1.0, 2.0, 2.0] as i32 per-shape-mode? reader wants
        # integral element type for the scales input -> use i32 "shape" mode.
        scales = _s.pack("<4i", 1, 1, 2, 2)
        clid = self.const(scales, [4], precision="I32")
        dims = self._dims_of(src)
        xml = (f'    <layer id="{lid}" name="interp{lid}" type="Interpolate" version="opset1">\n'
               f'      <data mode="linear" shape_calculation_mode="scales" coordinate_transformation_mode="half_pixel"/>\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{self._dims_xml(dims)}</port>\n'
               f'        <port id="1" precision="I32"><dim>4</dim></port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(out_dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        self.edges.append((clid, 0, lid, 1))
        return lid

    def depth_to_space(self, src, out_dims, block_size=2):
        """PixelShuffle: [1, C*bs*bs, H, W] -> [1, C, H*bs, W*bs]."""
        lid = self.next_id
        self.next_id += 1
        dims = self._dims_of(src)
        xml = (f'    <layer id="{lid}" name="dts{lid}" type="DepthToSpace" version="opset1">\n'
               f'      <data mode="depth_first" block_size="{block_size}"/>\n'
               f'      <input>\n'
               f'        <port id="0" precision="FP16">{self._dims_xml(dims)}</port>\n'
               f'      </input>\n'
               f'      <output>\n        {self._port(out_dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        return lid

    def concat(self, srcs, out_dims, axis=1):
        """Concat layers along axis (axis passed as data attr, reader-proven)."""
        lid = self.next_id
        self.next_id += 1
        ports = []
        for s in srcs:
            ports.append(f'<port id="{len(ports)}" precision="FP16">{self._dims_xml(self._dims_of(s))}</port>')
        xml = (f'    <layer id="{lid}" name="cat{lid}" type="Concat" version="opset1">\n'
               f'      <data axis="{axis}"/>\n'
               f'      <input>\n' + "\n".join(ports) + "\n"
               f'      </input>\n'
               f'      <output>\n        {self._port(out_dims)}\n      </output>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        for i, s in enumerate(srcs):
            self.edges.append((s, 0, lid, i))
        return lid

    def result(self, src):
        lid = self.next_id
        self.next_id += 1
        dims = self._dims_of(src)
        prec = self._prec_of(src)
        xml = (f'    <layer id="{lid}" name="output{lid}" type="Result" version="opset1">\n'
               f'      <input>\n        {self._port(dims, prec)}\n      </input>\n'
               f'    </layer>')
        self.layers.append(xml)
        self.layers_by_id[lid] = xml
        self.edges.append((src, 0, lid, 0))
        self.outputs.append(lid)
        return lid

    def _dims_of(self, lid):
        # scan stored layer xml for its output port dims (small graphs, fine)
        xml = self.layers_by_id[lid]
        seg = xml.split("<output>")[1].split("</port>")[0]
        import re
        dims = [int(m) for m in re.findall(r"<dim>(\d+)</dim>", seg)]
        return dims

    def _dims_from_ports(self, lid, port_idx):
        xml = self.layers[lid]
        import re
        inputs = xml.split("<input>")[1].split("</input>")[0]
        ports = inputs.split("<port ")
        seg = ports[port_idx + 1].split("</port>")[0]
        return [int(m) for m in re.findall(r"<dim>(\d+)</dim>", seg)]

    def _prec_of(self, lid):
        xml = self.layers[lid]
        import re
        m = re.search(r'precision="(FP16|FP32|U8|I32)"', xml.split("<output>")[1] if "<output>" in xml else xml)
        return m.group(1) if m else "FP16"

    def to_xml(self):
        layers = "\n".join(self.layers)
        edges = "\n".join(
            f'    <edge from-layer="{a}" from-port="{b}" to-layer="{c}" to-port="{d}"/>'
            for a, b, c, d in self.edges)
        return (f'<?xml version="1.0"?>\n'
                f'<net name="{self.name}" version="10">\n'
                f'  <layers>\n{layers}\n  </layers>\n'
                f'  <edges>\n{edges}\n  </edges>\n'
                f'</net>\n').encode()


# ---------------------------------------------------------------- runtime
class NpuEngine:
    def __init__(self):
        self.ze = C.WinDLL("ze_loader.dll")
        self._bind_core()
        self.hDriver = None
        self.hDevice = None
        self.hContext = None
        self._cmdlist = None
        self.g = {}  # graph fn table as python callables
        self.compiler_version = None
        self._open_device()

    def _bind_core(self):
        ze = self.ze
        voidpp = C.POINTER(C.c_void_p)
        self._voidpp = voidpp
        ze.zeInitDrivers.argtypes = [C.POINTER(C.c_uint32), voidpp, C.c_void_p]
        ze.zeInitDrivers.restype = ze_result_t
        ze.zeDriverGetProperties.argtypes = [C.c_void_p, C.c_void_p]
        ze.zeDriverGetProperties.restype = ze_result_t
        ze.zeDeviceGet.argtypes = [C.c_void_p, C.POINTER(C.c_uint32), voidpp]
        ze.zeDeviceGet.restype = ze_result_t
        ze.zeDeviceGetProperties.argtypes = [C.c_void_p, C.c_void_p]
        ze.zeDeviceGetProperties.restype = ze_result_t
        ze.zeDriverGetExtensionFunctionAddress.argtypes = [C.c_void_p, C.c_char_p, voidpp]
        ze.zeDriverGetExtensionFunctionAddress.restype = ze_result_t
        ze.zeContextCreate.argtypes = [C.c_void_p, C.c_void_p, voidpp]
        ze.zeContextCreate.restype = ze_result_t
        ze.zeContextDestroy.argtypes = [C.c_void_p]
        ze.zeContextDestroy.restype = ze_result_t
        ze.zeMemAllocHost.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, voidpp]
        ze.zeMemAllocHost.restype = ze_result_t
        ze.zeMemFree.argtypes = [C.c_void_p, C.c_void_p]
        ze.zeMemFree.restype = ze_result_t
        ze.zeCommandListCreate.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, voidpp]
        ze.zeCommandListCreate.restype = ze_result_t
        ze.zeCommandListDestroy.argtypes = [C.c_void_p]
        ze.zeCommandListDestroy.restype = ze_result_t
        ze.zeCommandListClose.argtypes = [C.c_void_p]
        ze.zeCommandListClose.restype = ze_result_t
        ze.zeCommandListReset.argtypes = [C.c_void_p]
        ze.zeCommandListReset.restype = ze_result_t
        ze.zeCommandQueueCreate.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, voidpp]
        ze.zeCommandQueueCreate.restype = ze_result_t
        ze.zeCommandQueueDestroy.argtypes = [C.c_void_p]
        ze.zeCommandQueueDestroy.restype = ze_result_t
        ze.zeCommandQueueExecuteCommandLists.argtypes = [C.c_void_p, C.c_uint32, voidpp, C.c_void_p]
        ze.zeCommandQueueExecuteCommandLists.restype = ze_result_t
        ze.zeCommandQueueSynchronize.argtypes = [C.c_void_p, C.c_uint64]
        ze.zeCommandQueueSynchronize.restype = ze_result_t

    def _open_device(self):
        desc = struct.pack("<I I I", ZT_INIT_DRIVER_TYPE_DESC, 0, ZE_INIT_DRIVER_TYPE_FLAG_NPU)
        # pass as raw buffer via ctypes
        class InitDesc(C.Structure):
            _fields_ = [("stype", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]
        d = InitDesc(ZT_INIT_DRIVER_TYPE_DESC, None, ZE_INIT_DRIVER_TYPE_FLAG_NPU)
        num = C.c_uint32(0)
        _err(self.ze.zeInitDrivers(C.byref(num), None, C.byref(d)), "zeInitDrivers(NPU)")
        if num.value == 0:
            raise RuntimeError("no NPU driver")
        drv = (C.c_void_p * num.value)()
        _err(self.ze.zeInitDrivers(C.byref(num), drv, C.byref(d)), "zeInitDrivers(get)")
        self.hDriver = drv[0]

        ndev = C.c_uint32(0)
        _err(self.ze.zeDeviceGet(self.hDriver, C.byref(ndev), None), "zeDeviceGet")
        devs = (C.c_void_p * ndev.value)()
        _err(self.ze.zeDeviceGet(self.hDriver, C.byref(ndev), devs), "zeDeviceGet(get)")
        self.hDevice = None
        for j in range(ndev.value):
            # full 1_2 properties struct: driver writes the whole thing
            class DevProps(C.Structure):
                _fields_ = [
                    ("stype", C.c_uint32), ("pNext", C.c_void_p),
                    ("type", C.c_int), ("vendorId", C.c_uint32), ("deviceId", C.c_uint32),
                    ("flags", C.c_uint32), ("subdeviceId", C.c_uint32), ("coreClockRate", C.c_uint32),
                    ("maxMemAllocSize", C.c_uint64), ("maxHardwareContexts", C.c_uint32),
                    ("maxCommandQueuePriority", C.c_uint32), ("numThreadsPerEU", C.c_uint32),
                    ("physicalEUSimdWidth", C.c_uint32), ("numEUsPerSubslice", C.c_uint32),
                    ("numSubslicesPerSlice", C.c_uint32), ("numSlices", C.c_uint32),
                    ("timerResolution", C.c_uint64), ("timestampValidBits", C.c_uint32),
                    ("kernelTimestampValidBits", C.c_uint32), ("uuid", C.c_byte * 16),
                    ("name", C.c_char * 256),
                    ("numMemoryTypes", C.c_uint32), ("numAsyncComputeQueues", C.c_uint32),
                    ("numAsyncCopyQueues", C.c_uint32),
                ]
            p = DevProps(ZT_DEVICE_PROPERTIES_1_2, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (C.c_byte * 16)(), b"", 0, 0, 0)
            self.ze.zeDeviceGetProperties(devs[j], C.byref(p))
            if p.type == ZE_DEVICE_TYPE_VPU:
                self.hDevice = devs[j]
                break
        if not self.hDevice:
            raise RuntimeError("no VPU device on NPU driver")

        cd = ContextDesc(ZT_CONTEXT_DESC, None, 0)
        hctx = C.c_void_p()
        _err(self.ze.zeContextCreate(self.hDriver, C.byref(cd), C.byref(hctx)), "zeContextCreate")
        self.hContext = hctx

        # graph extension table
        pfn = C.c_void_p()
        _err(self.ze.zeDriverGetExtensionFunctionAddress(
            self.hDriver, ZE_EXTENSION_GRAPH, C.byref(pfn)), "graph ext addr")
        tbl = C.cast(pfn, C.POINTER(C.c_void_p))
        raw = []
        for k in range(len(GRAPH_FN_NAMES)):
            v = tbl[k]
            if not v:
                break
            raw.append(v)
        fp = C.CFUNCTYPE(ze_result_t)  # generic prototype template
        self._fn_addrs = raw
        voidpp = self._voidpp

        # bind the ones we need, each with a concrete prototype
        def bind(idx, proto):
            if idx >= len(raw):
                return None
            return proto(raw[idx])

        P_DEVGRAPH = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_void_p)
        P_CREATE2 = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_void_p, C.POINTER(GraphDesc2), voidpp)
        P_DESTROY = C.CFUNCTYPE(ze_result_t, C.c_void_p)
        P_GPROPS = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.POINTER(GraphProperties))
        P_GPROPS2 = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.POINTER(GraphProperties2))
        P_ARGPROPS = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_uint32, C.POINTER(GraphArgumentProperties))
        P_SETARG = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_uint32, C.c_void_p)
        P_APPEND_INIT = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, voidpp)
        P_APPEND_EXEC = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, voidpp)
        P_GRAPH_INIT = C.CFUNCTYPE(ze_result_t, C.c_void_p)
        P_BUILDLOG = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.POINTER(C.c_uint32), C.c_char_p)
        P_GETNATIVE2 = C.CFUNCTYPE(ze_result_t, C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_void_p))

        self.g = {
            "DeviceGetGraphProperties": bind(8, P_DEVGRAPH),
            "DeviceGetGraphProperties2": bind(19, P_DEVGRAPH),
            "Create2": bind(16, P_CREATE2),
            "Destroy": bind(1, P_DESTROY),
            "GetProperties": bind(2, P_GPROPS),
            "GetProperties2": bind(21, P_GPROPS2),
            "GetArgumentProperties": bind(3, P_ARGPROPS),
            "SetArgumentValue": bind(4, P_SETARG),
            "AppendGraphInitialize": bind(5, P_APPEND_INIT),
            "AppendGraphExecute": bind(6, P_APPEND_EXEC),
            "GraphInitialize": bind(22, P_GRAPH_INIT),
            "BuildLogGetString": bind(15, P_BUILDLOG),
            "GetNativeBinary2": bind(20, P_GETNATIVE2),
        }

        props = DeviceGraphProperties(GX_DEVICE_GRAPH_PROPERTIES, None, 0, GraphCompilerVersion(), 0, 0)
        _err(self.g["DeviceGetGraphProperties"](self.hDevice, C.byref(props)),
             "DeviceGetGraphProperties")
        self.compiler_version = (props.compilerVersion.major, props.compilerVersion.minor)
        self.graph_formats = props.graphFormatsSupported
        self.max_opset = props.maxOVOpsetVersionSupported

    # ------------------------------------------------------------- graph api
    def create_graph(self, ir_or_blob, weights=b"", build_flags=b"", ir_header=None, native=False):
        """Create a graph. native=False: ir_or_blob is our IR xml (compiled
        in-driver). native=True: it's a previously compiled ELF binary
        (GetNativeBinary2) loaded via ZE_GRAPH_FORMAT_NATIVE -- zero compile."""
        if native:
            blob = ir_or_blob
        else:
            hdr = ir_header or self.compiler_version
            blob = build_ngraph_lite_container(ir_or_blob, weights, hdr)
        fmt = GX_FORMAT_NATIVE if native else GX_FORMAT_NGRAPH_LITE
        buf = (C.c_ubyte * len(blob)).from_buffer_copy(blob)
        desc = GraphDesc2()
        desc.stype = GX_GRAPH_DESC_2
        desc.pNext = None
        desc.format = fmt
        desc.inputSize = len(blob)
        desc.pInput = C.cast(buf, C.c_void_p)
        desc.pBuildFlags = build_flags
        desc.flags = 0
        h = C.c_void_p()
        r = self.g["Create2"](self.hContext, self.hDevice, C.byref(desc), C.byref(h))
        if r != ZE_RESULT_SUCCESS:
            log = self._last_build_log(h)
            raise RuntimeError(f"zeGraphCreate2 failed 0x{r & 0xFFFFFFFF:08X}\nbuild log: {log}")
        return NpuGraph(self, h)

    def _last_build_log(self, hGraph):
        fn = self.g["BuildLogGetString"]
        if not fn or not hGraph.value:
            return "(no log handle)"
        size = C.c_uint32(0)
        r1 = fn(hGraph, C.byref(size), None)
        if r1 != ZE_RESULT_SUCCESS or size.value == 0:
            return f"(log unavailable rc={r1})"
        buf = C.create_string_buffer(size.value)
        fn(hGraph, C.byref(size), buf)
        return buf.value.decode(errors="replace")

    # ------------------------------------------------------------ execution
    def alloc_host(self, size):
        ad = HostMemAllocDesc(ZT_HOST_MEM_ALLOC_DESC, None, ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED)
        p = C.c_void_p()
        _err(self.ze.zeMemAllocHost(self.hContext, C.byref(ad), size, 64, C.byref(p)),
             f"zeMemAllocHost({size})")
        return p

    def free_host(self, p):
        if p and p.value:
            self.ze.zeMemFree(self.hContext, p)

    def _make_cmdlist(self):
        ld = ListDesc(ZT_COMMAND_LIST_DESC, None, 0)
        h = C.c_void_p()
        _err(self.ze.zeCommandListCreate(self.hContext, self.hDevice, C.byref(ld), C.byref(h)),
             "zeCommandListCreate")
        return h

    def _make_queue(self):
        qd = QueueDesc(ZT_COMMAND_QUEUE_DESC, None, 0, 0, 0, 0, 0)
        h = C.c_void_p()
        _err(self.ze.zeCommandQueueCreate(self.hContext, self.hDevice, C.byref(qd), C.byref(h)),
             "zeCommandQueueCreate")
        return h

    def submit_and_wait(self, hCmdList):
        _err(self.ze.zeCommandListClose(hCmdList), "zeCommandListClose")
        hq = self._make_queue()
        try:
            lists = (C.c_void_p * 1)(hCmdList)
            _err(self.ze.zeCommandQueueExecuteCommandLists(
                hq, 1, C.cast(lists, C.POINTER(C.c_void_p)), None),
                "ExecuteCommandLists")
            _err(self.ze.zeCommandQueueSynchronize(hq, 0xFFFFFFFFFFFFFFFF),
                 "QueueSynchronize")
        finally:
            self.ze.zeCommandQueueDestroy(hq)


class NpuGraph:
    def __init__(self, engine, handle):
        self.e = engine
        self.h = handle
        self._ptrs = None
        self.args = self._read_args()

    def _read_args(self):
        gp = GraphProperties(GX_GRAPH_PROPERTIES, None, 0)
        _err(self.e.g["GetProperties"](self.h, C.byref(gp)), "GraphGetProperties")
        args = []
        for i in range(gp.numGraphArgs):
            ap = GraphArgumentProperties()
            ap.stype = GX_GRAPH_ARGUMENT_PROPERTIES
            _err(self.e.g["GetArgumentProperties"](self.h, i, C.byref(ap)),
                 f"GetArgumentProperties({i})")
            name = ap.name.decode(errors="replace")
            dims = [ap.dims[k] for k in range(5)]
            args.append(NpuTensorInfo(name, ap.type, dims, ap.devicePrecision))
        return args

    @property
    def inputs(self):
        return [a for a in self.args if a.arg_type == GX_ARG_TYPE_INPUT]

    @property
    def outputs(self):
        return [a for a in self.args if a.arg_type == GX_ARG_TYPE_OUTPUT]

    def set_argument(self, index, host_ptr):
        _err(self.e.g["SetArgumentValue"](self.h, index, host_ptr), "SetArgumentValue")

    def bind_buffers(self):
        """Allocate host buffers for every argument, bind them, and return
        (in_ptrs, out_ptrs) keyed by argument index."""
        import numpy as np
        ptrs = []
        for a in self.args:
            p = self.e.alloc_host(a.nbytes)
            ptrs.append(p)
            self.set_argument(self.args.index(a), p)
        self._ptrs = ptrs
        return ptrs

    def write_input(self, arg_index, np_array):
        """np_array float32 -> host buffer (auto fp16 pack when device is FP16)."""
        import numpy as np
        a = self.args[arg_index]
        p = self._ptrs[arg_index]
        if a.precision_name == "FP16":
            h = np.frombuffer((C.c_uint16 * a.numel).from_address(p.value), dtype=np.uint16)
            h[:] = np.frombuffer(np_array.astype(np.float16).tobytes(), dtype=np.uint16)
        elif a.precision_name == "FP32":
            h = np.ctypeslib.as_array(C.cast(p, C.POINTER(C.c_float)), shape=(a.numel,))
            h[:] = np_array.reshape(-1)
        else:
            raise RuntimeError(f"unsupported input precision {a.precision_name}")

    def read_output(self, arg_index):
        """Host buffer -> np.float32 (auto fp16 unpack when device is FP16)."""
        import numpy as np
        a = self.args[arg_index]
        p = self._ptrs[arg_index]
        if a.precision_name == "FP16":
            raw = np.frombuffer((C.c_uint16 * a.numel).from_address(p.value), dtype=np.uint16).copy()
            return raw.view(np.float16).astype(np.float32).reshape(a.dims)
        elif a.precision_name == "FP32":
            raw = np.ctypeslib.as_array(C.cast(p, C.POINTER(C.c_float)), shape=(a.numel,)).copy()
            return raw.reshape(a.dims)
        raise RuntimeError(f"unsupported output precision {a.precision_name}")

    def init(self):
        cl = self.e._make_cmdlist()
        try:
            _err(self.e.g["AppendGraphInitialize"](cl, self.h, None, 0, None),
                 "AppendGraphInitialize")
            self.e.submit_and_wait(cl)
        finally:
            self.e.ze.zeCommandListDestroy(cl)

    def execute(self):
        cl = self.e._cmdlist
        if cl is None:
            cl = self.e._cmdlist = self.e._make_cmdlist()
        self.e.ze.zeCommandListReset(cl)
        _err(self.e.g["AppendGraphExecute"](cl, self.h, None, None, 0, None),
             "AppendGraphExecute")
        self.e.submit_and_wait(cl)

    def destroy(self):
        if self.h and self.h.value:
            self.e.g["Destroy"](self.h)
            self.h = C.c_void_p()


def main():
    print("== opening NPU ==")
    e = NpuEngine()
    print(f"compiler v{e.compiler_version[0]}.{e.compiler_version[1]} "
          f"formats=0x{e.graph_formats:X} maxOpset={e.max_opset}")

    print("\n== building ReLU IR (own emitter, opset11, FP16) ==")
    ir = IrBuilder("relu_probe")
    x = ir.parameter([1, 4, 8, 8])
    y = ir.relu(x)
    ir.result(y)
    xml = ir.to_xml()
    print(xml.decode())

    print("== zeGraphCreate2 (NGRAPH_LITE, in-driver compile) ==")
    import time
    t0 = time.time()
    g = e.create_graph(xml)
    print(f"graph created in {time.time() - t0:.1f}s")
    for a in g.args:
        print(" ", a)

    # allocate and bind
    xs = ir.inputs[0]
    yo = ir.outputs[0]
    in_info, out_info = g.inputs[0], g.outputs[0]
    p_in = e.alloc_host(in_info.nbytes)
    p_out = e.alloc_host(out_info.nbytes)
    g.set_argument(g.args.index(in_info), p_in)
    g.set_argument(g.args.index(out_info), p_out)

    # fill input with negative+positive values
    import numpy as np
    data = np.arange(in_info.numel, dtype=np.float32) - in_info.numel / 2
    data *= 0.25
    buf = (C.c_float * in_info.numel).from_address(p_in.value) if in_info.precision_name == "FP32" else None
    if in_info.precision_name == "FP16":
        h16 = np.frombuffer((C.c_uint16 * in_info.numel).from_address(p_in.value), dtype=np.uint16)
        h16[:] = np.frombuffer((data.astype(np.float16)).tobytes(), dtype=np.uint16)
    else:
        buf[:] = data.tolist()

    print("== initialize (graph init stage) ==")
    t0 = time.time()
    cl = e._make_cmdlist()
    _err(e.g["AppendGraphInitialize"](cl, g.h, None, 0, None), "AppendGraphInitialize")
    e.submit_and_wait(cl)
    print(f"initialized in {time.time() - t0:.1f}s")

    print("== execute ==")
    t0 = time.time()
    e.ze.zeCommandListReset(cl)
    _err(e.g["AppendGraphExecute"](cl, g.h, None, None, 0, None), "AppendGraphExecute")
    e.submit_and_wait(cl)
    dt = time.time() - t0
    print(f"executed in {dt * 1000:.2f} ms")

    if out_info.precision_name == "FP16":
        out = np.frombuffer((C.c_uint16 * out_info.numel).from_address(p_out.value),
                            dtype=np.uint16).copy()
        out = out.view(np.float16).astype(np.float32)
    else:
        out = np.ctypeslib.as_array(C.cast(p_out, C.POINTER(C.c_float)),
                                    shape=(out_info.numel,)).copy()
    expected = np.maximum(data, 0).astype(np.float32)
    ok = np.allclose(out, expected, rtol=0, atol=2e-2)
    print(f"numeric check: {'PASS' if ok else 'FAIL'}  max|diff|={np.abs(out - expected).max():.5f}")
    print("sample out:", out[:8])

    e.ze.zeCommandListDestroy(cl)
    e.free_host(p_in)
    e.free_host(p_out)
    g.destroy()
    e.ze.zeContextDestroy(e.hContext)
    print("\nEND-TO-END OK" if ok else "\nEND-TO-END FAILED")


if __name__ == "__main__":
    main()

