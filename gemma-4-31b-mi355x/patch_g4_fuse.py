"""Fuse Gemma 4 MLP GELU(tanh)*up + MXFP4 activation quant into one aiter Triton kernel feeding the asm
gemm_a4w4 down_proj (replaces Inductor gelu_mul + per_1x32_f4_quant_hip). Enabled when G4_FUSE_GELU=1.
Assumes down_proj runs vLLM's AITER asm MXFP4 path (weight (16,16)-shuffled, scale swizzled), which is the
case for amd/gemma-4-31B-it-MXFP4 on gfx950 with VLLM_ROCM_USE_AITER=1."""
import os, sys, importlib.util


def _pkg_dir(name):  # locate without importing: aiter probes the GPU on import, and docker build has none
    return os.path.dirname(importlib.util.find_spec(name).origin)


p = os.path.join(_pkg_dir("vllm"), "model_executor/models/gemma4.py")
s = open(p).read()
if "g4_gelu_mxfp4_gemm" in s:
    print("patch_g4_fuse: already applied", file=sys.stderr); sys.exit(0)
op = '''

def _g4_gelu_mxfp4_gemm(gate_up: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    import aiter
    from aiter.ops.triton.activation import act_mul_and_mxfp4_quant
    M = gate_up.shape[0]
    xq, xs = act_mul_and_mxfp4_quant(gate_up, "gelu_tanh", shuffle=True)
    xq = xq.view(aiter.dtypes.fp4x2)
    xs = xs.view(aiter.dtypes.fp8_e8m0)
    y = aiter.gemm_a4w4(xq, weight.view(xq.dtype), xs, weight_scale.view(xs.dtype),
                        dtype=torch.bfloat16, bpreshuffle=True)
    return y[:M]


def _g4_gelu_mxfp4_gemm_fake(gate_up: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((gate_up.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=gate_up.device)


from vllm.utils.torch_utils import direct_register_custom_op as _g4_reg
_g4_reg(op_name="g4_gelu_mxfp4_gemm", op_func=_g4_gelu_mxfp4_gemm, mutates_args=[],
        fake_impl=_g4_gelu_mxfp4_gemm_fake)
_G4_FUSE_GELU = os.environ.get("G4_FUSE_GELU", "0") == "1"
'''
anchor = "\nclass Gemma4MLP(nn.Module):"
assert anchor in s
s = s.replace(anchor, op + anchor, 1)
if "\nimport os\n" not in s:
    s = s.replace("\nimport regex as re\n", "\nimport os\nimport regex as re\n", 1)
old = """        self.act_fn = get_act_and_mul_fn(hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)"""
new = """        self.act_fn = get_act_and_mul_fn(hidden_activation)
        # Single rank only: the fused path calls the GEMM directly and would skip down_proj's
        # row-parallel all-reduce, leaving each rank with a partial MLP output under TP > 1.
        self._g4_fuse = (_G4_FUSE_GELU and hidden_activation in ("gelu_pytorch_tanh", "gelu_tanh")
                         and self.down_proj.tp_size == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if self._g4_fuse and hasattr(self.down_proj, "weight_scale"):
            return torch.ops.vllm.g4_gelu_mxfp4_gemm(gate_up, self.down_proj.weight, self.down_proj.weight_scale)
        x = self.act_fn(gate_up)"""
assert old in s, "Gemma4MLP body not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)

# aiter's _act_mul_and_dynamic_mxfp4_quant_kernel declares scaleM_pad (= cdiv(M, 256) * 256) as tl.constexpr, so
# every new batch size rounded to 256 triggers a ~0.2 s Triton JIT compile mid-serving (64+ variants up to 16K
# tokens). It is only used in a mask comparison, so make it a runtime argument.
ap = os.path.join(_pkg_dir("aiter"), "ops/triton/_triton_kernels/activation.py")
a = open(ap).read()
head = "def _act_mul_and_dynamic_mxfp4_quant_kernel("
i = a.index(head); j = a.index("):", i)
sig = a[i:j]
if "scaleM_pad: tl.constexpr" in sig:
    a = a[:i] + sig.replace("scaleM_pad: tl.constexpr", "scaleM_pad") + a[j:]
    open(ap, "w").write(a)
# Positive check, so a respelled or removed parameter fails the build instead of shipping the recompile storm.
import re
_k = open(ap).read(); _i = _k.index(head); _sig = _k[_i:_k.index("):", _i)]
assert re.search(r"\bscaleM_pad\s*,", _sig) and not re.search(r"\bscaleM_pad\s*:", _sig), \
    "aiter _act_mul_and_dynamic_mxfp4_quant_kernel: scaleM_pad is not a plain runtime argument"
print(f"patch_g4_fuse: applied (G4_FUSE_GELU={os.environ.get('G4_FUSE_GELU')})", file=sys.stderr)
