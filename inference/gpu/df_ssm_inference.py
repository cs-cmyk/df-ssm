"""
DF-SSM v9 — cuBLAS INT8 Scaffold

Scaffold matmul uses torch._int_mm (INT8 tensor cores, 624 TOPS on A100).
Binary weights (+1/-1) stored as int8. Input quantized to int8 on the fly.
Custom CUDA only for conv step and SSM step (stateful ops).

Build:  pip install -e . --no-build-isolation --force-reinstall --no-deps
Run:    python df_ssm_inference.py model.dfssm --batch_size 32 --n_tokens 128
"""

import time, torch, torch.nn.functional as F
import df_ssm_kernels

D_MODEL, D_INNER, D_XBC = 2048, 4096, 4352
D_STATE, NHEADS, HEADDIM = 128, 64, 32
D_CONV, LORA_RANK, N_LAYERS = 4, 16, 48
VOCAB_SIZE, IN_PROJ_OUT = 50288, 8512
SSM_DIM = NHEADS * HEADDIM
HEADER_SIZE = 64


def _unpack_to_int8(packed, out_features, in_features):
    """Unpack bit-packed uint32 weights to int8 (+1/-1)."""
    w = torch.zeros(out_features, in_features, dtype=torch.int8, device=packed.device)
    for b in range(32):
        bits = (packed >> b) & 1  # (out_features, packed_cols)
        w[:, b::32] = (2 * bits - 1).to(torch.int8)
    return w


class LayerWeights:
    __slots__ = (
        # INT8 scaffold (pre-transposed for torch._int_mm)
        "in_w_T", "in_scale",      # (D_MODEL, IN_PROJ_OUT) int8, (IN_PROJ_OUT,) fp16
        "out_w_T", "out_scale",    # (D_INNER, D_MODEL) int8, (D_MODEL,) fp16
        # LoRA (keep as float for simplicity)
        "in_lora_A", "in_lora_B",  # (IN_PROJ_OUT, LORA_RANK) float, (LORA_RANK, D_MODEL) float
        "out_lora_A", "out_lora_B",
        # SSM params
        "conv_weight", "conv_bias",
        "A_log", "D", "dt_bias",
        "norm_weight", "layer_norm_weight",
    )


class DFSSMModel:
    def __init__(self, path, device="cuda", batch_size=1):
        self.device = device
        self.batch_size = batch_size
        self.layers = []
        self._load(path)
        self._alloc_state()
        self._graph = None

    def _read(self, f, shape, dtype):
        numel = 1
        for s in shape: numel *= s
        nb = numel * torch.tensor([], dtype=dtype).element_size()
        return torch.frombuffer(bytearray(f.read(nb)), dtype=dtype).reshape(shape).to(self.device)

    def _load(self, path):
        with open(path, "rb") as f:
            assert f.read(HEADER_SIZE)[:8] == b"DFSSM001"
            self.embedding = self._read(f, (VOCAB_SIZE, D_MODEL), torch.float16)

            for _ in range(N_LAYERS):
                w = LayerWeights()

                # Load packed → unpack to int8 → transpose for torch._int_mm
                in_packed = self._read(f, (IN_PROJ_OUT, D_MODEL // 32), torch.int32)
                w.in_scale = self._read(f, (IN_PROJ_OUT,), torch.float16)
                in_w = _unpack_to_int8(in_packed, IN_PROJ_OUT, D_MODEL)
                w.in_w_T = in_w.T.contiguous()  # (D_MODEL, IN_PROJ_OUT) int8
                del in_packed, in_w

                # LoRA: load int8 + scales, convert to float for matmul
                in_lA = self._read(f, (IN_PROJ_OUT, LORA_RANK), torch.int8)
                in_lA_s = self._read(f, (IN_PROJ_OUT,), torch.float16)
                in_lB = self._read(f, (LORA_RANK, D_MODEL), torch.int8)
                in_lB_s = self._read(f, (LORA_RANK,), torch.float16)
                w.in_lora_A = (in_lA.float() * in_lA_s.float().unsqueeze(1)).half()
                w.in_lora_B = (in_lB.float() * in_lB_s.float().unsqueeze(1)).half()
                del in_lA, in_lA_s, in_lB, in_lB_s

                out_packed = self._read(f, (D_MODEL, D_INNER // 32), torch.int32)
                w.out_scale = self._read(f, (D_MODEL,), torch.float16)
                out_w = _unpack_to_int8(out_packed, D_MODEL, D_INNER)
                w.out_w_T = out_w.T.contiguous()  # (D_INNER, D_MODEL) int8
                del out_packed, out_w

                out_lA = self._read(f, (D_MODEL, LORA_RANK), torch.int8)
                out_lA_s = self._read(f, (D_MODEL,), torch.float16)
                out_lB = self._read(f, (LORA_RANK, D_INNER), torch.int8)
                out_lB_s = self._read(f, (LORA_RANK,), torch.float16)
                w.out_lora_A = (out_lA.float() * out_lA_s.float().unsqueeze(1)).half()
                w.out_lora_B = (out_lB.float() * out_lB_s.float().unsqueeze(1)).half()
                del out_lA, out_lA_s, out_lB, out_lB_s

                w.conv_weight = self._read(f, (D_XBC, D_CONV), torch.float16)
                w.conv_bias = self._read(f, (D_XBC,), torch.float16)
                w.A_log = self._read(f, (NHEADS,), torch.float32)
                w.D = self._read(f, (NHEADS,), torch.float32)
                w.dt_bias = self._read(f, (NHEADS,), torch.float32)
                w.norm_weight = self._read(f, (D_INNER,), torch.float16)
                w.layer_norm_weight = self._read(f, (D_MODEL,), torch.float16)

                self.layers.append(w)

            self.norm_f_weight = self._read(f, (D_MODEL,), torch.float16)

    def _alloc_state(self):
        B, d = self.batch_size, self.device
        self.h_states = [
            torch.zeros(B, NHEADS, HEADDIM, D_STATE, dtype=torch.float16, device=d)
            for _ in range(N_LAYERS)
        ]
        self.conv_states = [
            torch.zeros(B, D_XBC, D_CONV - 1, dtype=torch.float16, device=d)
            for _ in range(N_LAYERS)
        ]
        # Pre-allocate intermediates for CUDA graph
        # INT8 buffers padded to 32 (torch._int_mm requires batch >= 17)
        B_pad = max(B, 32)
        self.x = torch.empty(B, D_MODEL, dtype=torch.float16, device=d)
        self.x_norm = torch.empty(B, D_MODEL, dtype=torch.float16, device=d)
        self.x_int8 = torch.zeros(B_pad, D_MODEL, dtype=torch.int8, device=d)
        self.proj = torch.empty(B, IN_PROJ_OUT, dtype=torch.float16, device=d)
        self.y_buf = torch.zeros(B, D_INNER, dtype=torch.float16, device=d)
        self.y_norm = torch.empty(B, D_INNER, dtype=torch.float16, device=d)
        self.y_int8 = torch.zeros(B_pad, D_INNER, dtype=torch.int8, device=d)
        self.out_buf = torch.empty(B, D_MODEL, dtype=torch.float16, device=d)
        self.norm_out = torch.empty(B, D_MODEL, dtype=torch.float16, device=d)
        self.x_scale = torch.empty(B, 1, dtype=torch.float16, device=d)
        self.B_pad = B_pad

    def reset_state(self):
        for h in self.h_states: h.zero_()
        for c in self.conv_states: c.zero_()

    @staticmethod
    def _rms_norm(x, weight):
        """RMS norm: x * rsqrt(mean(x^2) + eps) * weight"""
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + 1e-5)) * weight

    def _quantize_to_int8(self, x, scale_buf, int8_buf):
        """Per-token INT8 quantization into pre-allocated padded buffer."""
        B = x.size(0)
        scale_buf.copy_(x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 127.0)
        int8_buf[:B].copy_((x / scale_buf).round().clamp(-128, 127).to(torch.int8))
        return scale_buf, int8_buf

    def _scaffold_matmul(self, x_int8_padded, x_scale, w_T, w_scale):
        """INT8 scaffold using pre-padded int8 buffer."""
        B = self.batch_size
        out_int32 = torch._int_mm(x_int8_padded, w_T)  # (B_pad, out_features)
        return (out_int32[:B].half() * x_scale) * w_scale.unsqueeze(0)

    def _run_layers(self):
        for l in range(N_LAYERS):
            w = self.layers[l]

            # Step 1: RMS Norm
            self.x_norm.copy_(self._rms_norm(self.x, w.layer_norm_weight))

            # Step 2: in_proj scaffold (INT8 tensor cores)
            self._quantize_to_int8(self.x_norm, self.x_scale, self.x_int8)
            self.proj.copy_(self._scaffold_matmul(
                self.x_int8, self.x_scale, w.in_w_T, w.in_scale))

            # Step 3: in_proj LoRA
            self.proj.addmm_(self.x_norm @ w.in_lora_B.T, w.in_lora_A.T)

            # Step 4: Split proj → z, xBC, dt
            z = self.proj[:, :D_INNER]
            xBC = self.proj[:, D_INNER:D_INNER + D_XBC].contiguous()
            dt = self.proj[:, D_INNER + D_XBC:D_INNER + D_XBC + NHEADS].contiguous()

            # Step 5: Conv step (custom CUDA)
            df_ssm_kernels.conv_step(
                self.conv_states[l], xBC, w.conv_weight, w.conv_bias, D_XBC)

            # Step 6: SSM step (custom CUDA)
            x_ssm = xBC[:, :SSM_DIM]
            B_param = xBC[:, D_INNER:D_INNER + D_STATE]
            C_param = xBC[:, D_INNER + D_STATE:D_INNER + 2 * D_STATE]
            self.y_buf.zero_()
            df_ssm_kernels.ssm_step(
                self.h_states[l], xBC, dt,
                w.dt_bias, w.A_log, w.D, self.y_buf)

            # Step 7: Gate + norm
            silu_z = z * torch.sigmoid(z)
            gated = self.y_buf * silu_z
            self.y_norm.copy_(self._rms_norm(gated, w.norm_weight))

            # Step 8: out_proj scaffold (INT8 tensor cores)
            self._quantize_to_int8(self.y_norm, self.x_scale, self.y_int8)
            self.out_buf.copy_(self._scaffold_matmul(
                self.y_int8, self.x_scale, w.out_w_T, w.out_scale))

            # Step 9: out_proj LoRA
            self.out_buf.addmm_(self.y_norm @ w.out_lora_B.T, w.out_lora_A.T)

            # Step 10: Residual
            self.x.add_(self.out_buf)

        # Final norm
        self.norm_out.copy_(self._rms_norm(self.x, self.norm_f_weight))

    def _capture_graph(self):
        self._run_layers()
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._run_layers()

    def forward(self, token_ids):
        self.x.copy_(self.embedding[token_ids])
        if self._graph is None:
            self._capture_graph()
        self._graph.replay()
        return self.norm_out @ self.embedding.T

    def generate(self, n_tokens=128):
        B = self.batch_size
        self.reset_state()
        tokens = torch.zeros(B, dtype=torch.long, device=self.device)
        _ = self.forward(tokens)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_tokens):
            logits = self.forward(tokens)
            tokens = logits.argmax(dim=-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total = n_tokens * B
        print(f"batch={B}  {n_tokens} steps  {total} total tokens")
        print(f"  {elapsed*1000:.1f} ms  ({total/elapsed:.0f} tok/s total, "
              f"{elapsed/n_tokens*1e6:.0f} μs/step, {total/elapsed/B:.0f} tok/s/seq)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_tokens", type=int, default=128)
    a = p.parse_args()
    DFSSMModel(a.model, batch_size=a.batch_size).generate(n_tokens=a.n_tokens)
