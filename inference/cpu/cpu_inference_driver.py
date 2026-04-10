#!/usr/bin/env python3
"""
CPU inference driver — bit-packed scaffold, AVX-512 mask ops, state cached.

Build:
    gcc -O3 -march=native -fopenmp -shared -fPIC \
        -o libcpu_inference.so cpu_inference.c -lm

Run:
    OMP_NUM_THREADS=4 python cpu_inference_driver.py model.dfssm --n_tokens 128
"""

import ctypes
import time
import numpy as np

lib = ctypes.CDLL('./libcpu_inference.so')

D_MODEL    = 2048
D_INNER    = 4096
D_XBC      = 4352
D_STATE    = 128
NHEADS     = 64
HEADDIM    = 32
D_CONV     = 4
LORA_RANK  = 16
N_LAYERS   = 48
VOCAB_SIZE = 50288
IN_PROJ_OUT = 8512
HEADER_SIZE = 64

F_P  = ctypes.POINTER(ctypes.c_float)
I_P  = ctypes.POINTER(ctypes.c_int)
I8_P = ctypes.POINTER(ctypes.c_int8)
U32_P = ctypes.POINTER(ctypes.c_uint32)

lib.layer_forward.restype = None
lib.layer_forward.argtypes = [
    F_P,            # x
    U32_P, F_P, I_P,   # in_packed, in_scale, in_row_sums
    U32_P, F_P, I_P,   # out_packed, out_scale, out_row_sums
    I8_P, F_P, I8_P, F_P,  # in LoRA A,As,B,Bs
    I8_P, F_P, I8_P, F_P,  # out LoRA A,As,B,Bs
    F_P, F_P,       # conv_w, conv_b
    F_P, F_P, F_P,  # A_log, D, dt_bias
    F_P, F_P,       # norm_weight, ln_weight
    F_P, F_P,       # h_state, conv_state
    F_P, F_P, F_P, F_P,  # scratch
]

lib.precompute_row_sums.restype = None
lib.precompute_row_sums.argtypes = [U32_P, I_P, ctypes.c_int, ctypes.c_int]

lib.final_rms_norm.restype = None
lib.final_rms_norm.argtypes = [F_P, F_P, F_P, ctypes.c_int]

lib.embedding_lookup.restype = None
lib.embedding_lookup.argtypes = [F_P, ctypes.c_int, F_P, ctypes.c_int]

def ptr_f(a): return a.ctypes.data_as(F_P)
def ptr_i(a): return a.ctypes.data_as(I_P)
def ptr_i8(a): return a.ctypes.data_as(I8_P)
def ptr_u32(a): return a.ctypes.data_as(U32_P)


class LW:
    __slots__ = (
        'in_packed', 'in_scale', 'in_row_sums',
        'out_packed', 'out_scale', 'out_row_sums',
        'in_lA', 'in_lAs', 'in_lB', 'in_lBs',
        'out_lA', 'out_lAs', 'out_lB', 'out_lBs',
        'conv_w', 'conv_b', 'A_log', 'D', 'dt_bias',
        'norm_w', 'ln_w',
    )


class CPUModel:
    def __init__(self, path):
        self.layers = []
        self._load(path)
        self._alloc()

    def _rd(self, f, shape, dtype):
        n = 1
        for s in shape: n *= s
        buf = f.read(n * np.dtype(dtype).itemsize)
        return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()

    def _load(self, path):
        print(f"Loading {path}...")
        with open(path, 'rb') as f:
            hdr = f.read(HEADER_SIZE)
            assert hdr[:8] == b'DFSSM001'

            # Embedding: FP16 → FP32
            self.embedding = self._rd(f, (VOCAB_SIZE, D_MODEL), np.float16).astype(np.float32)
            print(f"  Embedding: {self.embedding.nbytes/1e6:.0f} MB (fp32)")

            scaffold_total = 0
            for li in range(N_LAYERS):
                w = LW()

                # Bit-packed scaffold (stays packed!)
                w.in_packed = self._rd(f, (IN_PROJ_OUT, D_MODEL // 32), np.uint32)
                w.in_scale = self._rd(f, (IN_PROJ_OUT,), np.float16).astype(np.float32)
                w.in_row_sums = np.empty(IN_PROJ_OUT, dtype=np.int32)
                lib.precompute_row_sums(ptr_u32(w.in_packed), ptr_i(w.in_row_sums),
                                        IN_PROJ_OUT, D_MODEL // 32)

                w.in_lA = self._rd(f, (IN_PROJ_OUT, LORA_RANK), np.int8)
                w.in_lAs = self._rd(f, (IN_PROJ_OUT,), np.float16).astype(np.float32)
                w.in_lB = self._rd(f, (LORA_RANK, D_MODEL), np.int8)
                w.in_lBs = self._rd(f, (LORA_RANK,), np.float16).astype(np.float32)

                w.out_packed = self._rd(f, (D_MODEL, D_INNER // 32), np.uint32)
                w.out_scale = self._rd(f, (D_MODEL,), np.float16).astype(np.float32)
                w.out_row_sums = np.empty(D_MODEL, dtype=np.int32)
                lib.precompute_row_sums(ptr_u32(w.out_packed), ptr_i(w.out_row_sums),
                                        D_MODEL, D_INNER // 32)

                w.out_lA = self._rd(f, (D_MODEL, LORA_RANK), np.int8)
                w.out_lAs = self._rd(f, (D_MODEL,), np.float16).astype(np.float32)
                w.out_lB = self._rd(f, (LORA_RANK, D_INNER), np.int8)
                w.out_lBs = self._rd(f, (LORA_RANK,), np.float16).astype(np.float32)

                w.conv_w = self._rd(f, (D_XBC, D_CONV), np.float16).astype(np.float32)
                w.conv_b = self._rd(f, (D_XBC,), np.float16).astype(np.float32)
                w.A_log = self._rd(f, (NHEADS,), np.float32)
                w.D = self._rd(f, (NHEADS,), np.float32)
                w.dt_bias = self._rd(f, (NHEADS,), np.float32)
                w.norm_w = self._rd(f, (D_INNER,), np.float32)
                w.ln_w = self._rd(f, (D_MODEL,), np.float32)

                scaffold_total += w.in_packed.nbytes + w.out_packed.nbytes
                self.layers.append(w)

            self.norm_f_w = self._rd(f, (D_MODEL,), np.float32)

        total = scaffold_total + self.embedding.nbytes
        print(f"  Scaffold (bit-packed): {scaffold_total/1e6:.0f} MB")
        print(f"  Total in RAM: {total/1e6:.0f} MB")

    def _alloc(self):
        self.h_states = [np.zeros((NHEADS, HEADDIM, D_STATE), dtype=np.float32)
                         for _ in range(N_LAYERS)]
        self.conv_states = [np.zeros((D_XBC, D_CONV - 1), dtype=np.float32)
                            for _ in range(N_LAYERS)]
        self.x = np.empty(D_MODEL, dtype=np.float32)
        self.x_norm = np.empty(D_MODEL, dtype=np.float32)
        self.proj = np.empty(IN_PROJ_OUT, dtype=np.float32)
        self.y_buf = np.empty(D_INNER, dtype=np.float32)
        self.out_buf = np.empty(D_MODEL, dtype=np.float32)
        self.norm_out = np.empty(D_MODEL, dtype=np.float32)

    def reset(self):
        for h in self.h_states: h[:] = 0
        for c in self.conv_states: c[:] = 0

    def forward(self, token_id):
        lib.embedding_lookup(ptr_f(self.embedding), token_id, ptr_f(self.x), D_MODEL)

        for l in range(N_LAYERS):
            w = self.layers[l]
            lib.layer_forward(
                ptr_f(self.x),
                ptr_u32(w.in_packed), ptr_f(w.in_scale), ptr_i(w.in_row_sums),
                ptr_u32(w.out_packed), ptr_f(w.out_scale), ptr_i(w.out_row_sums),
                ptr_i8(w.in_lA), ptr_f(w.in_lAs),
                ptr_i8(w.in_lB), ptr_f(w.in_lBs),
                ptr_i8(w.out_lA), ptr_f(w.out_lAs),
                ptr_i8(w.out_lB), ptr_f(w.out_lBs),
                ptr_f(w.conv_w), ptr_f(w.conv_b),
                ptr_f(w.A_log), ptr_f(w.D), ptr_f(w.dt_bias),
                ptr_f(w.norm_w), ptr_f(w.ln_w),
                ptr_f(self.h_states[l]), ptr_f(self.conv_states[l]),
                ptr_f(self.x_norm), ptr_f(self.proj),
                ptr_f(self.y_buf), ptr_f(self.out_buf),
            )

        lib.final_rms_norm(ptr_f(self.x), ptr_f(self.norm_f_w),
                           ptr_f(self.norm_out), D_MODEL)

        logits = self.norm_out @ self.embedding.T
        return logits

    def generate(self, n_tokens=128):
        self.reset()
        token = 0
        tokens = [token]

        _ = self.forward(token)
        self.reset()

        start = time.perf_counter()
        for _ in range(n_tokens):
            logits = self.forward(token)
            token = int(np.argmax(logits))
            tokens.append(token)
        elapsed = time.perf_counter() - start

        tok_s = n_tokens / elapsed
        print(f"{n_tokens} tokens in {elapsed*1000:.1f} ms "
              f"({tok_s:.0f} tok/s, {elapsed/n_tokens*1e6:.0f} us/tok)")
        return tokens


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('--n_tokens', type=int, default=128)
    a = p.parse_args()

    model = CPUModel(a.model)
    model.generate(a.n_tokens)
