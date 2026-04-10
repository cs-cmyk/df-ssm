#!/usr/bin/env python3
"""
Export frozen DFW scaffold + trained LoRA to .dfssm binary format.

Usage:
    python export_dfssm.py --scaffold dfssm_dfw_step1501.pt \
                           --lora dfw_lora_all_r16_final.pt \
                           --output model.dfssm
"""
import struct
import torch
import torch.nn.functional as F
import sys, os, argparse

sys.path.insert(0, '.')
from df_ssm_dfw_lora import *

# Constants (must match kernel)
D_MODEL    = 2048
D_INNER    = 4096
D_XBC      = 4352
D_STATE    = 128
NHEADS     = 64
HEADDIM    = 32
NGROUPS    = 1
D_CONV     = 4
LORA_RANK  = 16
N_LAYERS   = 48
IN_PROJ_OUT = 2 * D_INNER + 2 * NGROUPS * D_STATE + NHEADS  # 8512
OUT_PROJ_OUT = D_MODEL
HEADER_SIZE = 64


def pack_bits_to_int32(binary_tensor):
    """Pack a (rows, cols) tensor of 0/1 values into (rows, cols//32) int32."""
    rows, cols = binary_tensor.shape
    assert cols % 32 == 0, f"cols={cols} must be divisible by 32"
    binary_tensor = binary_tensor.to(torch.int32)
    packed = torch.zeros(rows, cols // 32, dtype=torch.int32)
    for bit in range(32):
        packed |= binary_tensor[:, bit::32] << bit
    return packed


def quantize_to_int8(weight):
    """Quantize FP weight to int8 with per-row scale."""
    scale = weight.abs().max(dim=-1).values / 127.0
    scale = scale.clamp(min=1e-8)
    quantized = (weight / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.half()


def write_tensor(f, tensor, expected_shape, expected_dtype):
    """Write a tensor to file, verifying shape and dtype."""
    assert tensor.shape == expected_shape, \
        f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"
    data = tensor.to(expected_dtype).contiguous().cpu().numpy().tobytes()
    f.write(data)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description='Export DFW+LoRA to .dfssm')
    parser.add_argument('--scaffold', type=str, required=True,
                        help='DFW scaffold checkpoint (e.g. dfssm_dfw_step1501.pt)')
    parser.add_argument('--lora', type=str, required=True,
                        help='LoRA checkpoint (e.g. dfw_lora_all_r16_final.pt)')
    parser.add_argument('--output', type=str, default='model.dfssm')
    parser.add_argument('--model', type=str, default='state-spaces/mamba2-1.3b')
    parser.add_argument('--Kw', type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("EXPORTING DFW + LoRA → .dfssm")
    print("=" * 60)

    # Load teacher config
    teacher, cfg, vocab_size = load_teacher(args.model, device='cpu')
    del teacher
    VOCAB_SIZE = vocab_size  # use actual vocab size from model

    # Load student with scaffold
    print(f"  Loading scaffold: {args.scaffold}")
    student = DFWMamba2LM(vocab_size=vocab_size, Ks=23, Kw=args.Kw, **cfg)
    ckpt = torch.load(args.scaffold, map_location='cpu', weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Freeze and add LoRA (creates FrozenDFWWithLoRA modules)
    freeze_scaffold_add_lora(student, lora_layers='all', lora_rank=16, lora_targets='both')

    # Load trained LoRA weights
    print(f"  Loading LoRA: {args.lora}")
    lora_ckpt = torch.load(args.lora, map_location='cpu', weights_only=False)
    for name, param in student.named_parameters():
        if name in lora_ckpt['lora_state']:
            param.data = lora_ckpt['lora_state'][name]

    student.eval()

    # Write .dfssm file
    print(f"  Writing: {args.output}")
    total_bytes = 0

    with open(args.output, 'wb') as f:
        # Header (64 bytes)
        header = struct.pack(
            '8s9I20s',
            b'DFSSM001',           # magic
            N_LAYERS,              # n_layer
            D_MODEL,               # d_model
            D_INNER,               # d_inner
            D_STATE,               # d_state
            NHEADS,                # nheads
            HEADDIM,               # headdim
            NGROUPS,               # ngroups
            VOCAB_SIZE,            # vocab_size
            args.Kw,               # Kw
            b'\x00' * 20           # reserved
        )
        assert len(header) == HEADER_SIZE
        f.write(header)
        total_bytes += HEADER_SIZE

        # Embedding (float16)
        emb = student.embedding.weight.data.half()
        total_bytes += write_tensor(f, emb, (VOCAB_SIZE, D_MODEL), torch.float16)
        print(f"    Embedding: {emb.numel() * 2 / 1e6:.1f} MB")

        # Per-layer weights
        for layer_idx in range(N_LAYERS):
            layer = student.layers[layer_idx]
            mixer = layer.mixer
            layer_bytes = 0

            for proj_name, proj_out, proj_in in [
                ('in_proj', IN_PROJ_OUT, D_MODEL),
                ('out_proj', OUT_PROJ_OUT, D_INNER),
            ]:
                proj = getattr(mixer, proj_name)

                if isinstance(proj, FrozenDFWWithLoRA):
                    # Scaffold: extract sign bits and pack
                    w = proj.frozen_weight.data  # (out, in) float
                    signs = (w.sign() + 1).to(torch.uint8) // 2  # 0 or 1
                    packed = pack_bits_to_int32(signs)
                    packed_cols = proj_in // 32
                    layer_bytes += write_tensor(f, packed, (proj_out, packed_cols), torch.int32)

                    # Scale: per-row absolute mean of original weight
                    scale = w.abs().mean(dim=1).half()
                    layer_bytes += write_tensor(f, scale, (proj_out,), torch.float16)

                    # LoRA A: int8 + per-row scale
                    lora_A_int8, lora_A_scale = quantize_to_int8(
                        proj.lora_A.data * proj.scaling)
                    layer_bytes += write_tensor(f, lora_A_int8, (proj_out, LORA_RANK), torch.int8)
                    layer_bytes += write_tensor(f, lora_A_scale, (proj_out,), torch.float16)

                    # LoRA B: int8 + per-row scale
                    lora_B_int8, lora_B_scale = quantize_to_int8(proj.lora_B.data)
                    layer_bytes += write_tensor(f, lora_B_int8, (LORA_RANK, proj_in), torch.int8)
                    layer_bytes += write_tensor(f, lora_B_scale, (LORA_RANK,), torch.float16)

                else:
                    # Shouldn't happen with lora_layers='all', but handle gracefully
                    # Treat as frozen DFW without LoRA — pack scaffold, zero LoRA
                    raise RuntimeError(f"Layer {layer_idx} {proj_name} is not FrozenDFWWithLoRA")

            # Conv1d
            conv_w = mixer.conv1d.weight.data.squeeze(1).half()  # (D_XBC, D_CONV)
            layer_bytes += write_tensor(f, conv_w, (D_XBC, D_CONV), torch.float16)

            conv_b = mixer.conv1d.bias.data.half()
            layer_bytes += write_tensor(f, conv_b, (D_XBC,), torch.float16)

            # A_log, D, dt_bias (float32)
            layer_bytes += write_tensor(f, mixer.A_log.data, (NHEADS,), torch.float32)
            layer_bytes += write_tensor(f, mixer.D.data, (NHEADS,), torch.float32)
            layer_bytes += write_tensor(f, mixer.dt_bias.data, (NHEADS,), torch.float32)

            # Inner norm (norm inside mixer)
            layer_bytes += write_tensor(f, mixer.norm.weight.data, (D_INNER,), torch.float32)

            # Layer norm (pre-layer RMSNorm)
            layer_bytes += write_tensor(f, layer.norm.weight.data, (D_MODEL,), torch.float32)

            total_bytes += layer_bytes
            if layer_idx == 0:
                print(f"    Layer 0: {layer_bytes / 1e6:.2f} MB "
                      f"(×48 = {layer_bytes * 48 / 1e6:.1f} MB)")

        # Final norm
        total_bytes += write_tensor(f, student.norm_f.weight.data, (D_MODEL,), torch.float32)

    file_size = os.path.getsize(args.output)
    print(f"\n  Written: {args.output}")
    print(f"  File size:     {file_size / 1e6:.1f} MB")
    print(f"  Data written:  {total_bytes / 1e6:.1f} MB")
    print(f"  Teacher FP16:  2687.5 MB")
    print(f"  Reduction:     {2687.5 / (file_size / 1e6):.1f}×")


if __name__ == '__main__':
    main()
