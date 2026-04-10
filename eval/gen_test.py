#!/usr/bin/env python3
"""Generation test for DFW scaffold + LoRA with density field state."""
import torch, torch.nn.functional as F, sys
sys.path.insert(0, '.')
from df_ssm_dfw_lora import *
from transformers import AutoTokenizer

def main():
    teacher, cfg, vocab_size = load_teacher('state-spaces/mamba2-1.3b', device='cpu')
    del teacher

    student = DFWMamba2LM(vocab_size=vocab_size, Ks=23, Kw=4, **cfg).cuda()
    ckpt = torch.load('dfssm_dfw_step1501.pt', map_location='cuda', weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'], strict=False)
    freeze_scaffold_add_lora(student, lora_layers='all', lora_rank=16, lora_targets='both')
    student = student.to('cuda')

    lora_ckpt = torch.load('dfw_lora_all_r16_final.pt', map_location='cuda', weights_only=False)
    for name, param in student.named_parameters():
        if name in lora_ckpt['lora_state']:
            param.data = lora_ckpt['lora_state'][name].to('cuda')

    student.eval()
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    df = DensityFieldConfig(K=23, use_sigma_delta=True, sd_accumulator_bits=8, block_len=64)

    prompts = [
        'The capital of France is',
        'Once upon a time there was',
        'To install Python on Linux',
        'The theory of relativity states that',
        'In the year 2050, humanity',
    ]

    # Generate without DF first (baseline)
    print("=" * 60)
    print("WITHOUT density field (FP16 state)")
    print("=" * 60)
    for prompt in prompts:
        text = generate(student, tok, prompt, max_tokens=80, df_config=None)
        print(f'\n--- {prompt} ---')
        print(text[:300])

    # Generate with DF
    print("\n" + "=" * 60)
    print("WITH density field (binary state, K=23)")
    print("=" * 60)
    for prompt in prompts:
        text = generate(student, tok, prompt, max_tokens=80, df_config=df)
        print(f'\n--- {prompt} ---')
        print(text[:300])


def generate(model, tok, prompt, max_tokens=80, df_config=None,
             temperature=0.7, top_p=0.9, rep_penalty=1.2):
    """Generate text with proper padding for DF-SSD."""
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    block_len = df_config.block_len if df_config else 64

    with torch.no_grad():
        for i in range(max_tokens):
            L = ids.size(1)
            pad_len = (block_len - L % block_len) % block_len
            if pad_len > 0:
                padded = F.pad(ids, (pad_len, 0), value=tok.eos_token_id or 0)
            else:
                padded = ids

            out = model(padded, df_config=df_config)

            # Logits for the last real token
            logits = out[0, -1, :].float()

            logits = logits / temperature

            seen = ids[0].unique()
            logits[seen] = logits[seen] / rep_penalty

            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_p, dim=0)
            mask = (cumsum - sorted_p) > top_p
            sorted_p[mask] = 0.0
            if sorted_p.sum() > 0:
                sorted_p = sorted_p / sorted_p.sum()
            else:
                sorted_p[0] = 1.0

            next_token = sorted_i[torch.multinomial(sorted_p, 1)]
            ids = torch.cat([ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tok.eos_token_id:
                break

    return tok.decode(ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    main()
