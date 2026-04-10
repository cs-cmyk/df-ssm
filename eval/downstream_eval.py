#!/usr/bin/env python3
"""
Downstream evaluation for DF-SSM via lm-evaluation-harness.

Usage:
    python downstream_eval.py --scaffold dfssm_dfw_step1501.pt \
                              --lora dfw_lora_all_r16_final.pt \
                              --tasks boolq,piqa,hellaswag,winogrande,arc_easy

Requires: pip install lm-eval
"""

import torch
import torch.nn.functional as F
import sys, argparse

sys.path.insert(0, '.')
from df_ssm_dfw_lora import *
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator


@register_model("dfssm")
class DFSSMEvalWrapper(LM):
    def __init__(self, scaffold_path, lora_path, model_name='state-spaces/mamba2-1.3b',
                 batch_size=1, device='cuda', **kwargs):
        super().__init__()
        
        # Load model
        teacher, cfg, vocab_size = load_teacher(model_name, device='cpu')
        del teacher
        
        self.model = DFWMamba2LM(vocab_size=vocab_size, Ks=23, Kw=4, **cfg).to(device)
        ckpt = torch.load(scaffold_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        freeze_scaffold_add_lora(self.model, lora_layers='all', lora_rank=16, lora_targets='both')
        self.model = self.model.to(device)
        
        lora_ckpt = torch.load(lora_path, map_location=device, weights_only=False)
        for name, param in self.model.named_parameters():
            if name in lora_ckpt['lora_state']:
                param.data = lora_ckpt['lora_state'][name].to(device)
        
        self.model.eval()
        self.device = device
        self._batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = vocab_size
        
        self.df = DensityFieldConfig(K=23, use_sigma_delta=True, 
                                      sd_accumulator_bits=8, block_len=64)
        print(f"DF-SSM loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params")
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return 2048
    
    @property
    def max_gen_toks(self):
        return 256
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens)
    
    def _model_call(self, inps):
        """Run model on input_ids, return logits."""
        with torch.no_grad():
            # Pad to multiple of 64 for DF-SSD
            B, L = inps.shape
            block_len = 64
            pad_len = (block_len - L % block_len) % block_len
            if pad_len > 0:
                inps = F.pad(inps, (pad_len, 0), value=self.tokenizer.eos_token_id or 0)
            
            logits = self.model(inps, df_config=self.df)
            
            # Remove padding from logits
            if pad_len > 0:
                logits = logits[:, pad_len:, :]
            
            return logits
    
    def _model_generate(self, context, max_length, stop, **kwargs):
        raise NotImplementedError("Generation not needed for downstream eval")
    
    def loglikelihood(self, requests, disable_tqdm=False):
        """Compute log-likelihood of continuations given contexts."""
        results = []
        
        for context, continuation in [req.args for req in requests]:
            ctx_enc = self.tok_encode(context)
            cont_enc = self.tok_encode(continuation)
            
            full_enc = ctx_enc + cont_enc
            
            # Truncate to max length
            if len(full_enc) > self.max_length:
                full_enc = full_enc[-self.max_length:]
                ctx_len = len(full_enc) - len(cont_enc)
            else:
                ctx_len = len(ctx_enc)
            
            input_ids = torch.tensor([full_enc], dtype=torch.long, device=self.device)
            
            logits = self._model_call(input_ids)
            
            # Get log probs for continuation tokens
            logits = logits[0, ctx_len - 1:-1, :]  # shift by 1 for next-token prediction
            targets = input_ids[0, ctx_len:]
            
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            total_ll = token_log_probs.sum().item()
            is_greedy = (logits.argmax(dim=-1) == targets).all().item()
            
            results.append((total_ll, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        """Compute rolling log-likelihood (for perplexity)."""
        results = []
        
        for (string,) in [req.args for req in requests]:
            enc = self.tok_encode(string)
            
            # Process in chunks
            total_ll = 0.0
            total_tokens = 0
            
            for start in range(0, len(enc), self.max_length):
                chunk = enc[start:start + self.max_length]
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                
                logits = self._model_call(input_ids)
                
                # Log probs for all tokens after the first
                logits = logits[0, :-1, :]
                targets = input_ids[0, 1:]
                
                log_probs = F.log_softmax(logits.float(), dim=-1)
                token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                
                total_ll += token_log_probs.sum().item()
                total_tokens += len(targets)
            
            results.append((total_ll, total_tokens))
        
        return results
    
    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Generation not implemented for eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaffold', type=str, required=True)
    parser.add_argument('--lora', type=str, required=True)
    parser.add_argument('--tasks', type=str, default='boolq,piqa,hellaswag,winogrande,arc_easy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='state-spaces/mamba2-1.3b')
    args = parser.parse_args()
    
    # Create model
    model = DFSSMEvalWrapper(
        scaffold_path=args.scaffold,
        lora_path=args.lora,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    
    # Run evaluation
    task_list = args.tasks.split(',')
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_list,
        batch_size=args.batch_size,
        device=str(model.device),
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("DOWNSTREAM EVALUATION RESULTS")
    print("=" * 70)
    
    for task_name, task_result in results['results'].items():
        print(f"\n{task_name}:")
        for metric, value in task_result.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Task':<20} {'Metric':<15} {'Score':<10}")
    print("-" * 45)
    for task_name, task_result in results['results'].items():
        # Find the main accuracy metric
        for metric in ['acc_norm', 'acc', 'accuracy']:
            if metric in task_result:
                val = task_result[metric]
                if isinstance(val, float):
                    print(f"{task_name:<20} {metric:<15} {val:.4f}")
                break


if __name__ == '__main__':
    main()
