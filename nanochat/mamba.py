"""
Mamba model implementation for nanochat
Based on the Mamba: Linear-Time Sequence Modeling with Selective State Spaces paper
Notable features:
- Selective State Space Model (S6) 
- Linear complexity in sequence length
- Hardware-efficient implementation
- Compatible with nanochat training pipeline
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    print0("Warning: mamba_ssm not available. Installing with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False

@dataclass
class MambaConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_embd: int = 768
    d_state: int = 16  # SSM state expansion factor
    d_conv: int = 4    # Local convolution width
    expand: int = 2    # Block expansion factor


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.n_embd
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        
        if MAMBA_AVAILABLE:
            # Use official mamba_ssm implementation
            self.mixer = Mamba(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
            )
        else:
            # Fallback: simple linear layer (for compatibility)
            print0(f"Warning: Using linear fallback for Mamba layer {layer_idx}")
            self.mixer = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # MLP similar to GPT
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * self.d_model, self.d_model, bias=False)
        )

    def forward(self, x):
        # Pre-norm architecture like GPT
        if MAMBA_AVAILABLE:
            x = x + self.mixer(norm(x))
        else:
            # Fallback linear transformation
            x = x + self.mixer(norm(x))
        x = x + self.mlp(norm(x))
        return x


class MambaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.silu(x)  # SwiGLU activation like modern models
        x = self.c_proj(x)
        return x


class MambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([MambaBlock(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Cast embeddings to bf16 for memory efficiency
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # Zero out output projection weights in all blocks
        for block in self.transformer.h:
            if hasattr(block.mlp, 'c_proj'):
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            elif hasattr(block.mlp, '2'):  # Sequential case
                torch.nn.init.zeros_(block.mlp[2].weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization scaled by fan-out
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model."""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        # Mamba has approximately linear complexity, unlike quadratic attention
        # Rough estimate: similar to transformer but without attention overhead
        l, d, t = self.config.n_layer, self.config.n_embd, self.config.sequence_len
        
        # SSM operations are roughly O(d * d_state * t) per layer
        ssm_flops = l * d * self.config.d_state * t * 6  # rough estimate
        mlp_flops = l * d * 4 * d * 2  # MLP operations
        num_flops_per_token = (ssm_flops + mlp_flops) / t + 6 * (nparams - nparams_embedding) / t
        
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        
        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        # AdamW for embeddings and lm_head
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Muon for matrix operations (SSM + MLP)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        # Combine optimizers
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        
        # Forward through the model
        x = self.transformer.wte(idx)
        x = norm(x)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = norm(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        # Logits softcap like GPT
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            # Training mode: compute loss
            logits = logits.float()  # Use fp32 for loss computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1, 
                reduction=loss_reduction
            )
            return loss
        else:
            # Inference mode: return logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Autoregressive generation for Mamba
        Mamba doesn't need KV cache but has internal SSM states
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            # Mamba processes the entire sequence each time
            # But it's still linear complexity unlike quadratic attention
            logits = self.forward(ids)
            logits = logits[:, -1, :]  # Last token logits
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token


# Alias for compatibility with existing code
Mamba = MambaModel