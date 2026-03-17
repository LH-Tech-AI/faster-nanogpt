"""
Optimized GPT Language Model with modern architecture improvements.

Key improvements over vanilla nanoGPT:
- RoPE (Rotary Positional Embeddings) instead of learned position embeddings
- RMSNorm instead of LayerNorm (faster, no bias needed)
- QK-Norm for attention stability (prevents loss spikes)
- ReLU² activation (better training efficiency for small models)
- Logit soft-capping (prevents logit explosion, stabilizes training)
- Muon optimizer for 2D hidden weights (faster convergence via Newton-Schulz)
- AdamW for embeddings and scalar params
- No bias in Linear/Norm layers by default

References:
- modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt
- Muon optimizer: https://kellerjordan.github.io/posts/muon/
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Logit soft-capping: Gemma 2 (Google DeepMind)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization. Faster than LayerNorm, no bias."""
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (for GPT-2 compat)."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def build_norm(ndim, config):
    """Create the appropriate normalization layer."""
    if config.norm_type == 'rmsnorm':
        return RMSNorm(ndim)
    return LayerNorm(ndim, bias=config.bias)


# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def precompute_rope_cache(seq_len, head_dim, base=10000.0, device=None):
    """Precompute cos/sin cache for RoPE."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, theta)  # (seq_len, head_dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE to tensor x of shape (B, n_head, T, head_dim)."""
    T = x.size(2)
    half = x.size(3) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x1 * cos_t - x2 * sin_t,
        x2 * cos_t + x1 * sin_t,
    ], dim=-1)


# ============================================================================
# Attention
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # QK-Norm for training stability
        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Flash attention (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        if self.use_rope and rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # Attention
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ============================================================================
# MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.activation

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == 'relu2':
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ============================================================================
# Transformer Block
# ============================================================================

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = build_norm(config.n_embd, config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config.n_embd, config)
        self.mlp = MLP(config)

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.ln_1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================================
# Config
# ============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False       # No bias for modern arch (slightly better and faster)
    # Modern architecture options
    use_rope: bool = True    # RoPE instead of learned position embeddings
    rope_base: float = 10000.0
    activation: str = 'relu2'   # 'relu2' or 'gelu'
    norm_type: str = 'rmsnorm'  # 'rmsnorm' or 'layernorm'
    qk_norm: bool = True        # QK-Norm for attention stability
    logit_soft_cap: float = 30.0  # soft-capping for logits, 0.0 to disable


# ============================================================================
# GPT Model
# ============================================================================

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Build transformer
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=build_norm(config.n_embd, config),
        )
        # Learned position embeddings only when not using RoPE
        if not config.use_rope:
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE cache
        if config.use_rope:
            head_dim = config.n_embd // config.n_head
            cos, sin = precompute_rope_cache(config.block_size, head_dim, config.rope_base)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

        # Init weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), position embeddings get subtracted.
        Token embeddings stay due to weight tying with lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
            rope_cos, rope_sin = self.rope_cos, self.rope_sin
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            rope_cos, rope_sin = None, None

        for block in self.transformer.h:
            x = block(x, rope_cos, rope_sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.config.logit_soft_cap > 0:
                cap = self.config.logit_soft_cap
                logits = cap * torch.tanh(logits / cap)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only forward lm_head on last position
            logits = self.lm_head(x[:, [-1], :])
            if self.config.logit_soft_cap > 0:
                cap = self.config.logit_soft_cap
                logits = cap * torch.tanh(logits / cap)
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Model surgery to decrease block size if necessary."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Crop learned position embeddings if present
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        # Recompute RoPE cache
        if self.config.use_rope:
            head_dim = self.config.n_embd // self.config.n_head
            cos, sin = precompute_rope_cache(
                block_size, head_dim, self.config.rope_base, device=self.rope_cos.device
            )
            self.rope_cos = cos
            self.rope_sin = sin
        # Crop causal mask buffer
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained GPT-2 weights (uses legacy architecture for compatibility)."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True, legacy architecture")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        # Legacy architecture for GPT-2 weight compatibility
        config_args['use_rope'] = False
        config_args['activation'] = 'gelu'
        config_args['norm_type'] = 'layernorm'
        config_args['qk_norm'] = False
        config_args['logit_soft_cap'] = 0.0
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                              use_muon=True, muon_lr=0.02, muon_momentum=0.95):
        """
        Configure optimizer: Muon for hidden 2D weights, AdamW for the rest.
        Falls back to pure AdamW if use_muon=False.
        """
        muon_params = []
        embed_params = []
        decay_params = []
        nodecay_params = []

        seen_ids = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            if 'wte' in name or 'wpe' in name or 'lm_head' in name:
                embed_params.append(param)
            elif param.ndim >= 2 and use_muon:
                muon_params.append(param)
            elif param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        if use_muon and muon_params:
            num_muon = sum(p.numel() for p in muon_params)
            num_adam = sum(p.numel() for p in embed_params + decay_params + nodecay_params)
            print(f"Muon: {len(muon_params)} tensors, {num_muon:,} parameters")
            print(f"AdamW: {len(embed_params) + len(decay_params) + len(nodecay_params)} tensors, "
                  f"{num_adam:,} parameters")

            adam_groups = [
                {'params': embed_params, 'lr': learning_rate, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'lr': learning_rate, 'weight_decay': 0.0},
            ]
            if decay_params:
                adam_groups.append(
                    {'params': decay_params, 'lr': learning_rate, 'weight_decay': weight_decay}
                )

            optimizer = MuonAdamW(
                muon_params=muon_params,
                adam_params=adam_groups,
                muon_lr=muon_lr,
                muon_momentum=muon_momentum,
                muon_weight_decay=weight_decay,
                adam_betas=betas,
            )
        else:
            all_decay = embed_params + decay_params + muon_params
            optim_groups = [
                {'params': all_decay, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0},
            ]
            num_decay = sum(p.numel() for p in all_decay)
            num_nodecay = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(all_decay)}, with {num_decay:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
                  f"with {num_nodecay:,} parameters")

            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")

        # Store base_lr for LR scheduling
        for g in optimizer.param_groups:
            if 'base_lr' not in g:
                g['base_lr'] = g['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Muon + AdamW Combined Optimizer
# ============================================================================

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D hidden-layer weights, AdamW for the rest.

    Muon (MomentUm Orthogonalized by Newton-schulz) uses Newton-Schulz iteration
    to orthogonalize momentum updates for faster convergence on matrix parameters.
    See: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(self, muon_params, adam_params,
                 muon_lr=0.02, muon_momentum=0.95, muon_weight_decay=0.01,
                 muon_nesterov=True, muon_ns_steps=5,
                 adam_betas=(0.9, 0.95), adam_eps=1e-8):

        all_groups = []

        # Muon param group
        muon_group = dict(
            params=list(muon_params),
            lr=muon_lr,
            base_lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            is_muon=True,
        )
        all_groups.append(muon_group)

        # AdamW param groups
        for g in adam_params:
            g['is_muon'] = False
            g['base_lr'] = g.get('lr', 6e-4)
            g['betas'] = adam_betas
            g['eps'] = adam_eps
            all_groups.append(g)

        defaults = dict(lr=muon_lr, weight_decay=0.0, is_muon=False)
        super().__init__(all_groups, defaults)

    @staticmethod
    @torch.no_grad()
    def _newton_schulz(G, steps=5):
        """
        Approximate orthogonalization via Newton-Schulz iteration.
        Uses tuned polynomial coefficients for fast convergence in bfloat16.
        """
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        transposed = False
        if X.size(-2) > X.size(-1):
            X = X.mT
            transposed = True
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.mT
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('is_muon', False):
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss

    def _muon_step(self, group):
        lr = group['lr']
        wd = group['weight_decay']
        beta = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(grad)

            buf = state['momentum_buffer']
            # Exponential moving average of gradients
            buf.lerp_(grad, 1 - beta)

            # Nesterov look-ahead: combine current gradient with momentum
            if nesterov:
                update = grad.lerp(buf, beta)  # (1-beta)*grad + beta*buf
            else:
                update = buf

            # Newton-Schulz orthogonalization
            update = self._newton_schulz(update, steps=ns_steps)
            # Dimensional scaling factor: sqrt(fan_out / fan_in)
            scale = (p.size(0) / p.size(1)) ** 0.5

            # Decoupled weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)
            # Parameter update
            p.add_(update.to(p.dtype), alpha=-lr * scale)

    def _adam_step(self, group):
        lr = group['lr']
        wd = group.get('weight_decay', 0.0)
        beta1, beta2 = group['betas']
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.float()
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)

            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            # Update biased first and second moment estimates
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)

            # Bias correction
            step = state['step']
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            # Decoupled weight decay
            if wd > 0:
                p.mul_(1 - lr * wd)

            # Compute and apply update
            step_size = lr / bc1
            denom = (exp_avg_sq / bc2).sqrt().add_(eps)
            update = exp_avg / denom
            p.add_(update.to(p.dtype), alpha=-step_size)
