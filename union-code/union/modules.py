import math
from inspect import isfunction

import torch
from torch import nn, einsum
from einops import rearrange

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def is_not_none(x):
    return x is not None

def fallback(val, d):
    return val if is_not_none(val) else (d() if isfunction(d) else d)

def cycle_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def split_into_batches(total, size):
    """
    Splits 'total' into groups of 'size', and if there's a remainder,
    creates one more group for the leftover.
    """
    n_full = total // size
    remainder = total % size
    result = [size] * n_full
    if remainder > 0:
        result.append(remainder)
    return result

# -------------------------------------------------------------------------
# Exponential moving average
# -------------------------------------------------------------------------

class ExpMovingAvg():
    """
    Maintains an exponential moving average of model parameters.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_params(self, target_model, source_model):
        for p_target, p_source in zip(target_model.parameters(), source_model.parameters()):
            old_w, new_w = p_target.data, p_source.data
            p_target.data = self._ema_update(old_w, new_w)

    def _ema_update(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# -------------------------------------------------------------------------
# Residual wrapper
# -------------------------------------------------------------------------

class ResidualAdd(nn.Module):
    """
    Wraps a module and adds its output back to the original input (residual).
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)

# -------------------------------------------------------------------------
# Sinusoidal positional embedding
# -------------------------------------------------------------------------

class SineCosinePosEmb(nn.Module):
    """
    Computes sinusoidal embeddings (sine and cosine) for each time step
    to be used in diffusion models or similar tasks.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        device = t.device
        half_dim = self.emb_dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -exponent)
        x = t[:, None] * freqs[None, :]
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x

# -------------------------------------------------------------------------
# Simple upsample/downsample
# -------------------------------------------------------------------------

def Up2x(channels):
    """
    2x upsampling using a transposed convolution.
    """
    return nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

def Down2x(channels):
    """
    2x downsampling using a stride-2 convolution.
    """
    return nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

# -------------------------------------------------------------------------
# Layer normalization
# -------------------------------------------------------------------------

class SpatialLayerNorm(nn.Module):
    """
    Normalizes across the channel dimension. Similar to GroupNorm(1, dim) or InstanceNorm,
    but implemented manually to keep exact control over the details.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gain + self.bias

class PreNormalization(nn.Module):
    """
    Applies a layer normalization before a given function 'fn'.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = SpatialLayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

# -------------------------------------------------------------------------
# ConvNeXt-based CS block
# -------------------------------------------------------------------------

class ConvNeXtStage(nn.Module):
    def __init__(self, in_dim, out_dim, *, emb_dim=None, expand=3, norm=True):
        super().__init__()

        self.embed_proj = None
        if emb_dim is not None:
            self.embed_proj = nn.Sequential(
                nn.GELU(),
                nn.Linear(emb_dim, in_dim)
            )

        self.depthwise = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=7,
            padding=3,
            groups=in_dim
        )

        # apply layer norm conditionally, depending on 'norm'
        self.block = nn.Sequential(
            SpatialLayerNorm(in_dim) if norm else nn.Identity(),
            nn.Conv2d(in_dim, out_dim * expand, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim * expand, out_dim, 3, padding=1)
        )

        self.residual_proj = (
            nn.Conv2d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, emb=None):
        h = self.depthwise(x)

        if self.embed_proj is not None:
            if emb is None:
                raise ValueError("Embeddings must be provided if emb_dim was set.")
            cond = self.embed_proj(emb)
            h += cond.unsqueeze(-1).unsqueeze(-1)

        h = self.block(h)
        return h + self.residual_proj(x)
# -------------------------------------------------------------------------
# Attention modules
# -------------------------------------------------------------------------

class FastAttention(nn.Module):
    """
    Linear attention variant. Uses softmax along K dimension first,
    then projects Q * K to V with linear complexity in sequence length.
    """
    def __init__(self, channels, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_output = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (hd) x y -> b h d (x y)', h=self.heads) * self.scale
        k = rearrange(k, 'b (hd) x y -> b h d (x y)', h=self.heads).softmax(dim=-1)
        v = rearrange(v, 'b (hd) x y -> b h d (x y)', h=self.heads)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h d (x y) -> b (h d) x y', h=self.heads, x=h, y=w)
        return self.to_output(out)

class FullAttention(nn.Module):
    """
    Standard full attention: Q * K^T -> softmax -> multiply by V.
    """
    def __init__(self, channels, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(channels, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_output = nn.Conv2d(hidden_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = rearrange(q, 'b (hd) x y -> b h d (x y)', h=self.heads) * self.scale
        k = rearrange(k, 'b (hd) x y -> b h d (x y)', h=self.heads)
        v = rearrange(v, 'b (hd) x y -> b h d (x y)', h=self.heads)

        # Compute attention
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_output(out)
