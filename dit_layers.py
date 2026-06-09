import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm_x * self.scale

class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        ada_params = self.linear(self.silu(emb)) # (b, d) -> (b, 6d)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(ada_params, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        ada_params = self.linear(self.silu(emb))
        scale, shift = torch.chunk(ada_params, 2, dim=1)

        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [i for i, layer in enumerate(self.conv1d) if isinstance(layer, nn.Conv1d)]

    def forward(self, x, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x
        
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        # Tự sinh cache cos, sin dựa trên độ dài chuỗi
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # [T, 1, D] để broadcast mượt mà với shape [B, T, H, D]
        return emb.cos()[:, None, :], emb.sin()[:, None, :]

    def rotate_queries_and_keys(self, q, k, seq_dim=1):
        # 1. Lấy thông tin thiết bị và độ dài chuỗi từ q
        device = q.device
        seq_len = q.shape[seq_dim]
        
        # 2. Tính cos/sin một lần duy nhất cho cả lượt
        cos, sin = self.forward(seq_len, device)
        
        # 3. Xoay cả hai bằng công thức: x * cos + rotate_half(x) * sin
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        return q, k

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        hidden_dim: int, 
        head_dim:int,
        num_heads: int, 
        qk_norm: bool = False, 
        pe_attn_head: Optional[int] = None, # Số lượng head được xoay RoPE
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.qk_norm = qk_norm
        
        # pn: Nếu không truyền, mặc định xoay tất cả heads
        self.pe_attn_head = pe_attn_head if pe_attn_head is not None else num_heads
        
        self.qkv = nn.Linear(hidden_dim, self.inner_dim * 3, bias=False)
        
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
            
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, hidden_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, rope=None):
        # x shape: [B, T, C]
        batch_size, seq_len, _ = x.shape
        
        # --- BƯỚC 1: QKV Projection & Rearrange ---
        qkv = self.qkv(x).chunk(3, dim=-1)
        # Shape: [B, T, H, D]
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> b t h d", h=self.num_heads), qkv)

        # --- BƯỚC 2: QK Norm (Tùy chọn) ---
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # --- BƯỚC 3: RoPE (Cơ chế Partial Heads - pe_attn_head) ---
        if rope is not None:
            pn = self.pe_attn_head
            if pn < self.num_heads:
                # Tách heads: [B, T, 0:pn, D] và [B, T, pn:H, D]
                q_rope, q_pass = q[:, :, :pn, :], q[:, :, pn:, :]
                k_rope, k_pass = k[:, :, :pn, :], k[:, :, pn:, :]
                
                # Chỉ xoay nhóm pn heads đầu tiên
                q_rope, k_rope = rope.rotate_queries_and_keys(q_rope, k_rope)
                
                # Nối lại theo chiều Head (dim=2 trong shape b t h d)
                q = torch.cat((q_rope, q_pass), dim=2)
                k = torch.cat((k_rope, k_pass), dim=2)
            else:
                # Xoay toàn bộ heads
                q, k = rope.rotate_queries_and_keys(q, k)

        # --- BƯỚC 4: Masking & Attention (SDPA) ---
        # Chuyển sang [B, H, T, D] để SDPA hiểu đúng
        q, k, v = map(lambda t: t.transpose(1, 2).contiguous(), (q, k, v))
        
        attn_mask = mask.unsqueeze(1).unsqueeze(1) if mask is not None else None
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0, # Luôn bằng 0 cho Attention Core
            is_causal=False
        )

        # --- BƯỚC 5: Kết thúc & Clean Padding ---
        out = rearrange(out.contiguous(), "b h t d -> b t (h d)")
        out = self.to_out(out) # Linear + Dropout

        if mask is not None:
            # Ép các vị trí Padding về 0 để tránh rò rỉ thông tin
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)

        return out

class DiTBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        head_dim, 
        ff_mult=4, 
        dropout=0.0, 
        qk_norm=None, 
        pe_attn_head=None
    ):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            hidden_dim=dim,
            head_dim=head_dim,
            num_heads=heads,
            qk_norm=False,
            pe_attn_head=pe_attn_head,
            dropout=dropout
        )
        
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_dim=dim, multiplier=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x
    

class FeedForward(nn.Module):
    """Transformer feed-forward network with GeLU."""

    def __init__(self, hidden_dim: int, multiplier: int = 4, dropout: float = 0.0, approximate='none'):
        super().__init__()
        inner = hidden_dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(approximate=approximate),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)