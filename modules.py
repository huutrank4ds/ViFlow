"""Model modules: tokenization, embeddings, positional encoding, and DiT blocks."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def rotate_half(x):
    """
    Hàm phụ trợ: Xoay nửa sau của ma trận lên trước và đổi dấu.
    Toán học: Đổi [x1, x2, x3, x4] thành [-x3, -x4, x1, x2]
    Đây là mấu chốt của ma trận xoay 2D.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TextEmbedding(nn.Module):
    """Phoneme embedding branch: Embedding -> Linear."""

    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)  # shape: [B, T_text, C]
        x = self.proj(x)  # shape: [B, T_text, C]
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        # GroupNorm(1, ...) hoạt động giống LayerNorm nhưng tối ưu cho tín hiệu 1D
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.GELU()
        
        # Shortcut nếu số kênh thay đổi
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, T]
        return self.act(self.norm(self.conv(x)) + self.res(x))

class PromptAudioEncoder(nn.Module):
    """
    Expert Architecture: Mel -> Residual Conv Stack -> Linear Projection.
    Tối ưu hóa khả năng trích xuất đặc trưng giọng nói từ Audio Prompt.
    """
    def __init__(
        self,
        n_mels: int = 100,           # Khớp với BigVGAN v2
        hidden_dim: int = 768,
        conv_channels: Sequence[int] = (256, 512, 512),
        kernel_size: int = 3,         # Kernel 3 hoặc 5 là đủ cho cục bộ
    ) -> None:
        super().__init__()
        
        # 1. Initial Projection: Đưa n_mels về channel đầu tiên
        self.input_proj = nn.Conv1d(n_mels, conv_channels[0], kernel_size=1)
        
        # 2. Convolutional Stack với Residual Connections
        blocks = []
        in_ch = conv_channels[0]
        for out_ch in conv_channels:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*blocks)
        
        # 3. Final Linear Projection để khớp với hidden_dim của DiT
        self.out_proj = nn.Linear(in_ch, hidden_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, T_prompt, n_mels] - Mel đã được pad.
        Returns:
            x: [B, T_prompt, hidden_dim]
        """
        # Chuyển trục để dùng Conv1d: [B, n_mels, T]
        x = mel.transpose(1, 2)
        
        x = self.input_proj(x)
        x = self.conv_stack(x)
        
        # Quay lại trục chuẩn cho Transformer: [B, T, C]
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        
        return x

class TargetProjector(nn.Module):
    """
    Target branch: Evolving state (x_t) -> Feature embedding.
    Đóng vai trò 'cửa ngõ' đưa đặc trưng Mel/Nhiễu vào không gian latent của DiT.
    """
    def __init__(self, n_mels: int, hidden_dim: int) -> None:
        super().__init__()
        # Thay vì chỉ Linear, ta dùng một block nhỏ để học đặc trưng tốt hơn
        self.proj = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) # Giúp ổn định giá trị khi x_t thay đổi từ nhiễu sang Mel
        )

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: [B, T_target, n_mels] (Có thể là x_t trong quá trình Flow)
        Returns:
            x: [B, T_target, hidden_dim]
        """
        return self.proj(target)
    

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        
        # 1. Tính toán tần số cơ sở (inv_freq)
        # Công thức: 1 / (base ^ (2i / d))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # Dùng register_buffer để không tính đạo hàm (không update weights) cho mảng này
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor: #type: ignore
        """
        x shape: [Batch, Seq_Len, Num_Heads, Head_Dim]
        position_ids shape: [Batch, Seq_Len]
        """
        seq_len = x.shape[1]
        
        # 2. Xử lý Tọa độ (Smart IDs)
        if position_ids is None:
            # Nếu người dùng không truyền Smart IDs, dùng đếm mù quáng (Naive)
            position_ids = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            position_ids = position_ids.unsqueeze(0) # shape: [1, seq_len]

        # 3. Tính góc xoay Theta cho từng vị trí
        # Nhân ma trận: Tọa độ * Tần số
        # freqs shape: [Batch, Seq_Len, Head_Dim / 2]
        freqs = torch.einsum("b t, d -> b t d", position_ids.float(), self.inv_freq)
        
        # Nhân đôi để có kích thước bằng Head_Dim
        emb = torch.cat((freqs, freqs), dim=-1) # shape: [Batch, Seq_Len, Head_Dim]
        
        # Thêm chiều Num_Heads vào giữa để khớp với shape của x
        emb = emb.unsqueeze(2) # shape: [Batch, Seq_Len, 1, Head_Dim]

        # 4. Tính Sin và Cos
        cos = emb.cos()
        sin = emb.sin()

        # 5. Áp dụng Phép xoay Vector (Rotary Operation)
        x_rotated = (x * cos) + (rotate_half(x) * sin)
        
        return x_rotated


class TimestepEmbedding(nn.Module):
    """Flow-matching timestep embedding for t in [0, 1]."""

    def __init__(self, hidden_dim: int, fourier_dim: int = 256) -> None:
        super().__init__()
        self.fourier_dim = fourier_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _fourier_features(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.fourier_dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                device=t.device,
                dtype=t.dtype,
            )
        )
        args = t * freqs.unsqueeze(0)
        feat = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return feat

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        feat = self._fourier_features(t)  # shape: [B, fourier_dim]
        emb = self.mlp(feat)  # shape: [B, hidden_dim]
        return emb


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm conditioned on timestep embedding."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(t_emb)  # shape: [B, 2C]
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # shape: [B, 1, C]
        shift = shift.unsqueeze(1)  # shape: [B, 1, C]
        y = self.norm(x) * (1 + scale) + shift  # shape: [B, T, C]
        return y


class MultiHeadSelfAttention(nn.Module):
    """Self-attention using PyTorch SDPA for FlashAttention integration."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope = RotaryEmbedding(head_dim=self.head_dim)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)  # shape: [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads)  # shape: [B, T, H, D]
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_heads)  # shape: [B, T, H, D]
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_heads)  # shape: [B, T, H, D]

        # Apply rotary positional embeddings
        q = self.rope(q, position_ids=position_ids)  # shape: [B, T, H, D]
        k = self.rope(k, position_ids=position_ids)  # shape: [B, T, H, D]

        q = q.transpose(1, 2)  # shape: [B, H, T, D]
        k = k.transpose(1, 2)  # shape: [B, H, T, D]
        v = v.transpose(1, 2)  # shape: [B, H, T, D]

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # shape: [B, H, T, D]

        attn = rearrange(attn, "b h t d -> b t (h d)")  # shape: [B, T, C]
        out = self.out_proj(attn)  # shape: [B, T, C]
        return out


class FeedForward(nn.Module):
    """Transformer feed-forward network with GeLU."""

    def __init__(self, hidden_dim: int, multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = hidden_dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """DiT block with AdaLN + SDPA attention + FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adaln_attn = AdaLayerNorm(hidden_dim)
        self.adaln_ffn = AdaLayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, multiplier=ffn_multiplier, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.adaln_attn(x, t_emb)  # shape: [B, T, C]
        h = self.attn(h, position_ids=position_ids, attn_mask=attn_mask)  # shape: [B, T, C]
        x = x + h  # shape: [B, T, C]

        h = self.adaln_ffn(x, t_emb)  # shape: [B, T, C]
        h = self.ffn(h)  # shape: [B, T, C]
        x = x + h  # shape: [B, T, C]
        return x
