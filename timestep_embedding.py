import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, fourier_dim: int = 256):
        """
        Lớp bọc (Wrapper) để đưa timestep qua Sinusoidal và MLP.
        """
        super().__init__()
        # 1. Lớp mã hóa lượng giác "chính chủ" F5-TTS
        self.sinusoidal_emb = SinusoidalPosEmb(fourier_dim)
        
        # 2. Mạng MLP để map fourier_dim (256) lên hidden_dim (768)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: Tensor [B] hoặc [B, 1] nằm trong khoảng [0, 1]
        """
        # Nếu t là [B, 1], bóp về [B] để khớp với logic của SinusoidalPosEmb
        if t.dim() == 2:
            t = t.squeeze(-1)
            
        # Bước 1: Tạo chữ ký sin/cos (256 dims)
        # scale=1000 đã được định nghĩa mặc định trong SinusoidalPosEmb.forward
        temb = self.sinusoidal_emb(t)
        
        # Bước 2: Phóng đại qua MLP để ra vector điều kiện cuối cùng
        return self.mlp(temb)