import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0):
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    angles = torch.outer(t, freqs).float()
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


# def get_pos_embed_indices(start, length, max_pos, scale=1.0):
#     scale = scale * torch.ones_like(start, dtype=torch.float32)
#     pos = (
#         start.unsqueeze(1)
#         + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
#     )
#     pos = torch.where(pos < max_pos, pos, max_pos - 1)
#     return pos


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

    
class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, conv_mult=2, kernel_size=7, dilation=1):
        super().__init__()
        # 1. Tự động tính toán intermediate_dim dựa trên hệ số nhân
        self.intermediate_dim = dim * conv_mult
        # 2. Depthwise Conv
        padding = ((kernel_size - 1) * dilation) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=dim, 
            dilation=dilation
        )
        # 3. LayerNorm & Pointwise mappings
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Linear 1: Phình to ra (D -> D * conv_mult)
        self.pwconv1 = nn.Linear(dim, self.intermediate_dim)
        self.act = nn.GELU()
        # 4. GRN (Global Response Normalization) 
        self.grn = GRN(self.intermediate_dim)
        # Linear 2: Thu nhỏ lại (D * conv_mult -> D)
        self.pwconv2 = nn.Linear(self.intermediate_dim, dim)

    def forward(self, x):
        residual = x
        
        # Step 1: Depthwise Conv (Bắt ngữ cảnh thời gian)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        
        # Step 2: MLP-style (Bắt đặc trưng kênh)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        return residual + x


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # --- 1. Feed Forward 1 (Macaron style) ---
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        # --- 2. Self-Attention Module ---
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # --- 3. Convolution Module (Tách rời các lớp) ---
        self.conv_norm = nn.LayerNorm(dim)
        self.conv_pw1 = nn.Conv1d(dim, dim * 2, kernel_size=1) # Pointwise
        self.conv_act_glu = nn.GLU(dim=1)
        self.conv_dw = nn.Conv1d(
            dim, dim, 
            kernel_size=conv_kernel_size, 
            padding=(conv_kernel_size - 1) // 2, 
            groups=dim
        ) # Depthwise
        self.conv_bn = nn.BatchNorm1d(dim)
        self.conv_act_silu = nn.SiLU()
        self.conv_pw2 = nn.Conv1d(dim, dim, kernel_size=1) # Pointwise
        self.conv_dropout = nn.Dropout(dropout) # Lớp cuối cùng của module conv

        # --- 4. Feed Forward 2 ---
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x shape: (B, T, D)
        
        # Step 1: Feed Forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Step 2: Self-Attention
        residual = x
        x = self.norm_attn(x)
        # attn mask trong PyTorch dùng key_padding_mask (B, T)
        x_attn, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + x_attn
        
        # Step 3: Convolution (Xử lý transpose tường minh)
        residual = x
        x = self.conv_norm(x)
        
        x = x.transpose(1, 2) # (B, D, T)
        x = self.conv_pw1(x)
        x = self.conv_act_glu(x)
        x = self.conv_dw(x)
        x = self.conv_bn(x)
        x = self.conv_act_silu(x)
        x = self.conv_pw2(x)
        x = x.transpose(1, 2) # Trả về (B, T, D)
        
        x = self.conv_dropout(x) # Dropout là lớp cuối
        x = residual + x
        
        # Step 4: Feed Forward 2
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, text_dim, padding_idx=0, 
                 extra_type='conformer', mask_padding=True,
                 convnext_layers=4, convnext_kernel=7,
                 conformer_layers=1, conformer_kernel=31, conformer_heads=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, text_dim, padding_idx=padding_idx)   
        self.mask_padding = mask_padding
        self.extra_type = extra_type
        
        # Nhúng vị trí (Positional Embedding)
        self.max_pos = 4096
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.max_pos), persistent=False)
        
        # Khởi tạo Block dựa trên lựa chọn
        if extra_type == 'conformer':
            print(f"🛠️ TextEncoder: Sử dụng {conformer_layers} lớp Conformer")
            self.text_blocks = nn.ModuleList([
                ConformerBlock(dim=text_dim, num_heads=conformer_heads, conv_kernel_size=conformer_kernel) 
                for _ in range(conformer_layers)
            ])
        elif extra_type == 'convnext':
            print(f"🛠️ TextEncoder: Sử dụng {convnext_layers} lớp ConvNeXtV2")
            self.text_blocks = nn.ModuleList([
                ConvNeXtV2Block(dim=text_dim, kernel_size=convnext_kernel) 
                for _ in range(convnext_layers)
            ])
        else:
            self.text_blocks = nn.ModuleList([])

    def forward(self, text_ids, seq_len, drop_text=False):
        batch, text_len = text_ids.shape
        valid_pos_mask = None
        
        if torch.is_tensor(seq_len):
            seq_len = seq_len.to(device=text_ids.device, dtype=torch.long)
            max_seq_len = int(seq_len.max().item())
        else:
            max_seq_len = int(seq_len)
            
        text_ids = F.pad(text_ids, (0, max_seq_len - text_len), value=0)
        if torch.is_tensor(seq_len):
            seq_pos = torch.arange(max_seq_len, device=text_ids.device).unsqueeze(0)
            valid_pos_mask = seq_pos < seq_len.unsqueeze(1)
            text_ids = text_ids.masked_fill(~valid_pos_mask, 0)

        # Mask cho Attention (B, T)
        if self.mask_padding:
            text_mask = (text_ids == 0) if self.mask_padding else None

        if drop_text:  # cfg for text
            text_ids = torch.zeros_like(text_ids)

        x = self.embedding(text_ids)

        if valid_pos_mask is not None:
            x = x.masked_fill(~valid_pos_mask.unsqueeze(-1), 0.0)
        
        # Cộng Positional Embedding
        freqs = self.freqs_cis[:max_seq_len, :]
        if valid_pos_mask is not None:
            freqs = freqs.unsqueeze(0) * valid_pos_mask.unsqueeze(-1).to(freqs.dtype)
        x = x + freqs

        # Modeling layers
        if self.mask_padding:
            x = x.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, x.size(-1)), 0.0)
            for block in self.text_blocks:
                if self.extra_type == 'conformer':
                    x = block(x, mask=text_mask)
                else:
                    x = block(x)
                    x = x.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, x.size(-1)), 0.0)    
        else:
            x = self.text_blocks(x)
        return x