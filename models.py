from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from dit_layers import (
    DiTBlock,
    ConvPositionEmbedding,
    RotaryEmbedding,
    AdaLayerNorm_Final
)

from text_embedding import TextEmbedding
from timestep_embedding import TimestepEmbedding

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False, mel_mask=None):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        
        combined = torch.cat((x, cond, text_embed), dim=-1) # [B, N, D*3]
        
        x = self.proj(combined) # [B, N, out_dim]
        x = self.conv_pos_embed(x, mask=mel_mask) + x
        return x


class ViFlowOTCFM(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=18,
        head_dim=64,
        heads=12,
        text_dim=None,
        text_mask_padding=True,
        mel_dim=100,
        vocab_size=71,
        ff_mult=4,
        text_embedding_type="convnext",
        text_conformer_layers=1,
        text_conformer_heads=4,
        text_convnext_layers=4,
        pe_attn_head=None,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        if text_dim is None:
            text_dim = mel_dim
        
        # 1. Mã hóa văn bản (Text Encoder nội bộ)
        # Text dim chốt cứng 100 để khớp với InputEmbedding của sếp
        self.text_encoder = TextEmbedding(
            vocab_size=vocab_size, 
            text_dim=text_dim,
            extra_type=text_embedding_type,
            conformer_layers=text_conformer_layers,
            conformer_heads=text_conformer_heads,
            convnext_layers=text_convnext_layers,
            mask_padding=text_mask_padding
        )
        
        # 2. Điều kiện thời gian (Flow Matching Timestep)
        self.time_embed = TimestepEmbedding(hidden_dim=dim)
        
        # 3. Lớp hòa trộn đầu vào (Combined: Noisy Mel + Ref Mel + Text)
        self.input_embed = InputEmbedding(
            mel_dim=mel_dim, 
            text_dim=text_dim, 
            out_dim=dim
        )
        
        # 4. Rotary Positional Embedding (RoPE) - Khởi tạo 1 lần dùng chung
        self.rope = RotaryEmbedding(head_dim)
        
        # 5. Danh sách các DiT Blocks (Trái tim của mô hình)
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, 
                head_dim=head_dim,
                heads=heads, 
                ff_mult=ff_mult,
                dropout=dropout,
                pe_attn_head=pe_attn_head
            ) for _ in range(depth)
        ])
        
        # 6. Lớp chuẩn hóa cuối cùng (Adaptive) và Projection
        self.final_norm = AdaLayerNorm_Final(dim)
        self.final_proj = nn.Linear(dim, mel_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            # attn_norm.linear là lớp sinh ra 6 tham số điều kiện
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # 2. Khởi tạo cho AdaLayerNorm_Final (norm_out)
        nn.init.constant_(self.final_norm.linear.weight, 0)
        nn.init.constant_(self.final_norm.linear.bias, 0)

        # 3. Khởi tạo cho Lớp Projection cuối cùng (proj_out)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)


    def forward(self, x, cond, text_ids, t, mel_lens=None, mask=None, drop_audio_cond=False, drop_text=False, drop_audio_mask=None, drop_text_mask=None):
        """
        x: Noisy Mel [B, T, 100]
        cond: Reference Mel [B, T, 100]
        text_ids: Token IDs [B, T]
        t: Timestep [B] (từ 0 đến 1)
        mask: Padding Mask [B, T]
        """
        if mel_lens is None:
            seq_len = x.shape[1]
        else:
            seq_len = mel_lens
        # Mã hóa văn bản
        text_embed = self.text_encoder(text_ids, seq_len, drop_text) # [b, t, d]
        # Tạo vector thời gian
        t_emb = self.time_embed(t) # [B, dim]

        if drop_text_mask is not None and drop_text_mask.any():
            text_embed[drop_text_mask] = 0.0
            
        if drop_audio_mask is not None and drop_audio_mask.any():
            cond[drop_audio_mask] = 0.0

        # Kết hợp (Noisy, Ref, Text) -> ConvPosEmbed -> Proj
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, mel_mask=mask) # [B, T, dim]
        
        # --- TRANSFORMER BLOCKS ---
        for block in self.transformer_blocks:
            # Truyền x, t_emb, mask và rope vào từng block
            x = block(x, t_emb, mask=mask, rope=self.rope)
            
        # Adaptive LayerNorm cuối cùng dựa trên t_emb
        x = self.final_norm(x, t_emb)
        
        # Dự đoán v_t (Trường vận tốc)
        v_t = self.final_proj(x) # [B, T, 100]
        
        # Tẩy sạch vùng padding ở đầu ra cuối cùng
        if mask is not None:
            v_t = v_t.masked_fill(~mask.unsqueeze(-1), 0.0)
            
        return v_t