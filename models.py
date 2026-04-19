"""Core architecture: Frontend branches + DiT backbone + OT-CFM training logic."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontend import MelSpectrogramFrontend, VietnameseTokenizer
from modules import (
    DiTBlock,
    PromptAudioEncoder,
    TargetProjector,
    TextEmbedding,
    TimestepEmbedding,
)


class ViFlowOTCFM(nn.Module):
    """
    Core Architecture: Chuyên biệt xử lý Flow Matching trên dữ liệu đã được chuẩn hóa.
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        num_dit_blocks: int = 12,
        num_heads: int = 12,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        n_mels: int = 100, # Khớp với BigVGAN v2 24kHz
        vocab_size: int = 135, # Vocab Hữu đã tối ưu
    ) -> None:
        super().__init__()
        
        # 1. Embeddings & Projectors
        self.text_embedding = TextEmbedding(vocab_size, hidden_dim)
        self.prompt_encoder = PromptAudioEncoder(n_mels, hidden_dim) # Conv-based
        self.target_projector = TargetProjector(n_mels, hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)

        # 2. Backbone DiT
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, ffn_multiplier, dropout)
            for _ in range(num_dit_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, n_mels)

    def create_mask_and_ids(self, lens_list: List[torch.Tensor], max_lens: List[int]):
        """
        Tạo Mask và Position IDs chuẩn cho RoPE.
        lens_list: [text_lens, prompt_lens, target_lens]
        max_lens: [max_T_text, max_T_prompt, max_T_target]
        """
        batch_size = lens_list[0].size(0)
        device = lens_list[0].device
        
        all_masks = []
        all_ids = []
        curr_offset = 0
        
        for l, m in zip(lens_list, max_lens):
            # 1. Tạo Mask như cũ
            mask = torch.arange(m, device=device)[None, :] < l[:, None]
            all_masks.append(mask)
            
            # 2. Tạo Position IDs với Offset cố định
            # Điều này đảm bảo: Token đầu tiên của Prompt LUÔN có ID là max_T_text
            # bất kể câu đó có bao nhiêu padding.
            ids = torch.arange(m, device=device)[None, :] + curr_offset
            # Expand ra toàn batch: [B, m]
            all_ids.append(ids.expand(batch_size, -1))
            
            curr_offset += m # Tăng offset cho phần tiếp theo
            
        concat_mask = torch.cat(all_masks, dim=1)
        concat_ids = torch.cat(all_ids, dim=1)
        
        return concat_mask, concat_ids

    def forward(
        self,
        text_ids: torch.Tensor,     # [B, L_text]
        text_lens: torch.Tensor,    # [B]
        prompt_mel: torch.Tensor,   # [B, L_prompt, n_mels]
        prompt_lens: torch.Tensor,  # [B]
        target_xt: torch.Tensor,    # [B, L_target, n_mels] (Ma trận x_t trong OT-CFM)
        target_lens: torch.Tensor,  # [B]
        t: torch.Tensor,            # [B] Thời điểm t từ 0 -> 1
    ) -> torch.Tensor:
        
        # 1. Chuyển đổi các nhánh về cùng hidden_dim
        # Nhánh Text
        x_text = self.text_embedding(text_ids) 
        
        # Nhánh Prompt (Conditioning)
        x_prompt = self.prompt_encoder(prompt_mel) 
        
        # Nhánh Target (Dữ liệu đang khớp - Flow)
        x_target = self.target_projector(target_xt)

        # 2. Ghép chuỗi (Concatenation)
        # Sequence format: [Text | Prompt | Target]
        x_all = torch.cat([x_text, x_prompt, x_target], dim=1)
        
        # 3. Xử lý Masking hợp nhất
        # Transformer trong DiT cần biết đâu là padding của cả 3 phần
        concat_mask, concat_ids = self.create_mask_and_ids(
            [text_lens, prompt_lens, target_lens],
            [x_text.size(1), x_prompt.size(1), x_target.size(1)]
        ) # Shape: [B, T_all]

        # 4. Timestep Embedding (Khuếch tán/Flow thời gian)
        t_emb = self.timestep_embedding(t) # [B, C]

        # 5. DiT Backbone processing
        # DiTBlock sẽ nhận x_all và dùng concat_mask để bỏ qua các vùng padding
        x = x_all
        for block in self.dit_blocks:
            x = block(x, t_emb=t_emb, mask=concat_mask, position_ids=concat_ids)

        x = self.final_norm(x)

        # 6. Velocity Prediction (Chỉ lấy phần hidden state của Target)
        # Chúng ta chỉ quan tâm đến vector field v_t tại vị trí của Target Mel
        target_start = x_text.size(1) + x_prompt.size(1)
        v_hat = self.velocity_head(x[:, target_start:, :]) # [B, L_target, n_mels]

        return v_hat