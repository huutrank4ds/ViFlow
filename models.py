"""Core architecture: Frontend branches + DiT backbone + OT-CFM training logic."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        prompt_conv_channels: Sequence[int] = (64, 128, 256), 
        prompt_conv_kernel: int = 3,
        vocab_size: int = 135, # Vocab Hữu đã tối ưu
    ) -> None:
        super().__init__()
        
        # 1. Embeddings & Projectors
        self.text_embedding = TextEmbedding(vocab_size, hidden_dim)
        self.prompt_encoder = PromptAudioEncoder(n_mels, hidden_dim, prompt_conv_channels, prompt_conv_kernel) # Conv-based
        self.target_projector = TargetProjector(n_mels, hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)

        # 2. Backbone DiT
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, ffn_multiplier, dropout)
            for _ in range(num_dit_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, n_mels)

    def _init_weights(self, m):
        """
        Quy tắc khởi tạo chuẩn cho các loại layer
        """
        if isinstance(m, nn.Linear):
            # Khởi tạo Linear theo phân phối chuẩn (Xavier/Glorot)
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Embedding):
            # Embedding thường dùng phân phối chuẩn nhỏ
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            # Norm layer: weight=1, bias=0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # Kaiming cho các lớp Convolution (phù hợp với ReLU/SiLU)
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def create_mask_and_ids(self, lens_list: List[torch.Tensor], max_lens: List[int]):
        """
        Tạo Valid Mask và Dense Position IDs cho RoPE.
        - lens_list: Danh sách các Tensor [B] chứa độ dài thực của từng phần (text, prompt, target).
        - max_lens: Danh sách các số nguyên là độ dài lớn nhất của từng phần trong batch.
        """
        batch_size = lens_list[0].size(0)
        device = lens_list[0].device
        
        all_masks = []
        all_ids = []
        # Offset riêng cho từng sample trong batch (vì độ dài thực khác nhau)
        curr_batch_offsets = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        for l, m in zip(lens_list, max_lens):
            # 1. Tạo Valid Mask (True = nội dung thực) -> Shape: [B, m]
            valid_mask = torch.arange(m, device=device)[None, :] < l[:, None]
            
            # 2. Tạo IDs nội bộ: dùng cumsum để padding không tăng ID
            # Ví dụ: [T, T, P, P] -> mask [1, 1, 0, 0] -> cumsum [1, 2, 2, 2]
            inner_ids = torch.cumsum(valid_mask.long(), dim=1)
            
            # 3. Cộng dồn với offset của các segment trước đó
            segment_ids = inner_ids + curr_batch_offsets[:, None]
            
            all_ids.append(segment_ids)
            all_masks.append(valid_mask)
            
            # 4. Cập nhật offset dựa trên độ dài THỰC (l) để segment sau nối tiếp segment trước
            curr_batch_offsets += l

        # Ghép tất cả các phần lại theo chiều ngang (dim=1)
        concat_mask = torch.cat(all_masks, dim=1) # [B, Total_T]
        concat_ids = torch.cat(all_ids, dim=1)   # [B, Total_T]
        
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
            x = block(x, t_emb=t_emb, attn_mask=~concat_mask[:, None, None, :], position_ids=concat_ids)

        x = self.final_norm(x)

        # 6. Velocity Prediction (Chỉ lấy phần hidden state của Target)
        # Chúng ta chỉ quan tâm đến vector field v_t tại vị trí của Target Mel
        target_start = x_text.size(1) + x_prompt.size(1)
        v_hat = self.velocity_head(x[:, target_start:, :]) # [B, L_target, n_mels]

        return v_hat