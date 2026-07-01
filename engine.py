import torch
import torch.nn.functional as F
import numpy as np

class ViFlowEngine:
    def __init__(self, sigma_min: float = 0.0):
        # Có thể thay đổi sigma_min để điều chỉnh mức độ nhiễu tối thiểu trong quá trình tạo xt
        self.sigma_min = sigma_min

    # Tạo batch cho training với logic Infilling + Flow Matching
    def get_train_batch(self, x1, mel_lens, mel_mask, min_p=0.3, max_p=0.7):
        """
        x1: [B, N, 100] - Mel nguyên bản
        mel_lens: [B] - Độ dài thực tế
        mel_mask: [B, N] - Mask của padding (True ở vị trí dữ liệu thật, False tại padding)
        """
        batch_size, max_seq_len, dim = x1.size()
        device = x1.device
    
        # Tính toán độ dài đoạn cần che (30% - 70%)
        m_ratio = torch.rand(batch_size, device=device) * (max_p - min_p) + min_p
        m_len = (m_ratio * mel_lens).long()
        
        # Tính toán vị trí bắt đầu che (mask_start)
        # Phần còn lại: rem_len = mel_lens - m_len
        # Ta chia rem_len thành prefix_len và suffix_len sao cho prefix >= suffix
        rem_len = mel_lens - m_len
        # prefix_len dao động từ [rem_len / 2] đến [rem_len]
        prefix_len = (torch.rand(batch_size, device=device) * (rem_len - (rem_len // 2)) + (rem_len // 2)).long()
        
        mask_start = prefix_len
        mask_end = mask_start + m_len
    
        # Tạo target_mask cho đoạn ở giữa
        indices = torch.arange(max_seq_len, device=device).unsqueeze(0) # [1, T]
        # Mask nằm trong khoảng [mask_start, mask_end) và không đè vào padding
        target_mask = (indices >= mask_start.unsqueeze(1)) & (indices < mask_end.unsqueeze(1))
        target_mask = target_mask & mel_mask
        
        m_exp = target_mask.unsqueeze(-1).float()
    
        # Tạo Flow Matching (Nhiễu hóa TOÀN BỘ Mel)
        t = torch.rand(batch_size, device=device)
        t_exp = t.view(-1, 1, 1)
        x0 = torch.randn_like(x1) # Nhiễu Gaussian trắng
    
        # Công thức tạo xt trên toàn bộ chuỗi: 
        # $x_t = (1 - (1 - \sigma_{min})t)x_0 + tx_1$
        xt = (1.0 - (1.0 - self.sigma_min) * t_exp) * x0 + t_exp * x1
    
        # Tạo cond (Reference Mel)
        cond = x1 * (1.0 - m_exp)
    
        # Vận tốc lý tưởng ut (Ground Truth cho toàn bộ chuỗi hoặc vùng mask)
        # $u_t = x_1 - (1 - \sigma_{min})x_0$
        ut = x1 - (1.0 - self.sigma_min) * x0
        
        return xt, cond, ut, t, target_mask

    # Hàm tính loss chỉ trên vùng target_mask
    def compute_loss(self, vt, ut, target_mask):
        # vt, ut shape: [B, T, D]
        # target_mask shape: [B, T] (1 ở vùng target, 0 ở vùng khác)
        mel_dim = vt.size(-1)
        # Tính bình phương sai lệch
        diff = (vt - ut) ** 2
        
        # Chỉ lấy loss ở vùng target_mask
        loss_sum = (diff.sum(dim=-1) * target_mask).sum()
        
        # Số lượng đơn vị tính loss (tổng số frames trong vùng target)
        num_elements = target_mask.sum()* mel_dim
        
        return loss_sum, num_elements

    
    # Hàm giải ODE với lịch trình thời gian Sway và CFG
    @torch.no_grad()
    def solve_ode(self, model, x0, steps, cond, text_ids, mel_mask, target_mask, sway_coef=-1.0, cfg_scale=1.5, solver="euler"):
        xt = x0
        device = x0.device
        batch_size = x0.size(0)
        
        # Lịch trình Sway
        rho = torch.linspace(0, 1, steps + 1, device=device)
        t_schedule = rho + (sway_coef * torch.sin(2 * np.pi * rho) / (2 * np.pi))

        # =========================================================
        # 🛠️ HÀM TRỢ GIÚP (HELPER): Đóng gói logic CFG để tái sử dụng
        # =========================================================
        def get_velocity(x_val, t_val):
            # Tạo tensor thời gian
            t_tensor = torch.full((batch_size,), t_val.item(), device=device)
            
            if cfg_scale > 1.0:
                # 1. Luồng có điều kiện đầy đủ
                v_cond = model(x=x_val, cond=cond, text_ids=text_ids, t=t_tensor, mask=mel_mask, drop_audio_cond=False)
                
                # 2. Luồng rỗng (Unconditional)
                v_uncond = model(
                    x=x_val, 
                    cond=cond, 
                    text_ids=text_ids, 
                    t=t_tensor, 
                    mask=mel_mask, 
                    drop_audio_cond=True, 
                    drop_text=True
                )
                # Phép ngoại suy CFG
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # Chạy luồng thường nếu không dùng CFG
                return model(x=x_val, cond=cond, text_ids=text_ids, t=t_tensor, mask=mel_mask)

        # =========================================================
        # 🚀 VÒNG LẶP GIẢI PHƯƠNG TRÌNH VI PHÂN (ODE LOOP)
        # =========================================================
        for i in range(steps):
            t_curr = t_schedule[i]
            t_next = t_schedule[i+1]
            dt = t_next - t_curr
            
            # 1. Lấy vận tốc v1 tại điểm hiện tại (Euler / Heun Predictor)
            v1 = get_velocity(xt, t_curr)
            
            # Kiểm tra chế độ Heun (bỏ qua bước Heun ở vòng lặp cuối cùng)
            if solver == "heun" and i < steps - 1:
                # 2. Bước nhảy nháp (Draft Step)
                x_draft = xt + v1 * dt
                
                # 3. Lấy vận tốc v2 tại điểm nháp (Heun Corrector)
                v2 = get_velocity(x_draft, t_next)
                
                # 4. Hiệu chỉnh: Lấy trung bình cộng vận tốc
                v_balanced = 0.5 * (v1 + v2)
                
                # 5. Bước nhảy thật
                xt = xt + v_balanced * dt
            else:
                # Bước nhảy Euler truyền thống
                # (Dùng luôn cho thuật toán Euler hoặc là bước chốt sổ cuối cùng của Heun)
                xt = xt + v1 * dt
                
        # =========================================================
        # 🧩 BƯỚC CUỐI: STITCHING (Hòa trộn kết quả)
        # =========================================================
        m_exp = target_mask.unsqueeze(-1).float()
        
        # x_final = (Kết quả ODE * mask) + (Reference * !mask)
        x_final = xt * m_exp + cond * (1.0 - m_exp)

        # Dọn dẹp vùng padding
        if mel_mask is not None:
            x_final = x_final.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        return x_final