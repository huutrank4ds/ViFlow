import torch
import torch.nn.functional as F

class ViFlowEngine:
    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    # --- PHẦN 1: TRAINING LOGIC (PATH SAMPLING) ---
    def sample_path(self, x1: torch.Tensor):
        """Tạo x_t, v_target và t cho quá trình huấn luyện."""
        batch_size = x1.size(0)
        device = x1.device
        
        t = torch.rand(batch_size, device=device)
        t_exp = t.view(-1, 1, 1)
        x0 = torch.randn_like(x1)

        # OT Path: x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
        xt = (1.0 - (1.0 - self.sigma_min) * t_exp) * x0 + t_exp * x1
        ut = x1 - (1.0 - self.sigma_min) * x0
        
        return xt, ut, t

    def compute_loss(self, v_pred, v_target, target_lens, n_mels, mask_fn):
        """Tính toán Masked MSE Loss."""
        target_mask = mask_fn(target_lens, max_len=v_target.size(1)).unsqueeze(-1)
        mse = F.mse_loss(v_pred, v_target, reduction="none")
        loss = (mse * target_mask).sum() / (target_mask.sum() * n_mels + 1e-5)
        return loss

    # --- PHẦN 2: INFERENCE LOGIC (ODE SOLVER) ---
    @torch.no_grad()
    def solve_ode(self, model, x0, steps, cond_dict):
        """
        Giải ODE theo phương pháp Euler để sinh Mel từ Nhiễu.
        x0: Nhiễu khởi tạo [B, T, n_mels]
        steps: Số bước (ví dụ 32, 50)
        cond_dict: Các thông tin điều kiện (text, prompt, lens...)
        """
        xt = x0
        dt = 1.0 / steps
        
        for i in range(steps):
            # Tính thời điểm t hiện tại (t chạy từ 0 đến 1)
            t = torch.full((x0.size(0),), i * dt, device=x0.device)
            
            # Dự đoán vận tốc v_t
            vt = model(
                target_xt=xt,
                t=t,
                **cond_dict
            )
            
            # Cập nhật trạng thái theo Euler: x_{t+dt} = x_t + v_t * dt
            xt = xt + vt * dt
            
        return xt # Kết quả cuối cùng là Mel Spectrogram sạch (x1)