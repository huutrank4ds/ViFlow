import os
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import librosa.display
import csv
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import gc
import glob

class EMA:
    def __init__(self, model_instance, beta=0.9999):
        from copy import deepcopy
        self.model = deepcopy(model_instance)
        self.beta = beta
        self.model.eval()
        # Đóng băng hoàn toàn EMA model
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Lấy model thực (handle cả DDP)
        active_model = model.module if hasattr(model, 'module') else model
        
        # Cập nhật trực tiếp qua parameter generator, không tạo Dictionary mới
        for ema_param, model_param in zip(self.model.parameters(), active_model.parameters()):
            ema_param.data.mul_(self.beta).add_(model_param.data, alpha=1.0 - self.beta)
        
        # Cập nhật cả buffers (như BatchNorm running mean) nếu có
        for ema_buffer, model_buffer in zip(self.model.buffers(), active_model.buffers()):
            ema_buffer.copy_(model_buffer)

class ViFlowTrainer:
    def __init__(self, model, optimizer, engine, train_loader, val_loader, config, scheduler=None, rank=0, steps=0, ema_state=None):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')
        self.is_ddp = dist.is_initialized()
        self.is_main = rank == 0
        
        self.model = model.to(self.device)
        self.raw_model = model.module if self.is_ddp else model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.engine = engine
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.step = steps
        
        # Lấy grad_accum_steps từ file config
        self.grad_accum = config['train']["grad_accum_steps"]
        self.scaler = GradScaler('cuda', enabled=config["train"]["use_amp"])
        
        self.ema = EMA(self.raw_model, beta=config['train'].get("ema_beta", 0.9999))
        if ema_state:
            self.ema.model.load_state_dict(ema_state)
            
        if self.is_main:
            self._init_logs()

    def _init_logs(self):
        os.makedirs(self.config['train']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['train']['log_dir'], exist_ok=True)
        os.makedirs("val_plots", exist_ok=True)
        self.log_path = os.path.join(self.config['train']['log_dir'], "train_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'step', 'train_loss', 'val_loss', 'ema_val_loss'])

    def get_lr(self):
        # Trả về LR hiện tại của nhóm tham số đầu tiên
        return self.optimizer.param_groups[0]['lr']

    def train_step(self, batch, p_drop_audio_cond, p_drop_text):
        self.model.train()
        mels = batch["mels"].to(self.device, non_blocking=True)
        mel_lens = batch["mel_lens"].to(self.device, non_blocking=True)
        mel_mask = batch["mel_mask"].to(self.device, non_blocking=True)
        phonemes = batch["phonemes"].to(self.device, non_blocking=True)
        batch_size = mels.shape[0]
    
        context = self.model.no_sync() if self.is_ddp and (self.step + 1) % self.grad_accum != 0 else torch.enable_grad()
        
        with context:
            xt, cond, ut, t, target_mask = self.engine.get_train_batch(mels, mel_lens, mel_mask)
            
            drop_audio_mask = torch.rand(batch_size, device=self.device) < p_drop_audio_cond
            drop_text_mask = torch.rand(batch_size, device=self.device) < p_drop_text
                
            with autocast("cuda", enabled=self.config["train"]["use_amp"]):
                vt = self.model(
                    x=xt, 
                    cond=cond, 
                    text_ids=phonemes, 
                    t=t, 
                    mel_lens=mel_lens, 
                    mask=mel_mask,
                    drop_audio_mask=drop_audio_mask,
                    drop_text_mask=drop_text_mask
                )
                # Lấy TỔNG loss và TỔNG số phần tử vùng target
                loss_sum, num_elements = self.engine.compute_loss(vt, ut, target_mask)
                
                # Tính mean cục bộ để backward (phải giữ grad ở đây)
                loss_to_backward = (loss_sum / num_elements) / self.grad_accum
                
            self.scaler.scale(loss_to_backward).backward()
    
        if (self.step + 1) % self.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config['train']["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.ema.update(self.model)

            if self.scheduler is not None:
                self.scheduler.step()
    
        self.step += 1
        
        # Trả về giá trị đã đồng bộ hóa (Sử dụng detach để fix Warning)
        return loss_sum.detach().item(), num_elements.detach().item()
        

    @torch.no_grad()
    def validate_and_sample(self, epoch, num_samples=2):
        self.raw_model.eval()
        
        total_val_loss_sum = 0.0
        total_val_elements = 0.0
        total_ema_loss_sum = 0.0
        
        collected_data = { "mels": [], "mel_lens": [], "mel_mask": [], "phonemes": [] }
        total_collected = 0

        # Vòng lặp validation thuần túy
        for batch in self.val_loader:
            mels = batch["mels"].to(self.device, non_blocking=True)
            mel_lens = batch["mel_lens"].to(self.device, non_blocking=True)
            mel_mask = batch["mel_mask"].to(self.device, non_blocking=True)
            phonemes = batch["phonemes"].to(self.device, non_blocking=True)

            if self.is_main and total_collected < num_samples:
                needed = num_samples - total_collected
                for k in collected_data.keys():
                    collected_data[k].append(batch[k][:needed].cpu())
                total_collected += batch[k][:needed].size(0)

            with autocast("cuda", enabled=self.config["train"]["use_amp"]):
                xt, cond, ut, t, target_mask = self.engine.get_train_batch(mels, mel_lens, mel_mask)
                v_online = self.raw_model(x=xt, cond=cond, text_ids=phonemes, t=t, mel_lens=mel_lens, mask=mel_mask)
                v_ema = self.ema.model(x=xt, cond=cond, text_ids=phonemes, t=t, mel_lens=mel_lens, mask=mel_mask)
                
                l_sum, n_el = self.engine.compute_loss(v_online, ut, target_mask)
                le_sum, _ = self.engine.compute_loss(v_ema, ut, target_mask)
                
                total_val_loss_sum += l_sum.item()
                total_val_elements += n_el.item()
                total_ema_loss_sum += le_sum.item()

        avg_val_loss = total_val_loss_sum / total_val_elements if total_val_elements > 0 else 0.0
        avg_ema_loss = total_ema_loss_sum / total_val_elements if total_val_elements > 0 else 0.0

        if self.is_main:
            sample_batch = {k: torch.cat(v, dim=0) for k, v in collected_data.items()}
            
            print("Bắt đầu vẽ mel dự đoán...")
            self.sample_and_plot(epoch, sample_batch, num_samples=num_samples)
            print(f"Đã vẽ xong và xuất mel dự đoán tại epoch {epoch} step {self.step}!")
            
            del sample_batch, collected_data
            gc.collect()
            torch.cuda.empty_cache()
            
        return avg_val_loss, avg_ema_loss

    @torch.no_grad()
    def sample_and_plot(self, epoch, sample_batch, num_samples=2):
        mels_gt = sample_batch["mels"].to(self.device)
        mel_lens = sample_batch["mel_lens"].to(self.device)
        mel_mask = sample_batch["mel_mask"].to(self.device)
        text_ids = sample_batch["phonemes"].to(self.device)

        p_len = (0.3 * mel_lens).long()
        indices = torch.arange(mels_gt.size(1), device=self.device).unsqueeze(0)
        target_mask = mel_mask & (indices >= p_len.unsqueeze(1))
        cond = mels_gt * (~target_mask).unsqueeze(-1).float()
        x0 = torch.randn_like(mels_gt)

        x_gen = self.engine.solve_ode(
            model=self.ema.model, x0=x0, cond=cond, text_ids=text_ids, mel_mask=mel_mask, target_mask=target_mask,
            steps=self.config['inference'].get('ode_steps', 32)
        )

        for i in range(min(num_samples, x_gen.size(0))):
            self._plot_mel(mels_gt[i], x_gen[i], mel_lens[i], p_len[i], epoch, i)
            
        del mels_gt, x_gen, x0, cond, target_mask

    def _plot_mel(self, real, fake, length, p_end, epoch, idx):
        real_np = real[:length].cpu().numpy().T
        fake_np = fake[:length].cpu().numpy().T
        
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvasAgg(fig)
        ax1, ax2 = fig.subplots(2, 1)
        
        librosa.display.specshow(real_np, ax=ax1, x_axis='time', y_axis='mel', sr=24000, hop_length=256)
        ax1.axvline(x=p_end.item()*256/24000, color='r', linestyle='--')
        ax1.set_title("Real Mel")
        
        librosa.display.specshow(fake_np, ax=ax2, x_axis='time', y_axis='mel', sr=24000, hop_length=256)
        ax2.axvline(x=p_end.item()*256/24000, color='r', linestyle='--')
        ax2.set_title(f"Gen Mel (Epoch {epoch})")
        
        fig.tight_layout()
        fig.savefig(f"val_plots/epoch_{epoch}_sample_{idx}.png")
        
        fig.clear()
        del fig, canvas, ax1, ax2

    def save_checkpoint(self, epoch, val_loss, max_to_keep=2):
        if not self.is_main: return
        ckpt = {
            'epoch': epoch, 'step': self.step,
            'model_state': self.raw_model.state_dict(),
            'ema_state': self.ema.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
            'val_loss': val_loss, 'config': self.config
        }
        save_dir = self.config['train']['checkpoint_dir']
        path = os.path.join(save_dir, f"viflow_epoch_{epoch}.pt")
        torch.save(ckpt, path)

        checkpoints = sorted(glob.glob(os.path.join(save_dir, "viflow_epoch_*.pt")), 
                             key=os.path.getmtime)
        if len(checkpoints) > max_to_keep:
            for old_ckpt in checkpoints[:-max_to_keep]:
                try:
                    os.remove(old_ckpt)
                    print(f"🗑️ Đã xóa checkpoint cũ: {old_ckpt}")
                except Exception as e:
                    print(f"⚠️ Không thể xóa {old_ckpt}: {e}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    if not path or not os.path.exists(path): 
        print("Không tìm thấy file checkpoint!")
        return 0, 0, None, 1000.0
    print(f"Đang đọc trọng số từ file checkpoint {path}!")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print("Nạp trọng số mô hình thành công!")
    
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print("Nạp trạng thái optimizer thành công!")
    
    if scheduler and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])
        print("Nạp trạng thái scheduler thành công!")
    ema_state = ckpt.get('ema_state')
    
    return ckpt.get('epoch'), ckpt.get('step'), ema_state, ckpt.get('val_loss')