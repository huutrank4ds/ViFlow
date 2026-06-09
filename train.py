import os
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import contextlib
import gc

# ÉP PHÂN MẢNH BỘ NHỚ VỀ DẠNG QUẢN LÝ LINH HOẠT
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from dataset import ViFlowH5Dataset, VietnamesePhonemeTokenizer, ViFlowCollate, load_viflow_metadata, prepare_viflow_cache
from dynamic_batching import UniversalBucketBatchSampler
from models import ViFlowOTCFM
from engine import ViFlowEngine
from trainer import ViFlowTrainer, load_checkpoint


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def setup_ddp(rank, world_size, config):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        # Ép port phải có trong config, ví dụ: 12355
        os.environ['MASTER_PORT'] = str(config['train']['port']) 
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )
        return torch.device(f'cuda:{rank}')
    else:
        return torch.device('cuda:0')

def train_worker(rank, world_size, config, train_meta, val_meta):
    is_distributed = world_size > 1
    try:
        # 1. SETUP DEVICE
        device = setup_ddp(rank, world_size, config)
        tokenizer = VietnamesePhonemeTokenizer(config['model']['vocab_path'])
        collate_fn = ViFlowCollate(tokenizer)
        
        # 2. KHỞI TẠO MÔ HÌNH (Truy cập trực tiếp key, thiếu là sập ngay)
        model = ViFlowOTCFM(
            dim=config['model']['hidden_dim'],
            depth=config['model']['num_dit_blocks'],
            head_dim=config['model']['head_dim'],
            heads=config['model']['num_heads'],
            text_dim=config['model']['text_dim'],
            mel_dim=config['model']['mel_dim'],
            vocab_size=tokenizer.vocab_size,
            text_embedding_type=config['model']['text_embedding_type'],
            ff_mult=config['model']['ff_mult'],
            text_conformer_layers=config['model']['text_conformer_layers'],
            text_conformer_heads=config['model']['text_conformer_heads'],
            text_convnext_layers=config['model']['text_convnext_layers'],
            pe_attn_head=config['model']['pe_attn_head'],
            dropout=config['model']['dropout']
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']))
        engine = ViFlowEngine()
        scheduler = None
        if config['train']['use_scheduler']:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=config['train']['warmup_steps'], 
                num_training_steps=config['train']['total_steps']
            )

        # 3. LOAD CHECKPOINT 
        resume_path = config['train']['resume_path']
        if resume_path and os.path.exists(resume_path):
            start_epoch, steps, ema_state, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler, device
            )
            start_epoch += 1
        else:
            start_epoch, steps, ema_state, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler, device
            )
        
        if is_distributed:
            model = DDP(model, device_ids=[rank], output_device=rank)

        # 4. DATA LOADER & SAMPLER
        train_dataset = ViFlowH5Dataset(train_meta)
        
        train_sampler = UniversalBucketBatchSampler(
            train_dataset, 
            max_frames=config['train']['max_frames'], # Ép buộc định nghĩa max_frames
            rank=rank,          
            world_size=world_size
        )
        train_sampler.set_epoch(start_epoch)

        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=config['train']['num_workers'],
            persistent_workers=True,
        )

        # 6. VALIDATION LOADER
        val_loader = None
        if rank == 0:
            val_dataset = ViFlowH5Dataset(val_meta)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['train']['val_batch_size'],
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=2,
                persistent_workers=True,
            )

        del train_meta, val_meta
        gc.collect()

        # 7. KHỞI TẠO TRAINER
        trainer = ViFlowTrainer(
            model, optimizer, engine, train_loader, val_loader, config, 
            scheduler=scheduler, rank=rank, steps=steps, ema_state=ema_state
        )

        # 8. MAIN LOOP
        if start_epoch + config['train']['num_epoch_kaggle_version'] + 1 < config['train']['num_epochs']:
            end_epoch = start_epoch + config['train']['num_epoch_kaggle_version'] + 1
        else:
            end_epoch = config['train']['num_epochs']
            
        for epoch in range(start_epoch, end_epoch):
            train_sampler.set_epoch(epoch)
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0)
            
            total_epoch_loss_sum = 0.0
            total_epoch_elements = 0.0
            join_context = model.join() if is_distributed else contextlib.nullcontext()

            with join_context:
                for batch in pbar:
                    try:
                        batch_loss_sum, batch_elements = trainer.train_step(
                            batch, 
                            p_drop_audio_cond = config['train']['p_drop_audio_cond'],
                            p_drop_text = config['train']['p_drop_text']
                        )
                        total_epoch_loss_sum += batch_loss_sum
                        total_epoch_elements += batch_elements
                        
                        if rank == 0:
                            current_batch_avg = batch_loss_sum / batch_elements
                            pbar.set_postfix({
                                "loss": f"{current_batch_avg:.4f}", 
                                "step": trainer.step
                            })
                            if trainer.step % 100 == 0:
                                print(f"| Step {trainer.step} | Learning rate: {trainer.get_lr()} |")
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"\n[RANK {rank}] OOM tại Step {trainer.step:.4f}")
                            torch.cuda.empty_cache()
                        raise e

            # ĐỒNG BỘ METRICS CUỐI EPOCH
            if is_distributed:
                metrics = torch.tensor([total_epoch_loss_sum, total_epoch_elements], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                global_loss_sum, global_elements = metrics.tolist()
                avg_train_loss = global_loss_sum / global_elements
            else:
                avg_train_loss = total_epoch_loss_sum / total_epoch_elements
            
            if rank == 0:
                print(f"Đang bắt đầu validation cho Epoch {epoch}...")
                val_loss, ema_val_loss = trainer.validate_and_sample(epoch)
                print(f"| Epoch {epoch:3d} | Step {trainer.step:7d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | EMA Loss: {ema_val_loss:.6f} |")
                trainer.save_checkpoint(epoch, val_loss)
                with open(trainer.log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, trainer.step, avg_train_loss, val_loss, ema_val_loss])

            if is_distributed: 
                dist.barrier()

    except Exception as e:
        print(f"Error Rank {rank}: {e}")
        raise e
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    with open("configs.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    prepare_viflow_cache(config)
    train_meta, val_meta = load_viflow_metadata(config)
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(train_worker, args=(world_size, config, train_meta, val_meta), nprocs=world_size, join=True)