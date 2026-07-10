import torch
import yaml
import json

from huggingface_hub import hf_hub_download
from bigvgan import BigVGAN, AttrDict

from models import ViFlowOTCFM
from trainer import load_checkpoint


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BIGVGAN_REPO = 'nvidia/bigvgan_v2_24khz_100band_256x'

def get_model(model_path, vocab_size,device):
    viflow_model = ViFlowOTCFM(
        dim=config['model']['hidden_dim'],
        depth=config['model']['num_dit_blocks'],
        head_dim=config['model']['head_dim'],
        heads=config['model']['num_heads'],
        text_dim=config['model']['text_dim'],
        mel_dim=config['model']['mel_dim'],
        vocab_size=vocab_size,
        text_embedding_type='convnext',
        text_conformer_layers=config['model']['text_conformer_layers'],
        text_conformer_heads=config['model']['text_conformer_heads'],
        text_convnext_layers=config['model']['text_convnext_layers'],
        pe_attn_head=config['model']['pe_attn_head'],
        dropout=config['model']['dropout']
    ).to(device)
    load_checkpoint(model_path, viflow_model, None, None, device)
    return viflow_model.eval()

def get_bigvgan_vocoder(hf_token, device):
    cfg_path = hf_hub_download(repo_id=BIGVGAN_REPO, filename="config.json", token=hf_token)
    ckpt_path = hf_hub_download(repo_id=BIGVGAN_REPO, filename="bigvgan_generator.pt", token=hf_token)

    with open(cfg_path, 'r') as f:
        h = AttrDict(json.load(f))
    vocoder = BigVGAN(h).to(device)
    vocoder.load_state_dict(torch.load(ckpt_path, map_location=device)['generator'])
    vocoder.eval().remove_weight_norm()
