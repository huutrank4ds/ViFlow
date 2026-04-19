"""Inference utilities: Euler ODE integration for OT-CFM mel generation."""

from __future__ import annotations

from typing import Sequence

import torch
from torchdiffeq import odeint

from models import ViFlowOTCFM


class EulerCFMSolver:
    """Simple Euler solver for dx/dt = v_theta(x_t, t, c)."""

    def __init__(self, model: ViFlowOTCFM, n_steps: int = 32) -> None:
        self.model = model
        self.n_steps = n_steps

    @torch.no_grad()
    def sample(
        self,
        texts: Sequence[str],
        prompt_wav: torch.Tensor,
        target_frames: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate mel with Euler integration.

        Args:
            texts: list[str] length B.
            prompt_wav: [B, T_prompt_wav]
            target_frames: number of output mel frames.

        Returns:
            mel_hifigan: [B, 80, T_target]
        """

        device = next(self.model.parameters()).device
        bsz = prompt_wav.size(0)
        prompt_wav = prompt_wav.to(device)

        x = torch.randn(bsz, target_frames, self.model.n_mels, device=device) * temperature
        # shape: [B, T_target, 80]

        dt = 1.0 / float(self.n_steps)
        for i in range(self.n_steps):
            t_val = float(i) / float(self.n_steps)
            t = torch.full((bsz, 1), t_val, device=device)
            v = self.model.predict_vector_field(
                texts=texts,
                prompt_wav=prompt_wav,
                target_state=x,
                t=t,
                update_vocab=False,
            )  # shape: [B, T_target, 80]
            x = x + dt * v  # shape: [B, T_target, 80]

        mel_hifigan = self.model.format_mel_for_hifigan(x)  # shape: [B, 80, T_target]
        return mel_hifigan


class ODEIntCFMSolver:
    """Alternative torchdiffeq-based solver wrapper (default integrator: rk4)."""

    def __init__(self, model: ViFlowOTCFM) -> None:
        self.model = model

    @torch.no_grad()
    def sample(
        self,
        texts: Sequence[str],
        prompt_wav: torch.Tensor,
        target_frames: int,
        n_eval_steps: int = 32,
        method: str = "rk4",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        bsz = prompt_wav.size(0)
        prompt_wav = prompt_wav.to(device)

        x0 = torch.randn(bsz, target_frames, self.model.n_mels, device=device) * temperature
        t_span = torch.linspace(0.0, 1.0, n_eval_steps, device=device)

        def func(t_scalar: torch.Tensor, x_state: torch.Tensor) -> torch.Tensor:
            t = torch.full((bsz, 1), float(t_scalar.item()), device=device)
            return self.model.predict_vector_field(
                texts=texts,
                prompt_wav=prompt_wav,
                target_state=x_state,
                t=t,
                update_vocab=False,
            )

        traj = odeint(func, x0, t_span, method=method)
        x_final = traj[-1]  # shape: [B, T_target, 80]
        mel_hifigan = self.model.format_mel_for_hifigan(x_final)  # shape: [B, 80, T_target]
        return mel_hifigan
