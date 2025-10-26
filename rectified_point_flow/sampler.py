"""Sampling utilities for Rectified Point Flow."""

import torch
from typing import Callable
from functools import partial


# Anchor helper

def _reset_anchor(x_t: torch.Tensor, x_0: torch.Tensor, anchor_indices: torch.Tensor | None = None) -> torch.Tensor:
    """Reset anchor parts to the ground truth anchor parts if anchor_indices is not None."""
    if anchor_indices is not None:
        x_t[anchor_indices] = x_0[anchor_indices]
    return x_t


# Base sampler

def flow_sampler(
    step_fn: Callable,
    flow_model_fn: Callable,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    anchor_indices: torch.Tensor | None = None,
    num_steps: int = 20,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Base sampler for rectified point flow.
    
    Args:
        step_fn: Integration step function that takes (x_t, t, dt, flow_model_fn, anchor_indices, x_0).
        flow_model_fn: Partial flow model function that takes (x, timesteps) and returns velocity.
        x_1: Initial noise (B, N, 3).
        x_0: Ground truth anchor points (B, N, 3).
        anchor_indices: Anchor point indices (B, N). If None, no anchor part constraints are applied (used in anchor-free mode).
        num_steps: Number of integration steps, default 20.
        return_trajectory: Whether to return full trajectory, default False.
        
    Returns:
        Final sampled points or trajectory
    """
    dt = 1.0 / num_steps
    x_t = x_1.clone()
    x_t = _reset_anchor(x_t, x_0, anchor_indices)

    if return_trajectory:
        trajectory = torch.empty((num_steps, *x_1.shape), device=x_1.device)
    else:
        trajectory = None

    for step in range(num_steps):
        t = 1 - step * dt
        x_t = step_fn(x_t, t, dt, flow_model_fn, anchor_indices, x_0)
        if return_trajectory:
            trajectory[step] = x_t.detach().clone()

    return trajectory if return_trajectory else x_t.detach()


# Integration step functions

def euler_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
) -> torch.Tensor:
    """Euler integration step."""
    v = flow_model_fn(x_t, t)
    x_t = x_t - dt * v
    x_t = _reset_anchor(x_t, x_0, anchor_indices)
    return x_t

def rk2_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
) -> torch.Tensor:
    """RK2 (midpoint method) integration step."""
    # K1
    v1 = flow_model_fn(x_t, t)

    # K2
    x_mid = x_t - 0.5 * dt * v1
    x_mid = _reset_anchor(x_mid, x_0, anchor_indices)
    t_next = max(0, t - 0.5 * dt)
    v2 = flow_model_fn(x_mid, t_next)

    # RK2 update
    x_t = x_t - dt * (v1 + v2) / 2
    x_t = _reset_anchor(x_t, x_0, anchor_indices)
    return x_t

def rk4_step(
    x_t: torch.Tensor,
    t: float,
    dt: float,
    flow_model_fn: Callable,
    anchor_indices: torch.Tensor,
    x_0: torch.Tensor,
) -> torch.Tensor:
    """RK4 (4th order Runge-Kutta) integration step."""
    # K1
    v1 = flow_model_fn(x_t, t)
    
    # K2
    x_temp = x_t - dt * v1 / 2
    x_temp = _reset_anchor(x_temp, x_0, anchor_indices)
    t_half = max(0, t - dt / 2)
    v2 = flow_model_fn(x_temp, t_half)
    
    # K3
    x_temp = x_t - dt * v2 / 2
    x_temp = _reset_anchor(x_temp, x_0, anchor_indices)
    v3 = flow_model_fn(x_temp, t_half)
    
    # K4
    x_temp = x_t - dt * v3
    x_temp = _reset_anchor(x_temp, x_0, anchor_indices)
    t_next = max(0, t - dt)
    v4 = flow_model_fn(x_temp, t_next)
    
    # RK4 update
    x_t = x_t - dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    x_t = _reset_anchor(x_t, x_0, anchor_indices)
    return x_t


# Sampler factory

def get_sampler(sampler_name: str):
    """Get sampler function by name.
    
    Args:
        sampler_name: Name of the sampler ('euler', 'rk2', 'rk4')
        
    Returns:
        Sampler function with input arguments (step_fn, flow_model_fn, x_1, x_0, anchor_indices, num_steps, return_trajectory=False)
    """
    step_fns = {
        'euler': euler_step,
        'rk2': rk2_step,
        'rk4': rk4_step,
    }
    if sampler_name not in step_fns:
        raise ValueError(f"Unknown sampler: {sampler_name}. Available: {list(step_fns.keys())}")
    
    return partial(flow_sampler, step_fns[sampler_name])
