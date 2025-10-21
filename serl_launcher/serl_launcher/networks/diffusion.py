"""Diffusion policy networks and utilities for JAX."""
from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.common.common import default_init


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding."""
    dim: int

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class ResNetBlock(nn.Module):
    """Residual block for U-Net."""
    features: int
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, x, time_emb, train: bool = False):
        h = x

        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = nn.swish(h)
        h = nn.Dense(self.features, kernel_init=default_init())(h)

        # Add time embedding
        time_h = nn.swish(time_emb)
        time_h = nn.Dense(self.features, kernel_init=default_init())(time_h)
        h = h + time_h[:, None, :] if len(h.shape) == 3 else h + time_h

        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = nn.swish(h)
        h = nn.Dense(self.features, kernel_init=default_init())(h)

        # Residual connection
        if x.shape[-1] != self.features:
            x = nn.Dense(self.features, kernel_init=default_init())(x)

        return x + h


class UNet1D(nn.Module):
    """
    1D U-Net for action denoising in diffusion policy.

    Args:
        action_dim: Dimension of action space
        action_horizon: Number of actions to predict (action chunk size)
        obs_encoding_dim: Dimension of observation encoding
        down_features: Feature dimensions for downsampling layers
        mid_features: Feature dimension for middle layer
        time_emb_dim: Dimension of time embedding
    """
    action_dim: int
    action_horizon: int = 1
    obs_encoding_dim: int = 512
    down_features: Sequence[int] = (256, 512, 1024, 2048)
    mid_features: int = 2048
    time_emb_dim: int = 128

    @nn.compact
    def __call__(self,
                 noisy_actions: jnp.ndarray,  # (batch, action_horizon, action_dim)
                 timestep: jnp.ndarray,  # (batch,)
                 obs_encoding: jnp.ndarray,  # (batch, obs_encoding_dim)
                 train: bool = False) -> jnp.ndarray:
        """
        Args:
            noisy_actions: Noisy action sequence (batch, action_horizon, action_dim)
            timestep: Diffusion timestep (batch,)
            obs_encoding: Encoded observations (batch, obs_encoding_dim)
            train: Training mode

        Returns:
            Predicted noise or denoised action (batch, action_horizon, action_dim)
        """
        # Time embedding
        time_emb = SinusoidalPosEmb(self.time_emb_dim)(timestep)

        # Flatten action sequence: (batch, action_horizon, action_dim) -> (batch, action_horizon * action_dim)
        batch_size = noisy_actions.shape[0]
        x = noisy_actions.reshape(batch_size, -1)

        # Ensure obs_encoding has correct shape
        # If obs_encoding is 1D (features,), expand to (1, features)
        # If obs_encoding is already 2D (batch, features), keep as is
        if len(obs_encoding.shape) == 1:
            obs_encoding = obs_encoding[None, :]
        elif len(obs_encoding.shape) == 2 and obs_encoding.shape[0] != batch_size:
            # If batch dimension doesn't match, repeat across batch
            obs_encoding = jnp.repeat(obs_encoding, batch_size, axis=0)

        # Concatenate observation encoding with flattened actions
        x = jnp.concatenate([x, obs_encoding], axis=-1)

        # Initial projection
        x = nn.Dense(self.down_features[0], kernel_init=default_init())(x)

        # Encoder (downsampling path)
        skip_connections = []
        for features in self.down_features:
            x = ResNetBlock(features)(x, time_emb, train=train)
            skip_connections.append(x)
            # Downsample by reducing dimension slightly
            if features != self.down_features[-1]:
                x = nn.Dense(features, kernel_init=default_init())(x)

        # Middle
        x = ResNetBlock(self.mid_features)(x, time_emb, train=train)

        # Decoder (upsampling path with skip connections)
        for i, features in enumerate(reversed(self.down_features)):
            # Skip connection
            skip = skip_connections[-(i+1)]
            x = jnp.concatenate([x, skip], axis=-1)
            x = ResNetBlock(features)(x, time_emb, train=train)

        # Final projection back to action space
        x = nn.swish(x)
        x = nn.Dense(self.action_horizon * self.action_dim, kernel_init=default_init())(x)

        # Reshape back to (batch, action_horizon, action_dim)
        x = x.reshape(batch_size, self.action_horizon, self.action_dim)

        return x


class DDPMSchedule:
    """DDPM noise schedule and utilities."""

    def __init__(self, num_train_timesteps: int = 20, beta_schedule: str = "cosine"):
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")

        self.num_train_timesteps = num_train_timesteps
        self.betas = betas

        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = jnp.concatenate([jnp.ones(1), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start, noise, timesteps):
        """
        Forward diffusion: q(x_t | x_0)

        Args:
            x_start: Clean actions (batch, action_horizon, action_dim)
            noise: Gaussian noise (same shape as x_start)
            timesteps: Timestep indices (batch,)

        Returns:
            Noisy actions at timestep t
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting: (batch,) -> (batch, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None]

        noisy_actions = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy_actions


class DDIMSampler:
    """DDIM sampler for fast inference."""

    def __init__(self, schedule: DDPMSchedule, num_inference_steps: int = 8):
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps

        # Create subset of timesteps for DDIM
        step_ratio = schedule.num_train_timesteps // num_inference_steps
        self.timesteps = jnp.arange(0, schedule.num_train_timesteps, step_ratio)[::-1]

    def step(self, model_output, timestep, sample, eta: float = 0.0):
        """
        Single DDIM denoising step.

        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            eta: Stochasticity parameter (0 = deterministic DDIM)

        Returns:
            Denoised sample at previous timestep
        """
        # Get schedule values
        alpha_prod_t = self.schedule.alphas_cumprod[timestep]

        # Find previous timestep
        prev_timestep_idx = jnp.where(self.timesteps == timestep)[0][0] + 1
        alpha_prod_t_prev = jnp.where(
            prev_timestep_idx < len(self.timesteps),
            self.schedule.alphas_cumprod[self.timesteps[prev_timestep_idx]],
            jnp.ones_like(alpha_prod_t)
        )

        # Predict x_0 from model output (predicted noise)
        pred_original_sample = (
            sample - jnp.sqrt(1 - alpha_prod_t) * model_output
        ) / jnp.sqrt(alpha_prod_t)

        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * jnp.sqrt(variance)

        # Compute direction pointing to x_t
        pred_sample_direction = jnp.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output

        # Compute x_{t-1}
        prev_sample = jnp.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        return prev_sample
