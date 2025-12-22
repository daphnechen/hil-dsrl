"""DSRL-specific network architectures."""
from typing import Optional
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class ActionCritic(nn.Module):
    """
    Q^A(s,a): Action-space critic for DSRL.

    Standard Q-network that operates on (state, action) pairs.
    Used to learn values of real actions in the environment.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        train: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass for action-space critic.

        Args:
            observations: State observations
            actions: Actions in original action space A
            train: Whether in training mode

        Returns:
            Q-value estimates (batch_size,)
        """
        # Encode observations
        if self.encoder is None:
            obs_enc = observations['state']
        else:
            obs_enc = self.encoder(observations)

        # Concatenate obs encoding with actions
        inputs = jnp.concatenate([obs_enc, actions], axis=-1)

        # MLP processing
        outputs = self.network(inputs, train)

        # Final Q-value projection
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class NoiseCritic(nn.Module):
    """
    Q^W(s,w): Latent-noise critic for DSRL.

    Q-network that operates on (state, noise) pairs.
    Learns to predict action values in the latent noise space.
    Distills knowledge from Q^A by exploiting noise aliasing.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        noise: jnp.ndarray,
        train: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass for noise-space critic.

        Args:
            observations: State observations
            noise: Latent noise vectors w in noise space W
            train: Whether in training mode

        Returns:
            Q-value estimates in noise space (batch_size,)
        """
        # Encode observations
        if self.encoder is None:
            obs_enc = observations['state']
        else:
            obs_enc = self.encoder(observations)

        # Concatenate obs encoding with noise
        inputs = jnp.concatenate([obs_enc, noise], axis=-1)

        # MLP processing
        outputs = self.network(inputs, train)

        # Final Q-value projection
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)


class SteeringActor(nn.Module):
    """
    π^W(s): Steering policy for DSRL.

    Learns which latent noise w to use at each state s.
    Outputs a Gaussian distribution over the noise space.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    noise_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = False
    ) -> distrax.Distribution:
        """
        Forward pass for steering policy.

        Args:
            observations: State observations
            train: Whether in training mode

        Returns:
            Gaussian distribution over noise space
        """
        # Encode observations
        if self.encoder is None:
            obs_enc = observations['state']
        else:
            obs_enc = self.encoder(observations)

        # MLP processing
        outputs = self.network(obs_enc, train)

        # Output Gaussian distribution parameters
        means = nn.Dense(
            self.noise_dim,
            kernel_init=default_init()
        )(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.noise_dim,
                kernel_init=default_init()
            )(outputs)
        else:
            # Learnable but state-independent std
            log_stds = self.param(
                'log_stds',
                nn.initializers.zeros,
                (self.noise_dim,)
            )
            log_stds = jnp.broadcast_to(log_stds, means.shape)

        # Clip log_stds for stability
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # Create Gaussian distribution
        if self.tanh_squash_distribution:
            # Use tanh-squashed Gaussian (similar to SAC)
            distribution = distrax.Transformed(
                distrax.MultivariateNormalDiag(means, jnp.exp(log_stds)),
                distrax.Block(distrax.Tanh(), ndims=1)
            )
        else:
            # Standard Gaussian (noise space is unbounded)
            distribution = distrax.MultivariateNormalDiag(means, jnp.exp(log_stds))

        return distribution


def ensemblize_dsrl_critics(critic_class, num_critics: int, **kwargs):
    """
    Create an ensemble of critics for DSRL (similar to SAC ensemble).

    Args:
        critic_class: Either ActionCritic or NoiseCritic
        num_critics: Number of critics in ensemble
        **kwargs: Arguments to pass to critic constructor

    Returns:
        Ensemble critic module
    """
    class EnsembleCritic(nn.Module):
        @nn.compact
        def __call__(self, *args, **inner_kwargs):
            ensemble = nn.vmap(
                critic_class,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=None,
                out_axes=0,
                axis_size=num_critics,
            )(**kwargs)
            return ensemble(*args, **inner_kwargs)

    return EnsembleCritic()
