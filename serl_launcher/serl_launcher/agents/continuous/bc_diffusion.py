"""Behavior Cloning agent with Diffusion Policy."""
from functools import partial
from typing import Any, Iterable, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.networks.diffusion import DDPMSchedule, DDIMSampler, UNet1D
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.vision.data_augmentations import batched_random_crop


class DiffusionPolicy(nn.Module):
    """Diffusion policy network wrapper."""
    encoder: nn.Module
    denoising_network: nn.Module

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 noisy_actions: jnp.ndarray,
                 timesteps: jnp.ndarray,
                 train: bool = False) -> jnp.ndarray:
        """
        Forward pass for diffusion policy.

        Args:
            observations: Observation dict
            noisy_actions: Noisy action sequence (batch, action_horizon, action_dim)
            timesteps: Diffusion timesteps (batch,)
            train: Training mode

        Returns:
            Predicted noise (batch, action_horizon, action_dim)
        """
        # Encode observations
        obs_encoding = self.encoder(observations, train=train, stop_gradient=False)

        # Denoise using U-Net
        predicted_noise = self.denoising_network(
            noisy_actions=noisy_actions,
            timestep=timesteps,
            obs_encoding=obs_encoding,
            train=train
        )

        return predicted_noise


class BCDiffusionAgent(flax.struct.PyTreeNode):
    """BC agent using diffusion policy for action generation."""
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    noise_schedule: Any = nonpytree_field()
    ddim_sampler: Any = nonpytree_field()
    timesteps_list: list = nonpytree_field()  # Pre-computed Python list for JIT compatibility

    def data_augmentation_fn(self, rng, observations):
        """Apply data augmentation to image observations."""
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        """
        Update the diffusion policy using DDPM training.

        Args:
            batch: Training batch with observations and actions
            pmap_axis: Axis for parallel mapping (not used for single GPU)

        Returns:
            Updated agent and training info
        """
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            # Sample random timesteps
            rng, timestep_key, noise_key = jax.random.split(rng, 3)

            batch_size = batch["actions"].shape[0]
            timesteps = jax.random.randint(
                timestep_key,
                (batch_size,),
                0,
                self.noise_schedule.num_train_timesteps
            )

            # Get clean actions and reshape for action horizon
            # Current: (batch, action_dim) -> (batch, action_horizon, action_dim)
            action_horizon = self.config["action_horizon"]
            clean_actions = batch["actions"][:, None, :].repeat(action_horizon, axis=1)

            # Sample noise
            noise = jax.random.normal(noise_key, clean_actions.shape)

            # Add noise to actions (forward diffusion)
            noisy_actions = self.noise_schedule.add_noise(clean_actions, noise, timesteps)

            # Predict noise using the model
            rng, dropout_key = jax.random.split(rng)
            predicted_noise = self.state.apply_fn(
                {"params": params},
                observations=batch["observations"],
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                train=True,
                rngs={"dropout": dropout_key},
                name="actor"
            )

            # MSE loss between predicted and actual noise
            loss = jnp.mean((predicted_noise - noise) ** 2)

            return loss, {
                "diffusion_loss": loss,
                "mean_timestep": jnp.mean(timesteps.astype(jnp.float32)),
            }

        # Compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("num_inference_steps",))
    def _forward_pass_jitted(
        self,
        observations: jnp.ndarray,
        noisy_actions: jnp.ndarray,
        timestep: jnp.ndarray,
        num_inference_steps: int,
    ) -> jnp.ndarray:
        """JIT-compiled forward pass for noise prediction."""
        return self.state.apply_fn(
            {"params": self.state.params},
            observations=observations,
            noisy_actions=noisy_actions,
            timesteps=timestep,
            train=False,
            name="actor"
        )

    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> jnp.ndarray:
        """
        Sample actions using DDIM reverse diffusion.

        Args:
            observations: Current observations
            seed: Random seed
            temperature: Not used for diffusion, kept for API compatibility
            argmax: Not used for diffusion, kept for API compatibility

        Returns:
            Sampled actions (batch, action_dim)
        """
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        action_horizon = self.config["action_horizon"]
        action_dim = self.config["action_dim"]

        # Start from pure noise
        rng = seed if seed is not None else jax.random.PRNGKey(0)
        actions = jax.random.normal(rng, (batch_size, action_horizon, action_dim))

        # DDIM denoising loop (use pre-computed timesteps list for JIT compatibility)
        alphas_cumprod = self.noise_schedule.alphas_cumprod
        num_steps = len(self.timesteps_list)

        for i, t in enumerate(self.timesteps_list):
            # JIT-compiled forward pass
            predicted_noise = self._forward_pass_jitted(
                observations=observations,
                noisy_actions=actions,
                timestep=jnp.full((batch_size,), t, dtype=jnp.int32),
                num_inference_steps=num_steps,
            )

            # DDIM step - t is now a Python int, not a traced value
            alpha_prod_t = alphas_cumprod[t]

            # Get next timestep's alpha
            if i + 1 < num_steps:
                alpha_prod_t_prev = alphas_cumprod[self.timesteps_list[i + 1]]
            else:
                alpha_prod_t_prev = jnp.ones_like(alpha_prod_t)

            # Predict x_0
            pred_original_sample = (
                actions - jnp.sqrt(1 - alpha_prod_t) * predicted_noise
            ) / jnp.sqrt(alpha_prod_t)

            # Compute direction pointing to x_t
            pred_sample_direction = jnp.sqrt(1 - alpha_prod_t_prev) * predicted_noise

            # Compute x_{t-1}
            actions = jnp.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # Return only the first action in the sequence (action_horizon dimension)
        # Shape: (batch, action_horizon, action_dim) -> (batch, action_dim)
        return actions[:, 0, :]

    def sample_actions_from_noise(
        self,
        observations: np.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Sample actions using DDIM with pre-specified noise.

        This is used by DSRL to steer the diffusion policy by providing
        learned noise instead of random sampling.

        Args:
            observations: Current observations
            noise: Pre-specified noise vector (batch, noise_dim)
                   where noise_dim = action_horizon * action_dim

        Returns:
            Sampled actions (batch, action_dim)
        """
        batch_size = jax.tree_util.tree_leaves(observations)[0].shape[0]
        action_horizon = self.config["action_horizon"]
        action_dim = self.config["action_dim"]

        # Reshape noise to (batch, action_horizon, action_dim)
        actions = noise.reshape(batch_size, action_horizon, action_dim)

        # DDIM denoising loop (JIT-compatible version)
        # Use pre-computed timesteps list (Python ints, not traced)
        alphas_cumprod = self.noise_schedule.alphas_cumprod
        num_steps = len(self.timesteps_list)

        for i, t in enumerate(self.timesteps_list):
            # JIT-compiled forward pass
            predicted_noise = self._forward_pass_jitted(
                observations=observations,
                noisy_actions=actions,
                timestep=jnp.full((batch_size,), t, dtype=jnp.int32),
                num_inference_steps=num_steps,
            )

            # DDIM step - t is now a Python int, not a traced value
            alpha_prod_t = alphas_cumprod[t]

            # Get next timestep's alpha
            if i + 1 < num_steps:
                t_next = self.timesteps_list[i + 1]
                alpha_prod_t_prev = alphas_cumprod[t_next]
            else:
                alpha_prod_t_prev = jnp.ones_like(alpha_prod_t)

            # Predict x_0
            pred_original_sample = (
                actions - jnp.sqrt(1 - alpha_prod_t) * predicted_noise
            ) / jnp.sqrt(alpha_prod_t)

            # Compute direction pointing to x_t
            pred_sample_direction = jnp.sqrt(1 - alpha_prod_t_prev) * predicted_noise

            # Compute x_{t-1}
            actions = jnp.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # Return only the first action in the sequence
        return actions[:, 0, :]

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """Get debug metrics for logging."""
        # Sample actions
        rng = jax.random.PRNGKey(0)
        sampled_actions = self.sample_actions(batch["observations"], seed=rng)

        # Compare with ground truth (only first action in sequence)
        batch_actions = batch["actions"]
        mse = jnp.mean((sampled_actions - batch_actions) ** 2, axis=-1)

        return {
            "mse": mse,
            "sampled_actions": sampled_actions,
        }

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        # Diffusion parameters
        action_horizon: int = 1,
        num_train_timesteps: int = 20,
        num_inference_timesteps: int = 8,
        beta_schedule: str = "cosine",
        # U-Net architecture
        unet_down_features: tuple = (256, 512, 1024, 2048),
        unet_mid_features: int = 2048,
        unet_time_emb_dim: int = 128,
        obs_encoding_dim: int = 512,
        # Optimizer
        learning_rate: float = 3e-4,
        augmentation_function: Optional[callable] = None,
    ):
        """
        Create a new BCDiffusionAgent.

        Args:
            rng: Random key
            observations: Sample observations for initialization
            actions: Sample actions for initialization
            encoder_type: Type of visual encoder
            image_keys: Keys for image observations
            use_proprio: Whether to use proprioceptive observations
            action_horizon: Number of future actions to predict
            num_train_timesteps: Number of diffusion steps during training
            num_inference_timesteps: Number of diffusion steps during inference
            beta_schedule: Noise schedule type
            unet_down_features: Feature dimensions for U-Net downsampling
            unet_mid_features: Middle layer feature dimension
            unet_time_emb_dim: Time embedding dimension
            obs_encoding_dim: Observation encoding dimension
            learning_rate: Learning rate
            augmentation_function: Optional data augmentation function

        Returns:
            BCDiffusionAgent instance
        """
        action_dim = actions.shape[-1]

        # Create encoder
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        # Create U-Net denoising network
        denoising_network = UNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_encoding_dim=obs_encoding_dim,
            down_features=unet_down_features,
            mid_features=unet_mid_features,
            time_emb_dim=unet_time_emb_dim,
        )

        # Create diffusion policy
        networks = {
            "actor": DiffusionPolicy(
                encoder=encoder_def,
                denoising_network=denoising_network,
            )
        }

        model_def = ModuleDict(networks)

        # Initialize noise schedule and sampler
        noise_schedule = DDPMSchedule(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )
        ddim_sampler = DDIMSampler(
            schedule=noise_schedule,
            num_inference_steps=num_inference_timesteps
        )

        # Initialize model parameters
        tx = optax.adam(learning_rate)

        rng, init_rng = jax.random.split(rng)

        # Create dummy inputs for initialization
        dummy_actions = jnp.zeros((1, action_horizon, action_dim))
        dummy_timesteps = jnp.zeros((1,), dtype=jnp.int32)

        params = model_def.init(
            init_rng,
            actor=[observations, dummy_actions, dummy_timesteps]
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )

        # Pre-compute timesteps as Python list for JIT compatibility
        timesteps_list = [int(t) for t in ddim_sampler.timesteps]

        agent = cls(
            state=state,
            config=config,
            noise_schedule=noise_schedule,
            ddim_sampler=ddim_sampler,
            timesteps_list=timesteps_list,
        )

        # Load pretrained weights if using pretrained encoder
        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
