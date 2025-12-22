"""DSRL (Diffusion Steering via Reinforcement Learning) Agent."""
from functools import partial
from typing import Optional, Tuple, FrozenSet
import numpy as np

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.dsrl_nets import ActionCritic, NoiseCritic, SteeringActor, ensemblize_dsrl_critics
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.agents.continuous.bc_diffusion import BCDiffusionAgent


class DSRLAgent(flax.struct.PyTreeNode):
    """
    DSRL Agent: Learn to steer a frozen diffusion policy via RL in latent noise space.

    Key Components:
    - Q^A(s,a): Action-space critic (learns values of real actions)
    - Q^W(s,w): Noise-space critic (learns values of noise inputs)
    - π^W(s): Steering policy (learns which noise to use)
    - π_dp^W: Frozen diffusion policy (not updated!)

    Training follows Algorithm 1 from DSRL paper:
    1. Update Q^A with TD learning on (s,a,r,s')
    2. Update Q^W by distilling from Q^A
    3. Update π^W to maximize Q^W
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    diffusion_policy: BCDiffusionAgent = nonpytree_field()  # FROZEN - never updated!

    def forward_critic_A(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for action-space critic Q^A(s,a)."""
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic_A",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic_A(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """Forward pass for target action-space critic."""
        return self.forward_critic_A(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_critic_W(
        self,
        observations: Data,
        noise: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for noise-space critic Q^W(s,w)."""
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            noise,
            name="critic_W",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_steering_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """Forward pass for steering policy π^W(s)."""
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def _diffusion_policy_with_noise(
        self,
        observations: Data,
        noise: jax.Array,
    ) -> jax.Array:
        """
        Apply frozen diffusion policy with specified noise.

        This is the key operation: π_dp^W(s, w)
        Takes noise w and produces action a.

        Args:
            observations: State observations
            noise: Latent noise vectors (batch_size, noise_dim)

        Returns:
            Actions from diffusion policy (batch_size, action_dim)
        """
        # IMPORTANT: The diffusion policy is frozen (jax.lax.stop_gradient)
        # We use its sample_actions method with specified noise
        # This requires modifying BCDiffusionAgent.sample_actions to accept pre-specified noise
        return self.diffusion_policy.sample_actions_from_noise(
            observations=observations,
            noise=noise,
        )

    def critic_A_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Loss function for action-space critic Q^A.

        Standard TD learning:
        Loss = E[(Q^A(s,a) - (r + γ Q^A_target(s', a')))^2]

        where a' = π_dp^W(s', π^W(s'))
        """
        batch_size = batch["rewards"].shape[0]
        discount = self.config.get("discount", 0.99)

        # Get current Q-values
        rng, critic_rng = jax.random.split(rng)
        predicted_qs = self.forward_critic_A(
            batch["observations"],
            batch["actions"],
            rng=critic_rng,
            grad_params=params,
        )

        # Compute target Q-values
        # 1. Get steering noise for next state: w' = π^W(s')
        rng, steering_rng, noise_sample_rng = jax.random.split(rng, 3)
        next_noise_dist = self.forward_steering_policy(
            batch["next_observations"],
            rng=steering_rng,
            train=False,
        )
        next_noise = next_noise_dist.sample(seed=noise_sample_rng)

        # 2. Get actions from diffusion: a' = π_dp^W(s', w')
        next_actions = self._diffusion_policy_with_noise(
            batch["next_observations"],
            next_noise,
        )

        # 3. Compute target: r + γ Q^A_target(s', a')
        rng, target_critic_rng = jax.random.split(rng)
        target_qs = self.forward_target_critic_A(
            batch["next_observations"],
            next_actions,
            rng=target_critic_rng,
        )

        # Handle ensembles by taking mean
        if len(predicted_qs.shape) == 2:
            # Ensemble: (num_critics, batch_size)
            predicted_qs = predicted_qs.mean(axis=0)
        if len(target_qs.shape) == 2:
            # Ensemble: (num_critics, batch_size)
            target_qs = target_qs.min(axis=0)  # Conservative: use min

        chex.assert_shape(predicted_qs, (batch_size,))
        chex.assert_shape(target_qs, (batch_size,))

        # TD target
        td_target = batch["rewards"] + discount * batch["masks"] * target_qs
        td_target = jax.lax.stop_gradient(td_target)

        # TD loss
        critic_loss = jnp.mean((predicted_qs - td_target) ** 2)

        info = {
            "critic_A_loss": critic_loss,
            "predicted_q_A": jnp.mean(predicted_qs),
            "target_q_A": jnp.mean(td_target),
        }

        return critic_loss, info

    def critic_W_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Loss function for noise-space critic Q^W.

        Distillation from Q^A:
        Loss = E[(Q^W(s,w) - Q^A(s, π_dp^W(s,w)))^2]

        where w ~ N(0,I) (random noise sampling for noise aliasing)
        """
        batch_size = batch["observations"][self.config["image_keys"][0]].shape[0]

        # Sample random noise from standard Gaussian
        rng, noise_rng = jax.random.split(rng)
        noise_dim = self.config["noise_dim"]
        random_noise = jax.random.normal(noise_rng, (batch_size, noise_dim))

        # Get actions from diffusion with random noise: a = π_dp^W(s, w)
        actions_from_noise = self._diffusion_policy_with_noise(
            batch["observations"],
            random_noise,
        )

        # Get Q^A(s, a) as target
        rng, critic_A_rng = jax.random.split(rng)
        q_A_values = self.forward_critic_A(
            batch["observations"],
            actions_from_noise,
            rng=critic_A_rng,
            train=False,
        )

        # Handle ensembles
        if len(q_A_values.shape) == 2:
            # Take mean across ensemble
            q_A_values = q_A_values.mean(axis=0)

        # Get Q^W(s, w) prediction
        rng, critic_W_rng = jax.random.split(rng)
        q_W_values = self.forward_critic_W(
            batch["observations"],
            random_noise,
            rng=critic_W_rng,
            grad_params=params,
        )

        # Handle ensembles
        if len(q_W_values.shape) == 2:
            q_W_values = q_W_values.mean(axis=0)

        chex.assert_shape(q_A_values, (batch_size,))
        chex.assert_shape(q_W_values, (batch_size,))

        # Distillation loss: match Q^W to Q^A
        q_A_values = jax.lax.stop_gradient(q_A_values)
        critic_W_loss = jnp.mean((q_W_values - q_A_values) ** 2)

        info = {
            "critic_W_loss": critic_W_loss,
            "q_W_values": jnp.mean(q_W_values),
            "q_A_values_for_W": jnp.mean(q_A_values),
        }

        return critic_W_loss, info

    def actor_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """
        Loss function for steering policy π^W.

        Policy gradient:
        Loss = -E[Q^W(s, π^W(s))]

        Maximize expected value of steered noise.
        """
        batch_size = batch["observations"][self.config["image_keys"][0]].shape[0]

        # Sample noise from steering policy: w = π^W(s)
        rng, steering_rng, sample_rng = jax.random.split(rng, 3)
        steering_dist = self.forward_steering_policy(
            batch["observations"],
            rng=steering_rng,
            grad_params=params,
        )
        steered_noise = steering_dist.sample(seed=sample_rng)

        # Evaluate Q^W(s, w)
        rng, critic_W_rng = jax.random.split(rng)
        q_W_values = self.forward_critic_W(
            batch["observations"],
            steered_noise,
            rng=critic_W_rng,
            train=False,
        )

        # Handle ensembles
        if len(q_W_values.shape) == 2:
            q_W_values = q_W_values.mean(axis=0)

        chex.assert_shape(q_W_values, (batch_size,))

        # Policy gradient: maximize Q^W
        actor_objective = q_W_values.mean()
        actor_loss = -actor_objective  # Negative for minimization

        info = {
            "actor_loss": actor_loss,
            "actor_objective": actor_objective,
        }

        return actor_loss, info

    def loss_fns(self, batch):
        """Get all loss functions for DSRL update."""
        return {
            "critic_A": partial(self.critic_A_loss_fn, batch),
            "critic_W": partial(self.critic_W_loss_fn, batch),
            "actor": partial(self.actor_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic_A", "critic_W"}
        ),
    ) -> Tuple["DSRLAgent", dict]:
        """
        DSRL update following Algorithm 1.

        Steps:
        1. Update Q^A with TD learning
        2. Update Q^W by distilling from Q^A
        3. Update π^W to maximize Q^W

        Args:
            batch: Training batch
            pmap_axis: Axis for parallel mapping
            networks_to_update: Which networks to update

        Returns:
            Updated agent and info dict
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        # Get loss functions
        loss_fns = self.loss_fns(batch)

        # Only compute gradients for specified networks
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid networks to update: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # Apply gradient updates
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network for Q^A
        if "critic_A" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    # Note: Not JIT-compiled because diffusion denoising loop isn't JIT-compatible
    # @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions using DSRL steering.

        Process:
        1. π^W(s) → w_steer (steering noise)
        2. π_dp^W(s, w_steer) → a (action from diffusion)

        Args:
            observations: State observations
            seed: Random seed
            argmax: Whether to use mode of steering policy

        Returns:
            Actions from steered diffusion policy
        """
        if seed is None:
            seed = self.state.rng

        # Get steering noise from π^W
        steering_dist = self.forward_steering_policy(
            observations, rng=seed, train=False
        )

        if argmax:
            steered_noise = steering_dist.mode()
        else:
            steered_noise = steering_dist.sample(seed=seed)

        # Apply to diffusion policy
        actions = self._diffusion_policy_with_noise(
            observations,
            steered_noise,
        )

        return actions

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Frozen diffusion policy
        diffusion_policy: BCDiffusionAgent,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        image_keys: tuple = ("image",),
        use_proprio: bool = False,
        # Network architectures
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        actor_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        # DSRL-specific
        noise_dim: Optional[int] = None,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        # Training
        discount: float = 0.99,
        soft_target_update_rate: float = 0.005,
        learning_rate: float = 3e-4,
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        Create a DSRL agent.

        Args:
            diffusion_policy: Pre-trained frozen diffusion policy
            noise_dim: Dimension of latent noise space
                      (if None, inferred from action_dim * action_horizon)
            ... (other args same as SAC)

        Returns:
            DSRLAgent instance
        """
        action_dim = actions.shape[-1]

        # Infer noise dimension from diffusion policy if not specified
        if noise_dim is None:
            action_horizon = diffusion_policy.config.get("action_horizon", 1)
            noise_dim = action_dim * action_horizon

        # Create encoder (same as SAC/BC)
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

        # Create networks
        critic_A_backbone = MLP(**critic_network_kwargs)
        critic_W_backbone = MLP(**critic_network_kwargs)
        actor_backbone = MLP(**actor_network_kwargs)

        # Ensemble critics if specified
        if critic_ensemble_size > 1:
            critic_A_def = ensemblize_dsrl_critics(
                ActionCritic,
                num_critics=critic_ensemble_size,
                encoder=encoder_def,
                network=critic_A_backbone,
            )
            critic_W_def = ensemblize_dsrl_critics(
                NoiseCritic,
                num_critics=critic_ensemble_size,
                encoder=encoder_def,
                network=critic_W_backbone,
            )
        else:
            critic_A_def = ActionCritic(
                encoder=encoder_def,
                network=critic_A_backbone,
            )
            critic_W_def = NoiseCritic(
                encoder=encoder_def,
                network=critic_W_backbone,
            )

        actor_def = SteeringActor(
            encoder=encoder_def,
            network=actor_backbone,
            noise_dim=noise_dim,
            state_dependent_std=True,
            tanh_squash_distribution=False,  # Noise space is unbounded
        )

        # Create module dict
        networks = {
            "critic_A": critic_A_def,
            "critic_W": critic_W_def,
            "actor": actor_def,
        }

        model_def = ModuleDict(networks)

        # Initialize parameters
        from serl_launcher.common.optimizers import make_optimizer

        # Create separate optimizers for each network (like SAC)
        txs = {
            "actor": make_optimizer(learning_rate=learning_rate),
            "critic_A": make_optimizer(learning_rate=learning_rate),
            "critic_W": make_optimizer(learning_rate=learning_rate),
        }

        rng, init_rng = jax.random.split(rng)

        # Dummy inputs for initialization
        # dummy_noise = jnp.zeros((1, noise_dim))
        dummy_noise = jnp.zeros((noise_dim,))  # Remove batch dimension

        params = model_def.init(
            init_rng,
            critic_A=[observations, actions],
            critic_W=[observations, dummy_noise],
            actor=[observations],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            noise_dim=noise_dim,
            action_dim=action_dim,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
        )

        # Freeze diffusion policy
        frozen_diffusion = jax.lax.stop_gradient(diffusion_policy)

        agent = cls(
            state=state,
            config=config,
            diffusion_policy=frozen_diffusion,
        )

        # Load pretrained encoder weights if using pretrained
        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
