# Traceback (most recent call last):
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 890, in <module>
#     app.run(main)
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/absl/app.py", line 316, in run
#     _run_main(main, args)
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/absl/app.py", line 261, in _run_main
#     sys.exit(main(argv))
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 857, in main
#     learner(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 630, in learner
#     agent, update_info = agent.update(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 457, in update
#     new_state, info = self.state.apply_loss_fns(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/common/common.py", line 204, in apply_loss_fns
#     grads_and_aux = jax.tree_util.tree_map(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/common/common.py", line 205, in <lambda>
#     lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng),
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 336, in policy_loss_fn
#     timestep = jnp.asarray(bc_batch["timestep"], dtype=jnp.float32) # [BATCH_SIZE]
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/flax/core/frozen_dict.py", line 69, in __getitem__
#     v = self._dict[key]
# KeyError: 'timestep'
# jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 890, in <module>
#     app.run(main)
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/absl/app.py", line 316, in run
#     _run_main(main, args)
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/absl/app.py", line 261, in _run_main
#     sys.exit(main(argv))
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 857, in main
#     learner(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/../../../train_steer.py", line 630, in learner
#     agent, update_info = agent.update(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 457, in update
#     new_state, info = self.state.apply_loss_fns(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/common/common.py", line 204, in apply_loss_fns
#     grads_and_aux = jax.tree_util.tree_map(
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/common/common.py", line 205, in <lambda>
#     lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng),
#   File "/home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py", line 336, in policy_loss_fn
#     timestep = jnp.asarray(bc_batch["timestep"], dtype=jnp.float32) # [BATCH_SIZE]
#   File "/home/daphne/miniconda3/envs/hilserl2/lib/python3.9/site-packages/flax/core/frozen_dict.py", line 69, in __getitem__
#     v = self._dict[key]
# KeyError: 'timestep'






# (hilserl2) daphne @ joeljang ➜  iclr_constrained git:(main) ✗  source run_cl_learner.sh
# WARNING:absl:Tensorflow library not found, tensorflow.io.gfile operations will use native shim calls. GCS paths (i.e. 'gs://...') cannot be accessed.
# WARNING: You have not setup the ZED cameras, and currently cannot use them
#  Using STEER (BC on interventions with optional decay + RL).
#    BC timestep decay: 0.0
# Saving videos!
# The ResNet-10 weights already exist at '/home/daphne/.serl/resnet10_params.pkl'.
# Loaded 5.418792M parameters from ResNet-10 pretrained on ImageNet-1K
# replaced conv_init in encoder_front
# replaced norm_init in encoder_front
# replaced ResNetBlock_0 in encoder_front
# replaced ResNetBlock_1 in encoder_front
# replaced ResNetBlock_2 in encoder_front
# replaced ResNetBlock_3 in encoder_front
# I1117 22:03:09.833108 136920117135168 checkpoints.py:1111] Restoring orbax checkpoint from /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/classifier_ckpt/checkpoint_150
# I1117 22:03:09.833309 136920117135168 abstract_checkpointer.py:35] orbax-checkpoint version: 0.6.4
# I1117 22:03:09.836092 136920117135168 checkpointer.py:237] Restoring checkpoint from /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/classifier_ckpt/checkpoint_150.
# W1117 22:03:09.962724 136920117135168 transform_utils.py:230] The transformations API will eventually be replaced by an upgraded design. The current API will not be removed until this point, but it will no longer be actively worked on.
# I1117 22:03:09.968992 136920117135168 checkpointer.py:240] Finished restoring checkpoint from /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/cube_reach3/iclr_constrained/classifier_ckpt/checkpoint_150.
# I1117 22:03:09.969104 136920117135168 utils.py:240] [process=0][thread=MainThread] Skipping global process sync, barrier name: Checkpointer:restore.checkpoint_150
# I1117 22:03:09.973938 136920117135168 _schedule.py:129] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
# I1117 22:03:09.974215 136920117135168 _schedule.py:129] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
# I1117 22:03:09.974323 136920117135168 _schedule.py:129] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
# I1117 22:03:09.974415 136920117135168 _schedule.py:129] A polynomial schedule was set with a non-positive `transition_steps` value; this results in a constant schedule with value `init_value`.
# The ResNet-10 weights already exist at '/home/daphne/.serl/resnet10_params.pkl'.
# Loaded 5.418792M parameters from ResNet-10 pretrained on ImageNet-1K
# replaced conv_init in pretrained_encoder
# replaced norm_init in pretrained_encoder
# replaced ResNetBlock_0 in pretrained_encoder
# replaced ResNetBlock_1 in pretrained_encoder
# replaced ResNetBlock_2 in pretrained_encoder
# replaced ResNetBlock_3 in pretrained_encoder
# entity: null
# exp_descriptor: cube_reach3
# experiment_id: cube_reach3_20251117_220314
# group: null
# project: hil-serl-cubereach2
# tag: cube_reach3
# unique_identifier: '20251117_220314'

# wandb: Currently logged in as: daphc (social-rl) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
# wandb: Tracking run with wandb version 0.19.11
# wandb: Run data is saved locally in /tmp/tmp3q8tt8sz/wandb/run-20251117_220314-cube_reach3_20251117_220314
# wandb: Run `wandb offline` to turn off syncing.
# wandb: Syncing run cube_reach3_20251117_220314
# wandb: ⭐️ View project at https://wandb.ai/social-rl/hil-serl-cubereach2
# wandb: 🚀 View run at https://wandb.ai/social-rl/hil-serl-cubereach2/runs/cube_reach3_20251117_220314
#  demo buffer size: 1178
#  demo count: 20
#  bc buffer size: 1178
#  starting learner loop
# Filling up replay buffer: 236it [01:01,  3.85it/s]
#  sent initial network to actor
# learner:   0%|                                                                | 0/1000000 [00:00<?, ?it/s] Doing BC update.
#  Doing BC update.
# > /home/daphne/Desktop/jax-hitl-hil-serl/serl_launcher/serl_launcher/agents/continuous/sac_hybrid_single.py(386)policy_loss_fn()
# -> timestep = jnp.asarray(bc_batch["timestep"], dtype=jnp.float32) # [BATCH_SIZE]
# (Pdb) p bc_batch.keys()
# frozen_dict_keys(['actions', 'dones', 'grasp_penalty', 'masks', 'next_observations', 'observations', 'rewards'])
# (Pdb)




from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize, AlphaNetwork
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_cyan(x):
    return print("\033[96m {}\033[00m".format(x))


class SACAgentHybridSingleArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)

    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for single arm setups.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_optimism_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for the Q-OIL optimism critic ensemble (continuous actions only).
        Same architecture as the TD critic but with independent parameters. No target
        network exists for this critic; its Bellman target bootstraps off the TD
        critic's target_params so the intervention bonus does not propagate backward.
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="optimism_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_optimism_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for the Q-OIL optimism grasp critic (discrete gripper DQN).
        Same architecture as the TD grasp critic but with independent parameters.
        No target network; its Bellman target bootstraps off the TD grasp critic's
        target_params and online argmax (Double-DQN) so the bonus stays localized.
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="optimism_grasp_critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, rng=rng, grad_params=self.state.target_params
        )
    def forward_policy( # type: ignore
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )

        next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def optimism_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Q-OIL optimism critic loss.

        Same shape as the TD critic loss EXCEPT:
        - Predicted Q comes from the online optimism critic (`forward_optimism_critic`).
        - Bellman target bootstraps off the TD critic's target_params (NOT off the
          optimism critic itself). This is what keeps the bonus localized to actual
          intervention transitions and prevents leakage to predecessor states.
        - Target adds a per-transition bonus = bonus_frac * reward_scale * is_intervention.
        """
        batch_size = batch["rewards"].shape[0]
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q_td = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q_td = target_q_td - temperature * next_actions_log_probs

        bonus_abs = jnp.array(
            self.config.get("bonus_frac", 0.025)
            * self.config.get("reward_scale", 1.0),
            dtype=jnp.float32,
        )
        is_intervention = jnp.asarray(batch["is_intervention"], dtype=jnp.float32)
        chex.assert_shape(is_intervention, (batch_size,))
        target_q_opt = target_q_td + bonus_abs * is_intervention
        chex.assert_shape(target_q_opt, (batch_size,))

        predicted_qs = self.forward_optimism_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )
        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q_opt[None].repeat(
            self.config["critic_ensemble_size"], axis=0
        )
        chex.assert_equal_shape([predicted_qs, target_qs])
        optimism_critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "optimism_critic_loss": optimism_critic_loss,
            "predicted_opt_qs": jnp.mean(predicted_qs),
            "target_opt_qs": jnp.mean(target_qs),
            "bonus_abs": bonus_abs,
            "is_intervention_frac": is_intervention.mean(),
        }

        return optimism_critic_loss, info


    def grasp_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action = (batch["actions"][..., -1]).astype(jnp.int16) + 1 # Cast env action from [-1, 1] to {0, 1, 2}

         # Evaluate next grasp Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Select target next grasp Q based on the gripper action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1)
        chex.assert_shape(best_next_grasp_action, (batch_size,))

        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))

        # Forward pass through the online grasp critic to get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"],
            rng=rng,
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))

        # Select the predicted Q-values for the taken grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))

        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)

        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
        }

        return grasp_critic_loss, info

    def optimism_grasp_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Q-OIL optimism grasp critic loss (gripper DQN).

        Mirrors `grasp_critic_loss_fn` (Double-DQN), EXCEPT:
        - Bellman target bootstraps off the TD grasp critic's target_params and uses
          the TD online grasp critic for argmax. The optimism grasp critic NEVER
          bootstraps off itself - this keeps the bonus localized to actual
          intervention transitions.
        - Predicted Q comes from the online optimism grasp critic via params.
        - Target adds bonus_abs * is_intervention.
        """
        batch_size = batch["rewards"].shape[0]
        grasp_action = (batch["actions"][..., -1]).astype(jnp.int16) + 1

        # Target eval from TD grasp critic's slow copy.
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Online argmax from TD grasp critic (Double-DQN, TD side).
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            rng=rng,
        )
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1)
        chex.assert_shape(best_next_grasp_action, (batch_size,))

        target_next_grasp_q = target_next_grasp_qs[
            jnp.arange(batch_size), best_next_grasp_action
        ]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q_td = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )

        bonus_abs = jnp.array(
            self.config.get("bonus_frac", 0.025)
            * self.config.get("reward_scale", 1.0),
            dtype=jnp.float32,
        )
        is_intervention = jnp.asarray(batch["is_intervention"], dtype=jnp.float32)
        chex.assert_shape(is_intervention, (batch_size,))
        target_grasp_q_opt = target_grasp_q_td + bonus_abs * is_intervention
        chex.assert_shape(target_grasp_q_opt, (batch_size,))

        predicted_grasp_qs = self.forward_optimism_grasp_critic(
            batch["observations"],
            rng=rng,
            grad_params=params,
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))

        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))

        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q_opt])
        optimism_grasp_critic_loss = jnp.mean(
            (predicted_grasp_q - target_grasp_q_opt) ** 2
        )

        info = {
            "optimism_grasp_critic_loss": optimism_grasp_critic_loss,
            "predicted_opt_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_opt_grasp_qs": jnp.mean(target_grasp_q_opt),
        }

        return optimism_grasp_critic_loss, info

    def policy_loss_fn(self, batch, bc_batch, params: Params, rng: PRNGKey, current_step: Optional[int] = None):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        # Q-OIL: policy extraction switches from the TD critic to the optimism critic
        # so the optimistic signal biases exploration toward intervention regions.
        if self.config.get("use_optimism_critic", False):
            predicted_qs = self.forward_optimism_critic(
                batch["observations"],
                actions,
                rng=critic_rng,
            )
        else:
            predicted_qs = self.forward_critic(
                batch["observations"],
                actions,
                rng=critic_rng,
            )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        if self.config.get('use_bc_loss', False):
            assert bc_batch is not None
            # breakpoint()
            N = bc_batch["rewards"].shape[0]

            rng, bc_rng, bc_grasp_rng = jax.random.split(rng, 3)
            o_pre = bc_batch["observations"]
            a_exp = bc_batch["actions"]

            timestep = jnp.asarray(bc_batch["timestep"], dtype=jnp.float32) # [BATCH_SIZE]
            decay_coeff = jnp.array(self.config.get("bc_timestep_decay", 0.0), dtype=jnp.float32)
            decay_base = 1.0 - decay_coeff # scalar

            # Compute age: how many steps since the transition was collected
            # If current_step is not provided, fall back to old behavior (using timestep directly)
            ## adding asserts to make sure its always using current_step
            # print_green(timestep)
            # print_green(f"current_step: {current_step}")
            if current_step is not None:
                current_step_float = jnp.array(current_step, dtype=jnp.float32)
                age = current_step_float - timestep  # [BATCH_SIZE]
                # Clamp age to be non-negative (in case of data inconsistencies)
                age = jnp.maximum(age, 0.0)
                bc_weights = jnp.power(decay_base, age) # [BATCH_SIZE]
            else:
                # Fallback: use timestep directly (old behavior, for backward compatibility)
                # bc_weights = jnp.power(decay_base, timestep) # [BATCH_SIZE]
                raise ValueError("current_step must be provided when using BC loss.")
            # jax.debug.print(bc_weights)
            bc_loss_coeff = jnp.array(
                self.config.get("bc_loss_coeff", 1.0), dtype=jnp.float32
            )
            dist = self.forward_policy(o_pre, rng=bc_rng, grad_params=params)
            target_actions = a_exp[:,:-1]
            target_actions = jnp.clip(target_actions, -0.999, 0.999)  # Ensure actions are within valid range for tanh-squashed distribution
            bc_loss = (-dist.log_prob(target_actions) * bc_weights).mean()
            actor_loss += bc_loss_coeff * bc_loss

            beta = 0.1
            a_exp_grasp = jnp.array(jnp.round(a_exp[:,-1] + 1), dtype=jnp.int32)
            # chex.assert_shape(a_exp_grasp, (N,))
            # Q-OIL: BC trains the optimism grasp critic (the implicit policy used
            # at inference for the gripper) instead of the TD grasp critic.
            if self.config.get("use_optimism_critic", False):
                grasp_qs = self.forward_optimism_grasp_critic(o_pre, rng=bc_grasp_rng, grad_params=params)
            else:
                grasp_qs = self.forward_grasp_critic(o_pre, rng=bc_grasp_rng, grad_params=params)
            # chex.assert_shape(grasp_qs, (N, 3))
            grasp_logprobs = jax.nn.log_softmax(grasp_qs / beta, axis=1)
            # chex.assert_shape(grasp_logprobs, (N, 3))
            bc_grasp_loss = -grasp_logprobs[jnp.arange(N), a_exp_grasp]
            # chex.assert_shape(bc_grasp_loss, (N,))
            # Apply timestep decay weights to grasp loss as well
            bc_grasp_loss = (bc_grasp_loss * bc_weights).mean()

            actor_loss += bc_loss_coeff * bc_grasp_loss

            info = info | {
                "bc_loss": bc_loss,
                "bc_grasp_loss": bc_grasp_loss,
                "bc_loss_total": bc_loss + bc_grasp_loss,
                "bc_weights_mean": bc_weights.mean(),
                "bc_loss_coeff": bc_loss_coeff,
            }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}

    def loss_fns(self, batch, bc_batch = None, current_step: Optional[int] = None):
        if self.config.get("use_bc_loss", False):
            assert bc_batch is not None
            print_green("Doing BC update.")

        loss_dict = {
            "critic": partial(self.critic_loss_fn, batch),
            "grasp_critic": partial(self.grasp_critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch, bc_batch, current_step=current_step),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

        if self.config.get("use_optimism_critic", False):
            loss_dict["optimism_critic"] = partial(
                self.optimism_critic_loss_fn, batch
            )
            loss_dict["optimism_grasp_critic"] = partial(
                self.optimism_grasp_critic_loss_fn, batch
            )

        return loss_dict

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        bc_batch = None,
        **kwargs
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        if len(self.config["image_keys"]) > 0 and self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        current_step = kwargs.pop("current_step", None)

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, bc_batch=bc_batch, current_step=current_step, **kwargs)

        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"{networks_to_update} not within {loss_fns.keys()}"

        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
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

    def loss_bc(self, bc_batch, params: Params, rng: PRNGKey):
        assert bc_batch is not None
        N = bc_batch["rewards"].shape[0]

        rng, bc_rng, bc_grasp_rng = jax.random.split(rng, 3)
        o_pre = bc_batch["observations"]
        a_exp = bc_batch["actions"]
        dist = self.forward_policy(o_pre, rng=bc_rng, grad_params=params)
        log_probs = dist.log_prob(jnp.clip(a_exp[:,:-1], -0.999, 0.999))
        # chex.assert_shape(log_probs, (N,))
        bc_loss = -log_probs.mean()

        info = {
            "bc_loss": bc_loss,
        }

        beta = 0.1
        a_exp_grasp = jnp.array(jnp.round(a_exp[:,-1] + 1), dtype=jnp.int32)
        # chex.assert_shape(a_exp_grasp, (N,))
        # Q-OIL: BC trains the optimism grasp critic (used at inference) when enabled.
        if self.config.get("use_optimism_critic", False):
            grasp_qs = self.forward_optimism_grasp_critic(o_pre, rng=bc_grasp_rng, grad_params=params)
        else:
            grasp_qs = self.forward_grasp_critic(o_pre, rng=bc_grasp_rng, grad_params=params)
        # chex.assert_shape(grasp_qs, (N, 3))
        grasp_logprobs = jax.nn.log_softmax(grasp_qs / beta, axis=1)
        # chex.assert_shape(grasp_logprobs, (N, 3))
        bc_grasp_loss = -grasp_logprobs[jnp.arange(N), a_exp_grasp]
        # chex.assert_shape(bc_grasp_loss, (N,))
        bc_grasp_loss = bc_grasp_loss.mean()

        bc_loss += bc_grasp_loss

        info = info | {
            "bc_grasp_loss": bc_grasp_loss,
            "bc_loss_total": bc_loss,
        }

        return bc_loss, info

    @jax.jit
    def update_bc(self, bc_batch, pmap_axis = None):
        loss_fn_keys = ["critic", "grasp_critic", "actor", "temperature"]
        if self.config.get("use_optimism_critic", False):
            loss_fn_keys.extend(["optimism_critic", "optimism_grasp_critic"])
        loss_fns = {k: lambda params, rng: (0.0, {}) for k in loss_fn_keys}
        loss_fns["actor"] = partial(self.loss_bc, bc_batch)
        new_state, info = self.state.apply_loss_fns(
            loss_fns=loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
        )
        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)

        seed, grasp_key = jax.random.split(seed, 2)
        # Q-OIL: gripper "policy" is argmax of the optimism grasp critic
        # (the optimistic signal biases exploration toward intervention regions).
        if self.config.get("use_optimism_critic", False):
            grasp_q_values = self.forward_optimism_grasp_critic(observations, rng=grasp_key, train=False)
        else:
            grasp_q_values = self.forward_grasp_critic(observations, rng=grasp_key, train=False)

        # Select grasp actions based on the grasp Q-values
        grasp_action = grasp_q_values.argmax(axis=-1)
        grasp_action = grasp_action - 1 # Mapping back to {-1, 0, 1}

        return jnp.concatenate([ee_actions, grasp_action[..., None]], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        optimism_critic_def: Optional[nn.Module] = None,
        optimism_grasp_critic_def: Optional[nn.Module] = None,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        # Q-OIL knobs (only consumed when optimism_critic_def is not None)
        use_optimism_critic: bool = False,
        bonus_frac: float = 0.025,
        bc_loss_coeff: float = 1.0,
        reward_scale: float = 1.0,
        **kwargs,
    ):
        if use_optimism_critic:
            assert optimism_critic_def is not None, (
                "use_optimism_critic=True requires passing optimism_critic_def."
            )
            assert optimism_grasp_critic_def is not None, (
                "use_optimism_critic=True requires passing optimism_grasp_critic_def."
            )

        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }
        if use_optimism_critic:
            networks["optimism_critic"] = optimism_critic_def
            networks["optimism_grasp_critic"] = optimism_grasp_critic_def

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }
        if use_optimism_critic:
            txs["optimism_critic"] = make_optimizer(**critic_optimizer_kwargs)
            txs["optimism_grasp_critic"] = make_optimizer(**grasp_critic_optimizer_kwargs)

        rng, init_rng = jax.random.split(rng)

        # Initialize model parameters
        init_dict = {
            "actor": [observations],
            "critic": [observations, actions[..., :-1]],
            "grasp_critic": [observations],
            "temperature": [],
        }
        if use_optimism_critic:
            init_dict["optimism_critic"] = [observations, actions[..., :-1]]
            init_dict["optimism_grasp_critic"] = [observations]

        params = model_def.init(init_rng, **init_dict)["params"]

        rng, create_rng = jax.random.split(rng)
        # NOTE: target_params will include an `optimism_critic` subtree that is
        # polyak-updated alongside the TD critic (wasted compute) but is NEVER read.
        # The optimism critic loss bootstraps off the TD `target_critic`, not its own
        # target — this is the key design choice that keeps the bonus localized.
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        # Prepare configuration dictionary
        config_dict = dict(
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            target_entropy=target_entropy,
            backup_entropy=backup_entropy,
            image_keys=image_keys,
            reward_bias=reward_bias,
            augmentation_function=augmentation_function,
            use_optimism_critic=use_optimism_critic,
            bonus_frac=bonus_frac,
            bc_loss_coeff=bc_loss_coeff,
            reward_scale=reward_scale,
            **kwargs,
        )

        return cls(
            state=state,
            config=config_dict,
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        has_image: bool = True,
        bc_timestep_decay: float = 0.0,
        use_optimism_critic: bool = False,
        bonus_frac: float = 0.025,
        bc_loss_coeff: float = 1.0,
        reward_scale: float = 1.0,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

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

        if not has_image:
            encoder_def = None

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        if use_optimism_critic:
            # Same architecture, independent ensemble parameters. Shares the encoder
            # with the TD critic (matches the actor/critic encoder sharing pattern).
            optimism_critic_backbone = partial(MLP, **critic_network_kwargs)
            optimism_critic_backbone = ensemblize(
                optimism_critic_backbone, critic_ensemble_size
            )(name="optimism_critic_ensemble")
            optimism_critic_def = partial(
                Critic, encoder=encoders["critic"], network=optimism_critic_backbone
            )(name="optimism_critic")
        else:
            optimism_critic_def = None

        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")

        if use_optimism_critic:
            optimism_grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
            optimism_grasp_critic_def = partial(
                GraspCritic,
                encoder=encoders["grasp_critic"],
                network=optimism_grasp_critic_backbone,
            )(name="optimism_grasp_critic")
        else:
            optimism_grasp_critic_def = None

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            optimism_critic_def=optimism_critic_def,
            optimism_grasp_critic_def=optimism_grasp_critic_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            bc_timestep_decay=bc_timestep_decay,
            use_optimism_critic=use_optimism_critic,
            bonus_frac=bonus_frac,
            bc_loss_coeff=bc_loss_coeff,
            reward_scale=reward_scale,
            **kwargs,
        )

        if has_image and "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
