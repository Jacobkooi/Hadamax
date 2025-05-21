
import copy
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any
import flax


from flax import struct
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import wandb
import yaml
import argparse
from rad import rad
from networks import (NatureCNN, QNetwork, QNetwork_Impala)

import envpool
import gym
import numpy as np
from packaging import version
from functools import partial
from atari_scores import ATARI_SCORES  # Import the scores

@struct.dataclass
class LogEnvState:
    handle: jnp.array
    lives: jnp.array
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
assert is_legacy_gym, "Current version supports only gym 0.18.0"

class JaxLogEnvPoolWrapper(gym.Wrapper):

    def __init__(self, env, reset_info=True, async_mode=True):
        super(JaxLogEnvPoolWrapper, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.env_name = env.name
        self.env_random_score, self.env_human_score = ATARI_SCORES[self.env_name]
        # get if the env has lives
        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")
        self.reset_info = reset_info
        handle, recv, send, step = env.xla()
        self.init_handle = handle
        self.send_f = send
        self.recv_f = recv
        self.step_f = step

    def reset(self, **kwargs):
        observations = super(JaxLogEnvPoolWrapper, self).reset(**kwargs)

        env_state = LogEnvState(
            jnp.array(self.init_handle),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):

        new_handle, (observations, rewards, dones, infos) = self.step_f(state.handle, action )

        new_episode_return = state.episode_returns + infos["reward"]
        new_episode_length = state.episode_lengths + 1
        state = state.replace(
            handle=new_handle,
            episode_returns=(new_episode_return) * (1 - infos["terminated"]) * (1 - infos["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - infos["terminated"]) * (1 - infos["TimeLimit.truncated"]),
            returned_episode_returns=jnp.where( infos["terminated"] + infos["TimeLimit.truncated"],
                new_episode_return, state.returned_episode_returns, ),
            returned_episode_lengths=jnp.where( infos["terminated"] + infos["TimeLimit.truncated"],
                new_episode_length, state.returned_episode_lengths, ),)

        if self.reset_info:
            elapsed_steps = infos["elapsed_step"]
            terminated = infos["terminated"] + infos["TimeLimit.truncated"]
            infos = {}

        normalize_score = lambda x: (x - self.env_random_score) / (self.env_human_score - self.env_random_score)
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["normalized_returned_episode_returns"] = normalize_score(state.returned_episode_returns)
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["elapsed_step"] = elapsed_steps
        infos["returned_episode"] = terminated
        return (observations, state, rewards, dones, infos,)

def straight_through_round(x):
    """Straight-through estimator for rounding to 0 or 1."""
    # Round in forward pass, but pass-through the gradient in backward pass
    return x + jax.lax.stop_gradient(jnp.round(x) - x)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["NUM_UPDATES_DECAY"] = config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, \
        "NUM_MINIBATCHES must divide NUM_STEPS * NUM_ENVS"

    # Define the number of total environments
    total_envs = (config["NUM_ENVS"] + config["TEST_ENVS"]
                  if config.get("TEST_DURING_TRAINING", False)
                  else config["NUM_ENVS"])

    def make_env(config, num_envs):
        env_kwargs = {
            "episodic_life": config.get("episodic_life", True),
            "reward_clip": config.get("reward_clip", True),
            "repeat_action_probability": config.get("repeat_action_probability", 0.0),
            "frame_skip": config.get("frame_skip", 4),
            "noop_max": config.get("noop_max", 30),
        }

        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            **env_kwargs,
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    total_envs = (config["NUM_ENVS"] + config["TEST_ENVS"]
                  if config.get("TEST_DURING_TRAINING", False)
                  else config["NUM_ENVS"])

    env = make_env(config, total_envs)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng):

        original_seed = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        if config['ENCODER'] == "impala":
            network = QNetwork_Impala(
                action_dim=env.single_action_space.n,
                config=config
                # norm_input=config.get("NORM_INPUT", False),
            )
        else:
            network = QNetwork(
                action_dim=env.single_action_space.n,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
                config=config,
            )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            network_variables = network.init(rng, init_x, train=False)

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta)
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                last_q = jnp.max(q_vals[-1], axis=-1)
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            lambda_targets = _compute_targets(
                last_q, transitions.q_val, transitions.reward, transitions.done
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        # Augment the observations (B, C, H, W)
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree_map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
                infos = jax.tree_map(lambda x: x[:, : -config["TEST_ENVS"]], infos)
                infos.update({"test_" + k: v for k, v in test_infos.items()})

            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space.shape[
                    0
                ],  # first dimension of the observation space is number of stacked frames
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
            }

            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test_{k}": v.mean() for k, v in test_infos.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):

    # config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    if config["NUM_SEEDS"] > 1:
        raise NotImplementedError("Vmapped seeds not supported yet.")
    else:
        outs = jax.jit(make_train(config))(rng)
    print(f"Took {time.time()-t0} seconds to complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train PQN on Atari using JAX")

    # Basic algorithm and environment configuration
    parser.add_argument('--ALG_NAME', type=str, default="pqn", help="Algorithm name")
    parser.add_argument('--ENV_NAME', type=str, default="Seaquest-v5", help="Name of the Atari environment")
    parser.add_argument('--TOTAL_TIMESTEPS', type=float, default=1e7, help="Total timesteps for training")
    parser.add_argument('--TOTAL_TIMESTEPS_DECAY', type=float, default=1e7, help="Timesteps for decay functions (epsilon and lr)")
    parser.add_argument('--NUM_ENVS', type=int, default=128, help="Number of parallel environments")
    parser.add_argument('--NUM_STEPS', type=int, default=32, help="Steps per environment in each update")
    parser.add_argument('--NUM_EPOCHS', type=int, default=2, help="Number of epochs per update")
    parser.add_argument('--NUM_MINIBATCHES', type=int, default=32, help="Number of minibatches per epoch")
    parser.add_argument('--NORM_TYPE', type=str, choices=["layer_norm", "batch_norm", "none"], default="layer_norm", help="Normalization type for network")

    # Exploration and decay configuration
    parser.add_argument('--EPS_START', type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument('--EPS_FINISH', type=float, default=0.001, help="Final epsilon for exploration")
    parser.add_argument('--EPS_DECAY', type=float, default=0.1, help="Decay ratio for epsilon")
    parser.add_argument('--EPS_TEST', type=float, default=0.0, help="Epsilon for greedy test policy")

    # Learning rate, gradient clipping, and training settings
    parser.add_argument('--LR', type=float, default=0.00025, help="Learning rate")
    parser.add_argument('--OPT', type=str, default="radam", help="Optimizer to use")
    parser.add_argument('--LR_LINEAR_DECAY', action='store_true', help="Use linear decay for learning rate")
    parser.add_argument('--MAX_GRAD_NORM', type=float, default=10.0, help="Max gradient norm for clipping")
    parser.add_argument('--GAMMA', type=float, default=0.99, help="Discount factor for reward")
    parser.add_argument('--LAMBDA', type=float, default=0.65, help="Lambda for generalized advantage estimation")

    # Network architecture and activation
    parser.add_argument('--ENCODER', type=str, default="hadamax", help="Activation function")

    # Environment-specific kwargs
    parser.add_argument('--episodic_life', type=bool, default=True, help="Terminate episode when life is lost")
    parser.add_argument('--reward_clip', type=bool, default=True, help="Clip rewards to range [-1, 1]")
    parser.add_argument('--repeat_action_probability', type=float, default=0.0, help="Sticky action probability")
    parser.add_argument('--frame_skip', type=int, default=4, help="Number of frames to skip")
    parser.add_argument('--noop_max', type=int, default=30, help="Max number of no-ops on reset")

    # Evaluation configuration
    parser.add_argument('--TEST_DURING_TRAINING', type=bool, default=True, help="Run evaluation during training")
    parser.add_argument('--TEST_ENVS', type=int, default=8, help="Number of environments used for testing")

    # WandB and experiment tracking configuration
    parser.add_argument('--WANDB_MODE', type=str, default="online", help="Wandb mode (online, offline, disabled)")
    parser.add_argument('--PROJECT', type=str, default="purejaxql", help="Wandb project name")
    parser.add_argument('--ENTITY', type=str, default=None, help="Wandb entity/user")
    parser.add_argument('--HYP_TUNE', action='store_true', help="Run hyperparameter tuning")
    parser.add_argument('--WANDB_LOG_ALL_SEEDS', type=bool, default=False, help="Log all seeds in Wandb")

    # Random seed and other general parameters
    parser.add_argument('--SEED', type=int, default=0, help="Random seed")
    parser.add_argument('--NUM_SEEDS', type=int, default=1, help="Number of random seeds for multi-seed runs")
    parser.add_argument('--SAVE_PATH', type=str, default="./models", help="Path to save trained models and configurations")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = vars(args)  # Convert Namespace to dict
    print("Config:\n", yaml.dump(config))
    if config.get("HYP_TUNE", False):
        tune(config)
    else:
        single_run(config)

if __name__ == "__main__":
    main()
