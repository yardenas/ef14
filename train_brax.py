import functools
import logging
import os

import hydra
import jax
from brax.io import model
from omegaconf import OmegaConf

from safe_ef import benchmark_suites
from safe_ef.algorithms.penalizers import (
    CRPO,
    Lagrangian,
    LagrangianParams,
)
from safe_ef.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


def get_penalizer(cfg):
    if cfg.agent.penalizer.name == "lagrangian":
        penalizer = Lagrangian(cfg.agent.penalizer.multiplier_lr)
        init_lagrange_multiplier = jax.numpy.log(
            jax.numpy.exp(cfg.agent.penalizer.initial_lagrange_multiplier) - 1.0
        )
        penalizer_state = LagrangianParams(
            init_lagrange_multiplier,
            penalizer.optimizer.init(init_lagrange_multiplier),
        )
    elif cfg.agent.penalizer.name == "crpo":
        penalizer = CRPO(cfg.agent.penalizer.eta, cfg.agent.penalizer.cost_scale)
        penalizer_state = None
    else:
        raise ValueError(f"Unknown penalizer {cfg.agent.penalizer.name}")
    return penalizer, penalizer_state


def get_error_feedback(cfg):
    if cfg.agent.error_feedback.name == "centralized":
        import safe_ef.algorithms.ppo.error_feedback.centralized as centralized

        error_feedback = centralized.update_fn
    elif cfg.agent.error_feedback.name == "ef14":
        import safe_ef.algorithms.ppo.error_feedback.ef14 as ef14

        ef14_cfg = dict(cfg.agent.error_feedback)
        ef14_cfg.pop("name")
        error_feedback = functools.partial(
            ef14.update_fn,
            **ef14_cfg,
            num_trajectories_per_env=cfg.agent.num_trajectories_per_env,
        )
    elif cfg.agent.error_feedback.name == "ef21":
        import safe_ef.algorithms.ppo.error_feedback.ef21 as ef21

        ef21_cfg = dict(cfg.agent.error_feedback)
        ef21_cfg.pop("name")
        error_feedback = functools.partial(
            ef21.update_fn,
            **ef21_cfg,
            num_trajectories_per_env=cfg.agent.num_trajectories_per_env,
        )
    else:
        raise ValueError(f"Unknown error feedback {cfg.agent.error_feedback}")
    return error_feedback


def get_train_fn(cfg):
    if cfg.agent.name == "ppo":
        import jax.nn as jnn

        import safe_ef.algorithms.ppo.networks as ppo_networks
        import safe_ef.algorithms.ppo.train as ppo

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "render_episodes",
                "train_domain_randomization",
                "eval_domain_randomization",
                "render",
                "store_policy",
            ]
        }
        policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
        value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        error_feedback = get_error_feedback(cfg)
        agent_cfg.pop("penalizer")
        agent_cfg.pop("error_feedback")
        train_fn = functools.partial(
            ppo.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            restore_checkpoint_path=f"{get_state_path()}/ckpt",
            penalizer=penalizer,
            penalizer_params=penalizer_params,
            error_feedback_factory=error_feedback,
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


class Counter:
    def __init__(self):
        self.count = 0


def report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


@hydra.main(version_base=None, config_path="safe_ef/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_env, eval_env = benchmark_suites.make(cfg)
    train_fn = get_train_fn(cfg)
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=train_env,
            eval_env=eval_env,
            wrap_env=False,
            progress_fn=functools.partial(report, logger, steps),
        )
        if cfg.training.render:
            rng = jax.random.split(jax.random.PRNGKey(cfg.training.seed), 5)
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            logger.log_video(video, steps.count, "eval/video")
        if cfg.training.store_policy:
            path = get_state_path() + "/policy.pkl"
            model.save_params(get_state_path() + "/policy.pkl", params)
            logger.log_artifact(path, "model", "policy")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
