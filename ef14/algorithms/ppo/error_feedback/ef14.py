import functools
from typing import Literal, NamedTuple, Tuple, TypedDict

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics
from brax.training.types import Params, PRNGKey

from ef14.algorithms.ppo import _PMAP_AXIS_NAME, Metrics, TrainingState


class CompressionSpec(TypedDict):
    method: Literal["top", "random"]
    k: float


class State(NamedTuple):
    e_k: jax.Array
    w_k: jax.Array


def compress(
    copmression_spec: CompressionSpec, rng: jax.Array, params: Params
) -> Params:
    k = int(copmression_spec["k"] * len(params))
    if copmression_spec["method"] == "top":
        magnitudes = jnp.linalg.norm(params)
        values, ids = jax.lax.top_k(magnitudes, k)
    elif copmression_spec["method"] == "random":
        ids = jax.random.choice(rng, params.shape[0], shape=(k,), replace=False)
        values = params[ids]
    else:
        raise NotImplementedError("Compression method not implemented")
    outs = jnp.zeros_like(values)
    outs = outs.at[ids].set(values)
    return outs


def update_fn(
    loss_fn,
    optimizer,
    env,
    unroll_length,
    num_minibatches,
    make_policy,
    num_updates_per_batch,
    batch_size,
    num_envs,
    env_step_per_training_step,
    safe,
    *,
    num_trajectories_per_env,
    worker_compression: CompressionSpec,
    server_compression: CompressionSpec,
):
    def worker_step(
        data: types.Transition,
        params,
        e_i_k,
        key,
        normalizer_params,
    ):
        key, key_loss, key_compress = jax.random.split(key, 3)
        loss_and_pgrad_fn = gradients.loss_and_pgrad(
            loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )
        (_, aux), h_i_k = loss_and_pgrad_fn(params, normalizer_params, data, key_loss)
        x_i_k, pytree_def = jax.flatten_util.ravel_pytree(params)
        h_i_k, _ = jax.flatten_util.ravel_pytree(h_i_k)
        v_i_k = compress(worker_compression, key_compress, x_i_k + h_i_k)
        v_i_k = pytree_def(v_i_k)
        e_i_k = e_i_k + h_i_k - v_i_k
        return (v_i_k, e_i_k), aux

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, ef14_state, key = carry
        e_k, w_k = ef14_state
        key, compress_key = jax.random.split(key)
        step = lambda data, e_k: worker_step(data, params, e_k, key, normalizer_params)
        (v_k, e_k), aux = jax.vmap(step)(data, e_k)
        v_k = jax.tree.map(lambda x: x.mean(0), v_k)
        w_k_updates, optimizer_state = optimizer.update(v_k, optimizer_state)
        w_k = optax.apply_updates(w_k, w_k_updates)
        delta = jax.tree.map(lambda w, x: w - x, w_k, params)
        delta, pytree_def = jax.flatten_util.ravel_pytree(delta)
        tmp_server_compress = compress(server_compression, compress_key, delta)
        delta = pytree_def(delta)
        params = jax.tree.map(lambda x, d: x + d, params, tmp_server_compress)
        return (optimizer_state, params, e_k, key), aux

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, ef14_state, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, ef14_state, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, ef14_state, key), aux

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        extra_fields = ("truncation",)
        if safe:
            extra_fields += ("cost", "cumulative_cost")  # type: ignore

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            generate_unroll = lambda state: acting.generate_unroll(
                env,
                state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            generate_unroll = jax.vmap(generate_unroll)
            next_state, data = generate_unroll(current_state)
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_trajectories_per_env,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (x.shape[3], -1, x.shape[1]) + x.shape[4:]), data
        )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, error_feedback_state, _), aux = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (
                training_state.optimizer_state,
                training_state.params,
                training_state.error_feedback_state,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            penalizer_params=aux.pop("penalizer_params", None),
            env_steps=training_state.env_steps + env_step_per_training_step,
            error_feedback_state=error_feedback_state,
        )  # type: ignore
        return (new_training_state, state, new_key), aux

    def init(ppo_params):
        return State(jax.tree.map(lambda x: jnp.zeros_like(x), ppo_params), ppo_params)

    return training_step, init
