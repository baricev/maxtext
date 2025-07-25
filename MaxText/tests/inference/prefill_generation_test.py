"""Tests for KVCache prefill, chunked prefill and generation."""

import jax
import jax.numpy as jnp
from jax import random

from MaxText.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText.inference import kvcache


def _transpose_cache(cache_array, cache_module):
    return jnp.transpose(cache_array, cache_module.key_axis_order)


def _get_test_params():
    rng = jax.random.PRNGKey(0)
    batch = 1
    prefill_len = 8
    target_len = 12
    num_heads = 2
    head_dim = 4
    dtype = jnp.bfloat16
    return rng, batch, prefill_len, target_len, num_heads, head_dim, dtype


def test_regular_prefill():
    rng, batch, prefill_len, target_len, num_heads, head_dim, dtype = _get_test_params()
    cache_mod = kvcache.KVCache(prefill_len, target_len, dtype)
    key = jnp.ones((batch, prefill_len, num_heads, head_dim), dtype=dtype) * 0.1
    value = jnp.ones_like(key) * 0.2
    dec_seg = jnp.ones((batch, prefill_len), dtype=jnp.int32)

    variables = cache_mod.init({"params": rng}, key, value, dec_seg, MODEL_MODE_PREFILL)
    _, vars_after = cache_mod.apply(
        variables,
        key,
        value,
        dec_seg,
        MODEL_MODE_PREFILL,
        rngs={"params": random.PRNGKey(1)},
        mutable=True,
    )
    prefill_key = _transpose_cache(vars_after["cache"]["cached_prefill_key"].value, cache_mod)
    prefill_value = _transpose_cache(vars_after["cache"]["cached_prefill_value"].value, cache_mod)

    assert prefill_key.shape == (batch, prefill_len, num_heads, head_dim)
    assert prefill_value.shape == (batch, prefill_len, num_heads, head_dim)
    assert float(prefill_key[0, 0, 0, 0]) == 0.1
    assert float(prefill_value[0, 0, 0, 0]) == 0.2


def test_chunked_prefill():
    rng, batch, prefill_len, target_len, num_heads, head_dim, dtype = _get_test_params()
    cache_mod = kvcache.KVCache(prefill_len, target_len, dtype, use_chunked_prefill=True)
    chunk1 = 4
    key1 = jnp.ones((batch, chunk1, num_heads, head_dim), dtype=dtype) * 0.3
    value1 = jnp.ones_like(key1) * 0.4
    seg1 = jnp.ones((batch, chunk1), dtype=jnp.int32)

    variables = cache_mod.init({"params": rng}, key1, value1, seg1, MODEL_MODE_PREFILL)
    _, vars_after = cache_mod.apply(
        variables,
        key1,
        value1,
        seg1,
        MODEL_MODE_PREFILL,
        rngs={"params": random.PRNGKey(2)},
        mutable=True,
        previous_chunk=None,
    )

    chunk2 = prefill_len - chunk1
    key2 = jnp.ones((batch, chunk2, num_heads, head_dim), dtype=dtype) * 0.5
    value2 = jnp.ones_like(key2) * 0.6
    seg2 = jnp.ones((batch, chunk2), dtype=jnp.int32) * 2
    prev_tokens = jnp.ones((batch, chunk1), dtype=jnp.int32)

    _, vars_after = cache_mod.apply(
        vars_after,
        key2,
        value2,
        seg2,
        MODEL_MODE_PREFILL,
        rngs={"params": random.PRNGKey(3)},
        mutable=True,
        previous_chunk=prev_tokens,
    )

    prefill_key = _transpose_cache(vars_after["cache"]["cached_prefill_key"].value, cache_mod)
    prefill_value = _transpose_cache(vars_after["cache"]["cached_prefill_value"].value, cache_mod)
    seg_ids = vars_after["cache"]["cache_prefill_segment_id"].value

    assert jnp.all(prefill_key[:, :chunk1] == key1)
    assert jnp.all(prefill_key[:, chunk1:chunk1 + chunk2] == key2)
    assert jnp.all(prefill_value[:, :chunk1] == value1)
    assert jnp.all(prefill_value[:, chunk1:chunk1 + chunk2] == value2)
    assert jnp.all(seg_ids[:, :chunk1] == seg1)
    assert jnp.all(seg_ids[:, chunk1:chunk1 + chunk2] == seg2)


def test_autoregressive_generation():
    rng, batch, prefill_len, target_len, num_heads, head_dim, dtype = _get_test_params()
    cache_mod = kvcache.KVCache(prefill_len, target_len, dtype)
    key_prefill = jnp.ones((batch, prefill_len, num_heads, head_dim), dtype=dtype) * 0.1
    value_prefill = jnp.ones_like(key_prefill) * 0.2
    seg_prefill = jnp.ones((batch, prefill_len), dtype=jnp.int32)

    variables = cache_mod.init({"params": rng}, key_prefill, value_prefill, seg_prefill, MODEL_MODE_PREFILL)
    _, vars_after = cache_mod.apply(
        variables,
        key_prefill,
        value_prefill,
        seg_prefill,
        MODEL_MODE_PREFILL,
        rngs={"params": random.PRNGKey(4)},
        mutable=True,
    )

    token_key = jnp.ones((batch, 1, num_heads, head_dim), dtype=dtype) * 0.7
    token_value = jnp.ones_like(token_key) * 0.8
    _, vars_after = cache_mod.apply(
        vars_after,
        token_key,
        token_value,
        None,
        MODEL_MODE_AUTOREGRESSIVE,
        rngs={"params": random.PRNGKey(5)},
        mutable=True,
    )

    ar_key = _transpose_cache(vars_after["cache"]["cached_ar_key"].value, cache_mod)
    ar_value = _transpose_cache(vars_after["cache"]["cached_ar_value"].value, cache_mod)
    index = vars_after["cache"]["cache_ar_index"].value

    assert ar_key.shape == (batch, target_len - prefill_len, num_heads, head_dim)
    assert jnp.all(ar_key[:, 0] == token_key.squeeze(axis=1))
    assert jnp.all(ar_value[:, 0] == token_value.squeeze(axis=1))
    assert int(index[0]) == 1
