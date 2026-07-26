from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from torch_einops_utils import tree_map_tensor_to_device, lens_to_mask, maybe, masked_sum, z_score

from loguru import logger

from x_jepa.utils import exists, default, Experience, masked_mean

def coerce_experience(experience: Experience | dict) -> Experience:
    if isinstance(experience, Experience):
        return experience

    return Experience(
        states = experience['states'],
        actions = experience['actions'],
        actor_log_probs = experience.get('actor_log_probs'),
        rewards = experience.get('rewards'),
        terminated = experience.get('terminated'),
        truncated = experience.get('truncated'),
        infos = experience.get('infos'),
        episode_len = experience.get('episode_len', experience.get('episode_lens')),
        cumulative_rewards = experience.get('cumulative_rewards'),
        returns = experience.get('returns'),
        state_latents = experience.get('state_latents')
    )

def get_actor_log_probs_and_entropy(
    world_model,
    actor_module,
    states,
    actions,
    detach_base_state_encoder = False
):
    batch, seq_len = actions.shape[:2]
    actor = world_model.actors[actor_module]
    state_tokens, sensory_hiddens = world_model.get_actor_state_tokens(
        actor_module, states, detach_base_state_encoder = detach_base_state_encoder
    )
    state_latents = world_model.to_state_latent(state_tokens)

    wm_hiddens = None
    if actor.pass_world_model_hiddens:
        with torch.no_grad():
            wm_out = world_model(states, actions, return_loss = False)
            wm_hiddens = wm_out.get('hiddens')

    action_preds = actor.get_action_preds(
        state_latents = state_latents,
        state_tokens = state_tokens,
        sensory_layer_hiddens = sensory_hiddens,
        world_model_hiddens = wm_hiddens
    )

    action_preds = action_preds[:, :seq_len]

    log_probs, entropy = actor.get_log_probs_and_entropy(action_preds, actions)
    return log_probs, entropy, state_tokens, state_latents

# classic ppo

def ppo_loss(
    world_model,
    experience: Experience | dict,
    actor_module = 'reflexive',
    clip_eps = 0.2,
    value_loss_weight = 0.5,
    entropy_loss_weight = 0.01,
    masked_loss_average_mode = 'timestep',
    detach_base_state_encoder = False,
    return_breakdown = False
):
    experience = coerce_experience(experience)
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)
    batch, seq_len = actions.shape[:2]

    lens = maybe(tree_map_tensor_to_device)(experience.episode_len, device)
    mask = maybe(lens_to_mask)(lens, max_len = seq_len)

    discounts = None
    returns = experience.returns
    if isinstance(returns, (tuple, list)):
        discounts, returns = returns

    assert exists(returns), 'experience.returns must be provided for ppo_loss'
    returns = returns.to(device)

    log_probs, entropy, state_tokens, state_latents = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions, detach_base_state_encoder = detach_base_state_encoder
    )

    old_log_probs = default(experience.actor_log_probs, log_probs.detach()).to(device)
    old_log_probs = old_log_probs[:, :seq_len]

    val_state_tokens, _ = world_model.get_value_state_tokens(
        states, detach_base_state_encoder = detach_base_state_encoder
    )
    val_state_latents = world_model.to_state_latent(val_state_tokens)

    values = world_model.value_network(val_state_tokens, val_state_latents, discounts)
    values = values[:, :seq_len]
    returns = returns[:, :seq_len]

    advantages = returns - values.detach()
    advantages = z_score(advantages, mask = mask)

    ratios = (log_probs - old_log_probs.detach()).exp()
    surr1 = ratios * advantages
    surr2 = ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages

    masked_loss = partial(masked_mean, mask = mask, average_mode = masked_loss_average_mode)

    policy_loss = -masked_loss(torch.min(surr1, surr2))
    raw_value_loss = F.smooth_l1_loss(values, returns, reduction = 'none')
    value_loss = 0.5 * masked_loss(raw_value_loss)
    entropy_loss = masked_loss(entropy)

    total_loss = (
        policy_loss +
        value_loss * value_loss_weight -
        entropy_loss * entropy_loss_weight
    )

    if not return_breakdown:
        return total_loss

    return total_loss, (policy_loss, value_loss, entropy_loss)

# target policy optimization (TPO) - proposed by Jean Kaddour https://arxiv.org/abs/2604.06159

def tpo_loss(
    world_model,
    experience: Experience | dict,
    actor_module = 'reflexive',
    eta = 1.0,
    entropy_loss_weight = 0.01,
    masked_loss_average_mode = 'timestep',
    detach_base_state_encoder = False
):
    experience = coerce_experience(experience)
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)
    batch, seq_len = actions.shape[:2]

    lens = maybe(tree_map_tensor_to_device)(experience.episode_len, device)
    mask = maybe(lens_to_mask)(lens, max_len = seq_len)

    cum_rewards = experience.cumulative_rewards
    if not exists(cum_rewards) and exists(experience.rewards):
        reward_mask = maybe(lens_to_mask)(lens, max_len = experience.rewards.shape[1])
        cum_rewards = masked_sum(experience.rewards, mask = reward_mask, dim = 1)

    assert exists(cum_rewards), 'experience.cumulative_rewards or experience.rewards must be provided for tpo_loss'
    cum_rewards = cum_rewards.to(device)

    num_episodes = cum_rewards.numel()
    assert num_episodes > 1, 'tpo_loss requires a batch of more than one episode'

    if num_episodes < 8:
        logger.warning(f'tpo_loss received a batch size of {num_episodes}, which is less than recommended minimum of 8 episodes.')

    u = z_score(cum_rewards)

    log_probs, entropy, _, _ = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions, detach_base_state_encoder = detach_base_state_encoder
    )

    log_scores = masked_sum(log_probs, mask = mask, dim = 1)

    with torch.no_grad():
        log_q = F.log_softmax(log_scores + u / eta, dim = -1)

    log_p = F.log_softmax(log_scores, dim = -1)
    divergence_loss = -einsum(log_q.exp(), log_p, 'b, b ->')

    masked_loss = partial(masked_mean, mask = mask, average_mode = masked_loss_average_mode)
    entropy_loss = masked_loss(entropy)

    return divergence_loss - entropy_loss * entropy_loss_weight
