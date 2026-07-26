from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from torch_einops_utils import tree_map_tensor_to_device, lens_to_mask, maybe, masked_sum, z_score

from x_jepa.utils import exists, default, Experience, masked_mean

def get_actor_log_probs_and_entropy(world_model, actor_module, states, actions):
    actor = world_model.actors[actor_module]
    encoded_state, sensory_hiddens, _ = world_model.encode_states(world_model.state_encoder, states)
    state_tokens = encoded_state
    state_latents = world_model.to_state_latent(state_tokens)

    wm_hiddens = None
    if actor.pass_world_model_hiddens:
        wm_out = world_model(states, actions, return_loss = False)
        wm_hiddens = wm_out.get('hiddens')

    action_preds = actor.get_action_preds(
        state_latents = state_latents,
        state_tokens = state_tokens,
        sensory_layer_hiddens = sensory_hiddens,
        world_model_hiddens = wm_hiddens
    )

    log_probs, entropy = actor.get_log_probs_and_entropy(action_preds, actions)
    return log_probs, entropy, state_tokens, state_latents

# classic ppo

def ppo_loss(
    world_model,
    experience: Experience,
    actor_module = 'reflexive',
    clip_eps = 0.2,
    value_coef = 0.5,
    entropy_coef = 0.01,
    masked_loss_average_mode = 'timestep'
):
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)

    lens = maybe(tree_map_tensor_to_device)(experience.episode_len, device)
    mask = maybe(lens_to_mask)(lens, max_len = states.shape[1])

    discounts = None
    returns = experience.returns
    if isinstance(returns, (tuple, list)):
        discounts, returns = returns

    assert exists(returns), 'experience.returns must be provided for ppo_loss'
    returns = returns.to(device)

    log_probs, entropy, state_tokens, state_latents = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions
    )

    old_log_probs = default(experience.actor_log_probs, log_probs.detach()).to(device)

    values = world_model.value_network(state_tokens, state_latents, discounts)
    advantages = returns - values.detach()
    advantages = z_score(advantages, mask = mask)

    ratios = (log_probs - old_log_probs.detach()).exp()
    surr1 = ratios * advantages
    surr2 = ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages

    masked_loss = partial(masked_mean, mask = mask, average_mode = masked_loss_average_mode)

    policy_loss = -masked_loss(torch.min(surr1, surr2))
    raw_value_loss = F.mse_loss(values, returns, reduction = 'none')
    value_loss = 0.5 * masked_loss(raw_value_loss)
    entropy_loss = masked_loss(entropy)

    return policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

# target policy optimization (TPO) - proposed by Jean Kaddour https://arxiv.org/abs/2604.06159

def tpo_loss(
    world_model,
    experience: Experience,
    actor_module = 'reflexive',
    eta = 1.0,
    entropy_coef = 0.01,
    masked_loss_average_mode = 'timestep'
):
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)

    lens = maybe(tree_map_tensor_to_device)(experience.episode_len, device)
    mask = maybe(lens_to_mask)(lens, max_len = states.shape[1])

    cum_rewards = experience.cumulative_rewards
    if not exists(cum_rewards) and exists(experience.rewards):
        cum_rewards = masked_sum(experience.rewards, mask = mask, dim = 1)

    assert exists(cum_rewards), 'experience.cumulative_rewards or experience.rewards must be provided for tpo_loss'
    cum_rewards = cum_rewards.to(device)

    num_episodes = cum_rewards.numel()
    assert num_episodes > 1, 'tpo_loss requires a batch of more than one episode'

    u = z_score(cum_rewards)

    log_probs, entropy, _, _ = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions
    )

    log_scores = masked_sum(log_probs, mask = mask, dim = 1)

    with torch.no_grad():
        log_q = F.log_softmax(log_scores + u / eta, dim = -1)

    log_p = F.log_softmax(log_scores, dim = -1)
    divergence_loss = -einsum(log_q.exp(), log_p, 'b, b ->')

    masked_loss = partial(masked_mean, mask = mask, average_mode = masked_loss_average_mode)
    entropy_loss = masked_loss(entropy)

    return divergence_loss - entropy_coef * entropy_loss
