from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, einsum
from torch_einops_utils import tree_map_tensor_to_device

from x_jepa.utils import exists, default, Experience

# helpers

def z_score(t, eps = 1e-8):
    std = t.std(unbiased = False)
    return (t - t.mean()) / (std + eps) if std >= eps else torch.zeros_like(t)

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
    entropy_coef = 0.01
):
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)

    returns = experience.returns[1] if isinstance(experience.returns, (tuple, list)) else experience.returns
    assert exists(returns), 'experience.returns must be provided for ppo_loss'
    returns = returns.to(device)

    log_probs, entropy, state_tokens, state_latents = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions
    )

    old_log_probs = default(experience.actor_log_probs, log_probs.detach()).to(device)

    values = rearrange(world_model.value_network(state_tokens, state_latents), 'b n 1 -> b n')
    advantages = returns - values.detach()
    advantages = z_score(advantages)

    ratios = (log_probs - old_log_probs.detach()).exp()
    surr1 = ratios * advantages
    surr2 = ratios.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -reduce(torch.min(surr1, surr2), '... ->', 'mean')
    value_loss = 0.5 * F.mse_loss(values, returns)
    entropy_loss = reduce(entropy, '... ->', 'mean')

    return policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

# target policy optimization (TPO) - proposed by Jean Kaddour https://arxiv.org/abs/2604.06159

def tpo_loss(
    world_model,
    experience: Experience,
    actor_module = 'reflexive',
    eta = 1.0,
    entropy_coef = 0.01
):
    device = next(world_model.parameters()).device
    states, actions = tree_map_tensor_to_device((experience.states, experience.actions), device)

    cum_rewards = experience.cumulative_rewards
    if not exists(cum_rewards) and exists(experience.rewards):
        cum_rewards = reduce(experience.rewards, 'b n -> b', 'sum')
    assert exists(cum_rewards), 'experience.cumulative_rewards or experience.rewards must be provided for tpo_loss'
    cum_rewards = cum_rewards.to(device)

    u = z_score(cum_rewards)

    log_probs, entropy, _, _ = get_actor_log_probs_and_entropy(
        world_model, actor_module, states, actions
    )

    log_scores = reduce(log_probs, 'b n -> b', 'sum') / states.shape[1]

    with torch.no_grad():
        log_q = F.log_softmax(log_scores + u / eta, dim = -1)

    log_p = F.log_softmax(log_scores, dim = -1)
    divergence_loss = -einsum(log_q.exp(), log_p, 'b, b ->')
    entropy_loss = reduce(entropy, '... ->', 'mean')

    return divergence_loss - entropy_coef * entropy_loss
