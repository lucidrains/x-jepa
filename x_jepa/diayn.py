import math

import torch
from torch import nn, is_tensor, cat
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat
from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

# Skill Latents container

class SkillLatents(Module):
    def __init__(
        self,
        num_skills,
        dim_skill,
        l2norm_skills = False,
        trainable = True
    ):
        super().__init__()
        self.num_skills = num_skills
        self.dim_skill = dim_skill
        self.l2norm_skills = l2norm_skills

        skills = torch.randn(num_skills, dim_skill)
        if l2norm_skills:
            skills = l2norm(skills)

        self.skills = nn.Parameter(skills, requires_grad = trainable)

    def forward(
        self,
        skill_id = None
    ):
        if not exists(skill_id):
            return None

        if not is_tensor(skill_id):
            assert 0 <= skill_id < self.num_skills, f'skill_id {skill_id} must be in range [0, {self.num_skills})'

        latents = self.skills[skill_id]
        return l2norm(latents) if self.l2norm_skills else latents

# Skill Discriminator (predicts skill index z from states/actions - DIAYN Eysenbach et al. 2018)

class SkillDiscriminator(Module):
    def __init__(
        self,
        dim_state,
        num_skills,
        dim_action = 0,
        dim = 64,
        depth = 2,
        chunk_len = 1,
        state_encoder = None
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.has_action = dim_action > 0
        self.num_skills = num_skills
        self.chunk_len = chunk_len
        self.has_chunking = chunk_len > 1

        self.state_encoder = state_encoder
        dim_in = chunk_len * (dim_state + dim_action)
        self.net = MLP(dim_in, *((dim,) * depth), num_skills)

    def compute_loss(
        self,
        states,
        skill_ids,
        actions = None
    ):
        logits = self(states, actions = actions)

        if logits.ndim == 3 and skill_ids.ndim == 1:
            skill_ids = repeat(skill_ids, 'b -> b t', t = logits.shape[1])

        logits = rearrange(logits, '... k -> (...) k')
        skill_ids = rearrange(skill_ids, '... -> (...)')
        return F.cross_entropy(logits, skill_ids)

    def compute_intrinsic_reward(
        self,
        states,
        skill_ids,
        actions = None
    ):
        """
        DIAYN intrinsic reward: r_z = log q_phi(z | s, a) - log p(z)
        where p(z) = 1 / num_skills (uniform prior over skills).
        """
        if states.ndim == 3 and skill_ids.ndim == 1:
            skill_ids = repeat(skill_ids, 'b -> b t', t = states.shape[1])

        target_shape = skill_ids.shape
        logits = self(states, actions = actions)

        flat_logits = rearrange(logits, '... k -> (...) k')
        flat_log_probs = flat_logits.log_softmax(dim = -1)

        flat_skill_ids = rearrange(skill_ids, '... -> (...) 1')
        flat_log_q = flat_log_probs.gather(-1, flat_skill_ids)

        log_q = flat_log_q.reshape(target_shape)
        log_p = -math.log(self.num_skills)

        return log_q - log_p

    def forward(
        self,
        states,
        actions = None
    ):
        if exists(self.state_encoder):
            states = self.state_encoder(states)

        if exists(actions) and states.ndim == actions.ndim and states.shape[-2] == (actions.shape[-2] + 1):
            states = states[..., :actions.shape[-2], :]

        if not exists(actions) and self.has_action:
            actions = states.new_zeros(*states.shape[:-1], self.dim_action)

        to_cat = [t for t in (states, actions) if exists(t)]
        behavior = cat(to_cat, dim = -1) if len(to_cat) > 1 else states

        if not self.has_chunking or behavior.ndim < 3 or not divisible_by(behavior.shape[-2], self.chunk_len):
            return self.net(behavior)

        behavior = rearrange(behavior, '... (n c) d -> ... n (c d)', c = self.chunk_len)
        return self.net(behavior)
