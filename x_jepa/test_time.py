from __future__ import annotations

import math
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn, is_tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from einops import rearrange

from x_jepa.utils import exists, default, filter_kwargs_for_fn

# helpers

def tree_map(fn, tree):
    if is_tensor(tree):
        return fn(tree)
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    return tree

def resolve_module_path(agent, path, world_model_index = 0):
    path = path.removesuffix('.weight')

    for candidate in (path, f'world_models.{world_model_index}.{path}'):
        try:
            agent.get_submodule(candidate)
            return candidate
        except AttributeError:
            continue

    raise AttributeError(f'could not resolve module path {path!r} on Agent or world_models.{world_model_index}')

def expand_linear_paths(agent, paths, world_model_index = 0, allow_empty = False):
    resolved_paths = []
    seen = set()

    for raw_path in paths:
        base = resolve_module_path(agent, raw_path, world_model_index = world_model_index)
        target = agent.get_submodule(base)
        found = False

        for sub_name, mod in target.named_modules():
            if not isinstance(mod, nn.Linear):
                continue

            full_path = f'{base}.{sub_name}' if sub_name else base
            found = True

            if full_path not in seen:
                seen.add(full_path)
                resolved_paths.append(full_path)

        assert allow_empty or found, f'lora path {raw_path!r} contains no nn.Linear layers'

    return tuple(resolved_paths)

def collect_params(agent, paths, world_model_index = 0, exclude_ids = None):
    exclude_ids = default(exclude_ids, set())
    params, seen = [], set()

    for path in paths:
        resolved = resolve_module_path(agent, path, world_model_index = world_model_index)
        for param in agent.get_submodule(resolved).parameters():
            pid = id(param)
            if pid in seen or pid in exclude_ids:
                continue
            seen.add(pid)
            params.append(param)

    return params

# non-invasive forward-hook LoRA on nn.Linear

class LoRAHook(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank = 8,
        alpha = None
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear), f'LoRAHook expects nn.Linear, got {type(linear)}'
        assert rank > 0, 'lora rank must be positive'

        alpha = default(alpha, float(2 * rank))
        eff_rank = min(rank, linear.in_features, linear.out_features)
        device, dtype = linear.weight.device, linear.weight.dtype

        self.rank = eff_rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(eff_rank, linear.in_features, device = device, dtype = dtype))
        self.lora_b = nn.Parameter(torch.zeros(linear.out_features, eff_rank, device = device, dtype = dtype))

        nn.init.kaiming_uniform_(self.lora_a, a = math.sqrt(5))
        self.handle = linear.register_forward_hook(self._hook)

    def _hook(self, _, args, output):
        x = args[0]
        delta = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.scale
        return output + delta.to(output.dtype)

    def remove(self):
        if exists(self.handle):
            self.handle.remove()
            self.handle = None

# module path selectors
#
# 1. `gradient_path_module_paths`: entire network along the latent prediction gradient path (default)
# 2. `adajepa_module_paths`: AdaJEPA's `predlast` (+ optional `enclast`) subset
# 3. `last_swiglu_ffn_lora_paths`: linear layers of the transition predictor's last SwiGLU FFN

def last_swiglu_ffn_lora_paths(
    agent,
    world_model_index = 0,
    use_ema_for_planning = False
) -> tuple[str, ...]:
    transition_attr = 'ema_state_transition_for_planning.ema_model' if use_ema_for_planning else 'state_transition_for_planning'
    base = f'world_models.{world_model_index}.{transition_attr}'
    transition = agent.get_submodule(base)

    last_swiglu_subpath = None
    for sub_name, mod in transition.named_modules():
        if any(type(child).__name__ == 'SwiGLUGate' for child in mod.children()):
            last_swiglu_subpath = sub_name

    assert exists(last_swiglu_subpath), f'{base} has no SwiGLU feedforward block to target'
    full_path = f'{base}.{last_swiglu_subpath}' if last_swiglu_subpath else base
    return expand_linear_paths(agent, (full_path,), world_model_index = world_model_index)

def gradient_path_module_paths(
    agent,
    world_model_index = 0,
    use_ema_for_planning = False
):
    world_model = agent.get_world_model(world_model_index)
    wm_prefix = f'world_models.{world_model_index}'

    model_attr = 'ema_model.ema_model' if use_ema_for_planning else 'model'
    transition_attr = 'ema_state_transition_for_planning.ema_model' if use_ema_for_planning else 'state_transition_for_planning'

    encoder_paths = ['state_encoder']

    if any(exists(v) for v in agent.view_embs):
        encoder_paths.append('view_embs')

    encoder_paths.extend([
        'state_linear_rnn',
        f'{wm_prefix}.{model_attr}',
        'to_state_latent'
    ])

    if agent.has_episodic_mem:
        encoder_paths.append('episodic_mems')

    predictor_paths = []

    if not agent.is_transition_action_space_raw:
        predictor_paths.append('action_encoder')

        if agent.transition_action_space == 'global':
            predictor_paths.append('action_linear_rnn')

        predictor_paths.append('to_action_latent')

    predictor_paths.append(f'{wm_prefix}.{transition_attr}')

    if world_model.has_orthogonal_subspaces:
        predictor_paths.append(f'{wm_prefix}.orthogonal_subspaces')

    return dict(
        predictor_paths = tuple(predictor_paths),
        encoder_paths = tuple(encoder_paths)
    )

def adajepa_module_paths(
    agent,
    world_model_index = 0,
    predictor_last_blocks = 1,
    adapt_encoder = False,
    use_ema_for_planning = False
):
    world_model = agent.get_world_model(world_model_index)
    transition_attr = 'ema_state_transition_for_planning.ema_model' if use_ema_for_planning else 'state_transition_for_planning'
    base = f'world_models.{world_model_index}.{transition_attr}'

    if world_model.transition_is_prefix:
        transition = agent.get_submodule(base)
        depth = transition.transformer.depth
        start = max(depth - predictor_last_blocks, 0)
        predictor_paths = [f'{base}.transformer.layers.{idx}' for idx in range(start, depth)]
        predictor_paths.append(f'{base}.to_pred')
    else:
        predictor_paths = [base]

    encoder_paths = []

    if adapt_encoder:
        for idx, encoder in enumerate(agent.state_encoder):
            last_linear = None

            for name, module in encoder.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear = name

            assert exists(last_linear), f'state_encoder.{idx} has no linear projection head to adapt'
            encoder_paths.append(f'state_encoder.{idx}.{last_linear}' if last_linear else f'state_encoder.{idx}')

    return dict(
        predictor_paths = tuple(predictor_paths),
        encoder_paths = tuple(encoder_paths)
    )

# test-time adapter: plan -> execute -> adapt -> replan

class TestTimeAdapter:
    __test__ = False

    def __init__(
        self,
        agent,
        predictor_paths: tuple[str, ...] | None = None,
        encoder_paths: tuple[str, ...] | None = None,
        *,
        use_lora = False,
        lora_last_swiglu_ffn = False,
        lora_paths: tuple[str, ...] | None = None,
        lora_rank = 8,
        lora_alpha = 16.,
        lr_lora = None,
        world_model_index = 0,
        use_ema_for_planning = False,
        lr_predictor = 5e-4,
        lr_encoder = 1e-5,
        weight_decay = 0.,
        buffer_size = 5,
        steps = 1,
        max_grad_norm = 1.0
    ):
        if isinstance(lora_paths, str):
            lora_paths = (lora_paths,)

        assert not (lora_last_swiglu_ffn and exists(lora_paths)), 'pass either lora_last_swiglu_ffn = True or explicit lora_paths, not both'

        if lora_last_swiglu_ffn:
            lora_paths = last_swiglu_ffn_lora_paths(
                agent,
                world_model_index = world_model_index,
                use_ema_for_planning = use_ema_for_planning
            )

        if exists(lora_paths):
            lora_paths = tuple(lora_paths)
            assert len(lora_paths) > 0, 'lora_paths must be non-empty when provided; omit lora_paths to default to the full gradient path'
            use_lora = True

        self.agent = agent
        self.world_model_index = world_model_index
        self.use_ema_for_planning = use_ema_for_planning
        self.use_lora = use_lora
        self.lora_last_swiglu_ffn = lora_last_swiglu_ffn
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr_lora = lr_lora
        self.lr_predictor = lr_predictor
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.steps = steps
        self.max_grad_norm = max_grad_norm

        world_model = agent.get_world_model(world_model_index)
        assert world_model.temporal_compression == 1, 'TestTimeAdapter currently requires temporal_compression = 1'

        # when paths are omitted, default to updating the entire network along the gradient path

        grad_paths = gradient_path_module_paths(
            agent,
            world_model_index = world_model_index,
            use_ema_for_planning = use_ema_for_planning
        )

        self.lora_hooks: dict[str, LoRAHook] = dict()
        self._encoder_lora_keys: set[str] = set()

        if use_lora:
            if exists(lora_paths):
                expanded = expand_linear_paths(agent, lora_paths, world_model_index = world_model_index)
            else:
                pred_linears = expand_linear_paths(agent, grad_paths['predictor_paths'], world_model_index = world_model_index, allow_empty = True)
                enc_linears = expand_linear_paths(agent, grad_paths['encoder_paths'], world_model_index = world_model_index, allow_empty = True)
                self._encoder_lora_keys = set(enc_linears) - set(pred_linears)
                expanded = tuple(dict.fromkeys((*pred_linears, *enc_linears)))

            self.lora_paths = expanded
            for path in expanded:
                self.lora_hooks[path] = LoRAHook(agent.get_submodule(path), rank = lora_rank, alpha = lora_alpha)

            self.predictor_paths = tuple(default(predictor_paths, tuple()))
            self.encoder_paths = tuple(default(encoder_paths, tuple()))
        else:
            self.lora_paths = tuple()
            if not exists(predictor_paths) and not exists(encoder_paths):
                self.predictor_paths = grad_paths['predictor_paths']
                self.encoder_paths = grad_paths['encoder_paths']
            else:
                self.predictor_paths = tuple(default(predictor_paths, tuple()))
                self.encoder_paths = tuple(default(encoder_paths, tuple()))

        # guard against online vs EMA planning mismatch

        all_checked_paths = tuple(
            resolve_module_path(agent, p, world_model_index = world_model_index)
            for p in (*self.predictor_paths, *self.encoder_paths, *self.lora_paths)
        )
        wm_prefix = f'world_models.{world_model_index}'

        if use_ema_for_planning:
            assert all(
                not p.startswith(f'{wm_prefix}.state_transition_for_planning') and not p.startswith(f'{wm_prefix}.model')
                for p in all_checked_paths
            ), 'use_ema_for_planning = True requires paths targeting ema_model / ema_state_transition_for_planning'
        else:
            assert all(
                'ema_state_transition_for_planning' not in p and not p.startswith(f'{wm_prefix}.ema_model') and not p.startswith('ema_state_encoder')
                for p in all_checked_paths
            ), 'paths target EMA modules, but use_ema_for_planning = False'

        self.params = None
        self.initial_params = None
        self.optimizer = None
        self.buffer = deque(maxlen = buffer_size)

        # stateful tracking for manual step-by-step planning (`adapter.plan` -> `adapter.update`)

        self._step = 0
        self._memories = None
        self._prev_memories = None
        self._memories_after_state = None
        self._last_state = None
        self._last_action = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False

    def __call__(
        self,
        step = None,
        state = None,
        action = None,
        next_state = None,
        prev_memories = None,
        memories = None
    ):
        return self.observe(
            step = step,
            state = state,
            action = action,
            next_state = next_state,
            prev_memories = prev_memories,
            memories = memories
        )

    def reset(self):
        if exists(self.params):
            with torch.no_grad():
                for param, initial in zip(self.params, self.initial_params):
                    param.copy_(initial)

            self.optimizer.state.clear()

        self.buffer.clear()
        self._step = 0
        self._memories = None
        self._prev_memories = None
        self._memories_after_state = None
        self._last_state = None
        self._last_action = None

    def remove(self):
        self.reset()

        for hook in self.lora_hooks.values():
            hook.remove()

        self.lora_hooks.clear()
        self.params = None
        self.initial_params = None
        self.optimizer = None

    def interact_with_environment(self, env, *, callback = None, **kwargs):
        def composed_callback(
            step = None,
            state = None,
            action = None,
            next_state = None,
            prev_memories = None,
            memories = None,
            reward = None,
            terminated = None,
            truncated = None,
            info = None,
            is_done = None,
            cumulative_rewards = None,
            episode_len = None,
            world_model = None
        ):
            ttt_loss = self.observe(
                step = step,
                state = state,
                action = action,
                next_state = next_state,
                prev_memories = prev_memories,
                memories = memories
            )

            if exists(callback):
                cb_kwargs = filter_kwargs_for_fn(
                    callback,
                    step = step,
                    state = state,
                    action = action,
                    next_state = next_state,
                    prev_memories = prev_memories,
                    memories = memories,
                    ttt_loss = ttt_loss,
                    reward = reward,
                    terminated = terminated,
                    truncated = truncated,
                    info = info,
                    is_done = is_done,
                    cumulative_rewards = cumulative_rewards,
                    episode_len = episode_len,
                    world_model = world_model
                )
                callback(**cb_kwargs)

        return self.agent.interact_with_environment(
            env,
            world_model_index = self.world_model_index,
            use_ema_for_planning = self.use_ema_for_planning,
            callback = composed_callback,
            **kwargs
        )

    def _to_batched_tensor(self, x):
        device = self.agent.device
        if isinstance(x, (list, tuple)):
            tensors = type(x)(torch.as_tensor(v, dtype = torch.float32, device = device) for v in x)
            unbatched = tensors[0].ndim == 1
            if unbatched:
                tensors = type(x)(rearrange(t, '... -> 1 ...') for t in tensors)
            return tensors, unbatched

        t = torch.as_tensor(x, dtype = torch.float32, device = device)
        unbatched = t.ndim == 1
        if unbatched:
            t = rearrange(t, '... -> 1 ...')
        return t, unbatched

    @torch.no_grad()
    def plan(self, state, **plan_kwargs):
        state_batched, unbatched = self._to_batched_tensor(state)
        first = state_batched[0] if isinstance(state_batched, (list, tuple)) else state_batched
        batch = first.shape[0]

        state_seq = tree_map(lambda t: rearrange(t, 'b ... -> b 1 ...'), state_batched)
        empty_actions = torch.empty(batch, 0, self.agent.dim_action, device = self.agent.device)

        self._prev_memories = self._memories
        self._last_state = state_batched

        plan_out, next_memories = self.agent.plan(
            states = state_seq,
            actions = empty_actions,
            memories = self._memories,
            return_memories = True,
            world_model_index = self.world_model_index,
            use_ema_for_planning = self.use_ema_for_planning,
            **plan_kwargs
        )

        planned_actions = plan_out[0] if isinstance(plan_out, tuple) else plan_out
        step_action = planned_actions[:, 0]

        self._last_action = step_action
        self._memories_after_state = next_memories

        return step_action[0] if unbatched else step_action

    def update(self, next_state, action = None):
        assert exists(self._last_state), 'adapter.plan(state, ...) must be called before adapter.update(next_state)'

        next_state_batched, _ = self._to_batched_tensor(next_state)
        act_batched = self._to_batched_tensor(action)[0] if exists(action) else self._last_action

        if exists(self._memories_after_state):
            self._memories = self.agent.update_memories_with_action(
                act_batched,
                self._memories_after_state,
                world_model_index = self.world_model_index,
                use_ema_for_planning = self.use_ema_for_planning
            )

        loss = self.observe(
            step = None,
            state = self._last_state,
            action = act_batched,
            next_state = next_state_batched,
            prev_memories = self._prev_memories,
            memories = self._memories
        )
        self._step += 1
        return loss

    def observe(
        self,
        step = None,
        state = None,
        action = None,
        next_state = None,
        prev_memories = None,
        memories = None
    ):
        assert exists(state) and exists(action) and exists(next_state), 'state, action, and next_state must be provided to TestTimeAdapter.observe'

        if exists(step) and step == 0:
            self.reset()

        self.ensure_optimizer()

        device = self.agent.device
        self.buffer.append(tuple(
            tree_map(lambda t: t.detach().to(device), x)
            for x in (state, action, next_state, prev_memories, memories)
        ))

        if self.steps <= 0:
            with torch.no_grad():
                return self.latent_prediction_loss().detach()

        selected_ids = {id(p) for p in self.params}
        disabled_params = []
        enabled_params = []

        for param in self.agent.parameters():
            if id(param) not in selected_ids and param.requires_grad:
                param.requires_grad_(False)
                disabled_params.append(param)

        for param in self.params:
            if not param.requires_grad:
                param.requires_grad_(True)
                enabled_params.append(param)

        loss = None
        try:
            with torch.enable_grad():
                for _ in range(self.steps):
                    loss = self.latent_prediction_loss()

                    if not loss.requires_grad:
                        raise RuntimeError(
                            'Selected adaptation parameters received no gradient from the latent prediction loss '
                            '(targeted layers are disconnected from the planning encoder/transition path).'
                        )

                    self.optimizer.zero_grad()
                    loss.backward()

                    if exists(self.max_grad_norm):
                        clip_grad_norm_(self.params, self.max_grad_norm)

                    self.optimizer.step()
        finally:
            for param in disabled_params:
                param.requires_grad_(True)
            for param in enabled_params:
                param.requires_grad_(False)

        return loss.detach() if exists(loss) else None

    def ensure_optimizer(self):
        if exists(self.optimizer):
            return

        predictor_params = collect_params(self.agent, self.predictor_paths, world_model_index = self.world_model_index)
        encoder_params = collect_params(
            self.agent,
            self.encoder_paths,
            world_model_index = self.world_model_index,
            exclude_ids = {id(p) for p in predictor_params}
        )

        pred_lora_params, enc_lora_params = [], []
        for path, hook in self.lora_hooks.items():
            target = enc_lora_params if (not exists(self.lr_lora) and path in self._encoder_lora_keys) else pred_lora_params
            target.extend([hook.lora_a, hook.lora_b])

        all_params = predictor_params + encoder_params + pred_lora_params + enc_lora_params
        assert len(all_params) > 0, 'at least one parameter must be selected for adaptation'

        param_groups = []
        if len(predictor_params) > 0:
            param_groups.append(dict(params = predictor_params, lr = self.lr_predictor))

        if len(encoder_params) > 0:
            param_groups.append(dict(params = encoder_params, lr = self.lr_encoder))

        if len(pred_lora_params) > 0:
            param_groups.append(dict(params = pred_lora_params, lr = default(self.lr_lora, self.lr_predictor)))

        if len(enc_lora_params) > 0:
            param_groups.append(dict(params = enc_lora_params, lr = self.lr_encoder))

        self.params = all_params
        self.initial_params = [param.detach().clone() for param in self.params]
        self.optimizer = Adam(param_groups, weight_decay = self.weight_decay)

    def encode_latents(self, states, memories = None, no_grad = False):
        world_model = self.agent.get_world_model(self.world_model_index)
        prompt_model = world_model.ema_model.ema_model if self.use_ema_for_planning else world_model.model
        device = self.agent.device

        states = tree_map(lambda t: rearrange(t.to(device), 'b ... -> b 1 ...'), states)
        first = states[0] if isinstance(states, (list, tuple)) else states
        batch = first.shape[0]

        state_rnn_memories = model_memories = None

        if exists(memories):
            state_rnn_memories, _, model_memories = memories

        context = torch.no_grad if no_grad else nullcontext

        with context():
            state_tokens, sensory_layer_hiddens, _ = self.agent.encode_states(self.agent.state_encoder, states)
            state_tokens = self.agent.state_linear_rnn(state_tokens, memories = state_rnn_memories)

            wm_past_layers = list(sensory_layer_hiddens) if self.agent.pass_sensory_hiddens_to_world_model else None
            prepend_memories = self.agent.episodic_mems(batch) if (self.agent.has_episodic_mem and not exists(model_memories)) else None

            state_embeds = prompt_model(
                state_tokens,
                past_layers = wm_past_layers,
                memories = model_memories,
                prepend_memories = prepend_memories,
                excise_prepend_embeds = exists(prepend_memories)
            )
            return self.agent.to_state_latent(state_embeds[:, -1])

    def action_cond(self, actions):
        actions = actions.to(self.agent.device)
        if self.agent.is_transition_action_space_raw:
            return actions

        actions = rearrange(actions, 'b d -> b 1 d')
        encoded = self.agent.encode_actions(actions)

        if self.agent.transition_action_space == 'global':
            encoded = self.agent.action_linear_rnn(encoded)

        return rearrange(self.agent.to_action_latent(encoded), 'b 1 d -> b d')

    def predict_next_latent(self, state_latents, actions):
        world_model = self.agent.get_world_model(self.world_model_index)
        transition = (
            world_model.ema_state_transition_for_planning.ema_model
            if self.use_ema_for_planning else world_model.state_transition_for_planning
        )
        action_cond = self.action_cond(actions)

        if world_model.transition_is_prefix:
            residual = transition.predict(
                rearrange(state_latents, 'b d -> b 1 d'),
                rearrange(action_cond, 'b ... -> b 1 ...')
            )
            residual = rearrange(residual, 'b 1 d -> b d')
        else:
            residual = transition((state_latents, action_cond))

        next_latents = world_model.apply_transition_residual(state_latents, residual)

        if self.agent.has_state_latent_clamp:
            next_latents = next_latents.clamp(-self.agent.state_latent_clamp_value, self.agent.state_latent_clamp_value)

        return next_latents

    def latent_prediction_loss(self):
        losses = []

        for state, action, next_state, prev_memories, memories in self.buffer:
            state_latents = self.encode_latents(state, memories = prev_memories)
            next_state_latents = self.encode_latents(next_state, memories = memories, no_grad = True)
            pred_next_latents = self.predict_next_latent(state_latents, action)

            losses.append(F.mse_loss(pred_next_latents, next_state_latents))

        return sum(losses) / len(losses)
