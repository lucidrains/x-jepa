import pytest
import torch
from torch import nn
from torch.testing import assert_close

from x_jepa import (
    Agent,
    LoRAHook,
    TestTimeAdapter,
    adajepa_module_paths,
    gradient_path_module_paths,
    last_swiglu_ffn_lora_paths
)
from x_jepa.test_time import collect_params
from x_jepa.x_jepa import Transformer


def make_agent(
    dim = 16,
    depth = 2,
    mlp_transition = False,
    transition_action_space = 'raw'
):
    torch.manual_seed(0)

    state_encoder = nn.Sequential(
        nn.Linear(4, 32),
        nn.GELU(),
        nn.Linear(32, dim),
        nn.LayerNorm(dim)
    )
    action_encoder = nn.Linear(2, dim)
    action_decoder = None if transition_action_space == 'raw' else nn.Linear(dim, 2)

    state_transition = None
    if mlp_transition:
        action_cond_dim = 2 if transition_action_space == 'raw' else dim
        state_transition = nn.Sequential(
            nn.Linear(dim + action_cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, dim)
        )

    return Agent(
        state_encoder = state_encoder,
        action_encoder = action_encoder,
        action_decoder = action_decoder,
        model = Transformer(dim = dim, depth = depth, causal = True),
        state_transition = state_transition,
        dim_action = 2,
        continuous_actions = True,
        transition_action_space = transition_action_space,
        state_linear_rnn_depth = 1,
        action_linear_rnn_depth = 1,
        transition_horizon = 4,
        transition_lookahead = 2
    )


class ToyEnv:
    def __init__(self, batch = 2):
        self.batch = batch
        self.is_vector_env = True
        self.num_envs = batch
        self.state = None

    def reset(self):
        self.state = torch.zeros(self.batch, 4)
        return self.state.clone(), {}

    def step(self, action):
        action = torch.as_tensor(action, dtype = torch.float32).reshape(self.batch, 2)
        self.state = self.state + 0.2 * torch.cat([action, -action], dim = -1)
        reward = -self.state.norm(dim = -1)
        terminated = torch.zeros(self.batch, dtype = torch.bool)
        truncated = torch.zeros(self.batch, dtype = torch.bool)
        return self.state.clone(), reward, terminated, truncated, {}


def test_adajepa_and_gradient_path_module_paths():
    agent = make_agent(depth = 2)

    paths = adajepa_module_paths(agent, adapt_encoder = False)
    assert paths['predictor_paths'] == (
        'world_models.0.state_transition_for_planning.transformer.layers.1',
        'world_models.0.state_transition_for_planning.to_pred'
    )
    assert paths['encoder_paths'] == ()

    paths_with_enc = adajepa_module_paths(agent, adapt_encoder = True)
    assert paths_with_enc['encoder_paths'] == ('state_encoder.0.2',)

    paths_ema = adajepa_module_paths(agent, use_ema_for_planning = True)
    assert paths_ema['predictor_paths'] == (
        'world_models.0.ema_state_transition_for_planning.ema_model.transformer.layers.1',
        'world_models.0.ema_state_transition_for_planning.ema_model.to_pred'
    )

    mlp_agent = make_agent(mlp_transition = True)
    mlp_paths = adajepa_module_paths(mlp_agent)
    assert mlp_paths['predictor_paths'] == ('world_models.0.state_transition_for_planning',)


@pytest.mark.parametrize('transition_action_space', ('raw', 'global'))
def test_default_updates_entire_network_along_gradient_path(transition_action_space):
    agent = make_agent(depth = 2, transition_action_space = transition_action_space)
    adapter = agent.test_time_adapter(lr_predictor = 5e-3, lr_encoder = 5e-3, steps = 1)
    adapter.ensure_optimizer()

    state = torch.randn(3, 4)
    action = torch.randn(3, 2).tanh()
    next_state = torch.randn(3, 4)

    # verify default selected params match the exact autograd gradient path of latent_prediction_loss

    adapter.buffer.append((state, action, next_state, None, None))
    for p in agent.parameters():
        p.grad = None

    loss = adapter.latent_prediction_loss()
    loss.backward()

    grad_param_ids = {id(p) for p in agent.parameters() if p.grad is not None}
    selected_param_ids = {id(p) for p in adapter.params}
    assert selected_param_ids == grad_param_ids

    for p in agent.parameters():
        p.grad = None
    adapter.reset()

    off_path_params = [p for p in agent.parameters() if id(p) not in selected_param_ids]
    off_path_init = [p.detach().clone() for p in off_path_params]
    on_path_init = [p.detach().clone() for p in adapter.params]

    loss_1 = adapter.observe(step = 0, state = state, action = action, next_state = next_state)
    loss_2 = adapter.observe(step = 1, state = state, action = action, next_state = next_state)
    assert loss_2 < loss_1

    init_by_id = {id(p): init for p, init in zip(adapter.params, on_path_init)}
    for path in adapter.predictor_paths + adapter.encoder_paths:
        mod_params = list(agent.get_submodule(path).parameters())
        assert any(not torch.equal(p, init_by_id[id(p)]) for p in mod_params), f'{path} did not update'

    for p, init in zip(off_path_params, off_path_init):
        assert_close(p, init)

    adapter.reset()
    for p, init in zip(adapter.params, on_path_init):
        assert_close(p, init)


def test_adapter_updates_only_selected_params_and_resets_under_no_grad():
    agent = make_agent(depth = 2)
    paths = adajepa_module_paths(agent, adapt_encoder = True)

    adapter = TestTimeAdapter(
        agent,
        **paths,
        lr_predictor = 1e-2,
        lr_encoder = 1e-2,
        buffer_size = 4,
        steps = 2
    )

    selected_params = collect_params(agent, paths['predictor_paths'] + paths['encoder_paths'])
    selected_ids = {id(p) for p in selected_params}
    frozen_params = [p for p in agent.parameters() if id(p) not in selected_ids]

    initial_selected = [p.detach().clone() for p in selected_params]
    initial_frozen = [p.detach().clone() for p in frozen_params]

    state = torch.randn(3, 4)
    action = torch.randn(3, 2).tanh()
    next_state = torch.randn(3, 4)

    with torch.no_grad():
        loss_1 = adapter.observe(step = 0, state = state, action = action, next_state = next_state)
        loss_2 = adapter.observe(step = 1, state = state, action = action, next_state = next_state)

    assert torch.isfinite(loss_1) and torch.isfinite(loss_2)
    assert loss_2 < loss_1
    assert len(adapter.buffer) == 2

    assert any(not torch.equal(p, init) for p, init in zip(selected_params, initial_selected))
    for p, init in zip(frozen_params, initial_frozen):
        assert_close(p, init)
        assert p.grad is None

    adapter.reset()
    assert len(adapter.buffer) == 0
    assert len(adapter.optimizer.state) == 0
    for p, init in zip(selected_params, initial_selected):
        assert_close(p, init)

    with torch.no_grad():
        adapter.observe(step = 1, state = state, action = action, next_state = next_state)
        assert any(not torch.equal(p, init) for p, init in zip(selected_params, initial_selected))

        loss_after_auto_reset = adapter.observe(step = 0, state = state, action = action, next_state = next_state)
        assert len(adapter.buffer) == 1
        assert_close(loss_after_auto_reset, loss_1)


def test_planning_ergonomics_callback_wrapper_and_manual_loop():
    agent = make_agent(depth = 2, transition_action_space = 'global')
    paths = adajepa_module_paths(agent, adapt_encoder = False)

    with pytest.raises(AssertionError, match = 'use_ema_for_planning'):
        TestTimeAdapter(agent, **paths, use_ema_for_planning = True)

    adapter = agent.test_time_adapter(**paths, lr_predictor = 5e-3, buffer_size = 3, steps = 1)

    with pytest.raises(AssertionError, match = 'state, action, and next_state'):
        adapter.observe(step = 0, state = torch.randn(2, 4), action = None, next_state = torch.randn(2, 4))

    selected_params = collect_params(agent, paths['predictor_paths'])
    before = [p.detach().clone() for p in selected_params]

    # 1. Direct callback usage: `callback = adapter`

    agent.interact_with_environment(
        ToyEnv(batch = 2),
        max_steps = 3,
        goal_state = torch.zeros(1, 4),
        horizon = 2,
        pop_size = 8,
        generations = 1,
        commit_k_steps = 1,
        callback = adapter
    )

    assert len(adapter.buffer) == 3
    assert adapter.buffer[0][3] is None
    assert_close(
        adapter.encode_latents(adapter.buffer[1][0], memories = adapter.buffer[1][3], no_grad = True),
        adapter.encode_latents(adapter.buffer[0][2], memories = adapter.buffer[0][4], no_grad = True)
    )
    assert any(not torch.equal(p, init) for p, init in zip(selected_params, before))

    # 2. `adapter.interact_with_environment` composing user callback + `ttt_loss`

    adapter.reset()
    logged_losses = []
    adapter.interact_with_environment(
        ToyEnv(batch = 2),
        max_steps = 3,
        goal_state = torch.zeros(1, 4),
        horizon = 2,
        pop_size = 8,
        generations = 1,
        commit_k_steps = 1,
        callback = lambda ttt_loss = None: logged_losses.append(ttt_loss)
    )
    assert len(logged_losses) == 3 and all(torch.isfinite(l) for l in logged_losses)
    adapter.reset()

    # 3. Manual planning loop: `adapter.plan(state)` -> `env.step` -> `adapter.update(next_state)`

    env = ToyEnv(batch = 2)
    state, _ = env.reset()
    for _ in range(3):
        act = adapter.plan(state, goal_state = torch.zeros(1, 4), horizon = 2, pop_size = 8, generations = 1)
        next_state, _, _, _, _ = env.step(act)
        loss = adapter.update(next_state)
        assert torch.isfinite(loss)
        state = next_state

    assert len(adapter.buffer) == 3
    assert any(not torch.equal(p, init) for p, init in zip(selected_params, before))
    adapter.reset()


def test_lora_designated_layers_and_default_gradient_path_lora():
    agent = make_agent(depth = 2)
    modules_before = dict(agent.named_modules())
    base_snapshot = [(name, p, p.detach().clone()) for name, p in agent.named_parameters()]

    state = torch.randn(3, 4)
    action = torch.randn(3, 2).tanh()
    next_state = torch.randn(3, 4)

    lora_targets = (
        'state_transition_for_planning.transformer.layers.1.1.to_out',
        'world_models.0.state_transition_for_planning.to_pred.1.weight'
    )

    probe = agent.test_time_adapter(steps = 0)
    with torch.no_grad():
        z0 = probe.encode_latents(state, no_grad = True)
        pred_before_hook = probe.predict_next_latent(z0, action)

    with agent.test_time_adapter(
        lora_paths = lora_targets,
        lora_rank = 4,
        lora_alpha = 8.,
        lr_lora = 1e-2,
        steps = 2,
        buffer_size = 4
    ) as adapter:
        assert len(adapter.lora_hooks) == 2
        assert all(isinstance(h, LoRAHook) for h in adapter.lora_hooks.values())

        # forward-hook LoRA never mutates agent's module tree or named_parameters

        assert dict(agent.named_modules()) == modules_before
        assert [name for name, _ in agent.named_parameters()] == [name for name, _, _ in base_snapshot]

        with torch.no_grad():
            pred_at_init = adapter.predict_next_latent(z0, action)
        assert_close(pred_at_init, pred_before_hook)

        loss_1 = adapter.observe(step = 0, state = state, action = action, next_state = next_state)
        loss_2 = adapter.observe(step = 1, state = state, action = action, next_state = next_state)
        assert loss_2 < loss_1

        with torch.no_grad():
            pred_after_adapt = adapter.predict_next_latent(z0, action)
        assert not torch.allclose(pred_after_adapt, pred_before_hook)

        for _, param_ref, base_p in base_snapshot:
            assert_close(param_ref, base_p)

        adapter.reset()
        with torch.no_grad():
            assert_close(adapter.predict_next_latent(z0, action), pred_before_hook)

    # context manager exit removes all forward hooks cleanly

    assert len(agent.get_submodule('world_models.0.state_transition_for_planning.to_pred.1')._forward_hooks) == 0

    # `use_lora = True` with `lora_paths` omitted defaults to all nn.Linear layers along the gradient path

    with agent.test_time_adapter(use_lora = True, lora_rank = 4, lr_lora = 5e-3, steps = 1) as full_lora:
        grad_paths = gradient_path_module_paths(agent)
        assert len(full_lora.lora_hooks) > 2
        assert any(p.startswith('state_encoder.') for p in full_lora.lora_paths)
        assert any('state_transition_for_planning' in p for p in full_lora.lora_paths)
        loss_a = full_lora.observe(step = 0, state = state, action = action, next_state = next_state)
        loss_b = full_lora.observe(step = 1, state = state, action = action, next_state = next_state)
        assert loss_b < loss_a
        for _, param_ref, base_p in base_snapshot:
            assert_close(param_ref, base_p)

    # targeting a layer disconnected from the latent prediction path raises a clear RuntimeError

    with agent.test_time_adapter(lora_paths = ('action_encoder',), lr_lora = 1e-2) as bad_adapter:
        with pytest.raises(RuntimeError, match = 'no gradient'):
            bad_adapter.observe(step = 0, state = state, action = action, next_state = next_state)


def test_lora_last_swiglu_ffn_flag():
    agent = make_agent(depth = 2)
    base_snapshot = [(name, p, p.detach().clone()) for name, p in agent.named_parameters()]

    assert last_swiglu_ffn_lora_paths(agent) == (
        'world_models.0.state_transition_for_planning.transformer.layers.1.2.1',
        'world_models.0.state_transition_for_planning.transformer.layers.1.2.3'
    )
    assert last_swiglu_ffn_lora_paths(agent, use_ema_for_planning = True) == (
        'world_models.0.ema_state_transition_for_planning.ema_model.transformer.layers.1.2.1',
        'world_models.0.ema_state_transition_for_planning.ema_model.transformer.layers.1.2.3'
    )

    with pytest.raises(AssertionError, match = 'not both'):
        agent.test_time_adapter(lora_last_swiglu_ffn = True, lora_paths = ('to_state_latent.1',))

    mlp_agent = make_agent(mlp_transition = True)
    with pytest.raises(AssertionError, match = 'no SwiGLU'):
        mlp_agent.test_time_adapter(lora_last_swiglu_ffn = True)

    state = torch.randn(3, 4)
    action = torch.randn(3, 2).tanh()
    next_state = torch.randn(3, 4)

    with agent.test_time_adapter(
        lora_last_swiglu_ffn = True,
        lora_rank = 4,
        lr_lora = 1e-2,
        steps = 2
    ) as adapter:
        assert adapter.use_lora
        assert adapter.lora_paths == (
            'world_models.0.state_transition_for_planning.transformer.layers.1.2.1',
            'world_models.0.state_transition_for_planning.transformer.layers.1.2.3'
        )
        assert adapter.predictor_paths == ()
        assert adapter.encoder_paths == ()
        assert len(adapter.lora_hooks) == 2

        loss_1 = adapter.observe(step = 0, state = state, action = action, next_state = next_state)
        loss_2 = adapter.observe(step = 1, state = state, action = action, next_state = next_state)
        assert loss_2 < loss_1

        for _, param_ref, base_p in base_snapshot:
            assert_close(param_ref, base_p)

    assert len(agent.get_submodule('world_models.0.state_transition_for_planning.transformer.layers.1.2.1')._forward_hooks) == 0
    assert len(agent.get_submodule('world_models.0.state_transition_for_planning.transformer.layers.1.2.3')._forward_hooks) == 0
