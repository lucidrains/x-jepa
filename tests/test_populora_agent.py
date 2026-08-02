import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

from x_jepa.x_jepa import Agent, Transformer
from populora import Population, evaluate_population_distributed

@param('selection_type', ('deterministic', 'probabilistic', 'fuss'))
@param('parent_selection_type', ('tournament', 'fuss', 'roulette', 'queen_bee'))
@param('crossover_type', ('average', 'dare', 'layer_wise', 'extrapolative', 'svd_subspace'))
@param('mutation_type', ('full_gaussian', 'svd_structured', 'layer_selective_gaussian', 'component_masking', 'neftune_style'))
def test_populora_agent(
    selection_type,
    parent_selection_type,
    crossover_type,
    mutation_type
):
    model = Transformer(
        dim = 32,
        depth = 1,
        causal = True
    )

    agent = Agent(
        state_encoder = nn.Linear(16, 32),
        action_encoder = nn.Linear(4, 32),
        dim_action = 4,
        model = model
    )

    pop = Population(
        agent,
        pop_size = 4,
        low_rank = 4,
        lora_targets = (
            'state_encoder.0',
            'action_encoder'
        )
    )

    states = torch.randn(2, 4, 16)
    actions = torch.randn(2, 3, 4).tanh()
    returns = torch.randn(2, 4)

    def eval_fn(population, idx):
        with population.route(individual = idx):
            loss, _ = population.model(states, actions, returns = returns)
        return -loss.item()

    # 1 round of evolution

    fitnesses = evaluate_population_distributed(pop, eval_fn)

    res = pop.select(selection_type, fitnesses, survive_frac = 0.5, elite_frac = 0.25)
    parents = pop.select_parents(parent_selection_type, fitnesses, num_children = len(res.culled))
    pop.crossover_(crossover_type, parents, res.culled, fitnesses = fitnesses)
    pop.mutate_(mutation_type, individuals = res.culled)

    # evaluate and merge best individual

    fitnesses = evaluate_population_distributed(pop, eval_fn)
    best_idx = fitnesses.argmax().item()

    with pop.route(individual = best_idx):
        loss_best, _ = agent(states, actions, returns = returns)

    pop.merge_(individual = best_idx)

    loss_merged, _ = agent(states, actions, returns = returns)

    assert torch.allclose(loss_best, loss_merged, atol = 1e-5)
