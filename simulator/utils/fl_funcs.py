#!/usr/bin/env python3

import numpy as np

# Import PyTorch root package import torch
import torch

from copy import deepcopy
from .logger import Logger

def construct_cdf(empirical_pdf):
    empirical_cdf = [item for item in empirical_pdf]
    for i in range(1, len(empirical_cdf)):
        empirical_cdf[i] = empirical_cdf[i] + empirical_cdf[i - 1]
    empirical_cdf[-1] += 1.0

    return empirical_cdf

def sample_element_from_cdf(empirical_cdf, rndgen):
    # Emulate F^-1(u_rv)
    u_rv = rndgen.random()
    return np.searchsorted(empirical_cdf, u_rv)


def get_sampled_clients(H, num_clients, args, exec_ctx):
    # clients are pre-sampled for deterministic participation among runs
    if args.client_sampling_type == "uniform":
        sampled_clients = [exec_ctx.np_random.choice(num_clients, args.num_clients_per_round, replace=False) for _ in range(args.rounds)]
        return sampled_clients
    elif args.client_sampling_type == "uniform-sampling-with-replacement":
        sampled_clients = [exec_ctx.np_random.choice(num_clients, args.num_clients_per_round, replace=True) for _ in range(args.rounds)]
        return sampled_clients
    elif args.client_sampling_type == "important-sampling-with-replacement":
        LiSum = sum(H["Li_all_clients"])
        pdf_for_clients = [Li / LiSum for Li in H["Li_all_clients"]]
        cdf_for_clients = construct_cdf(pdf_for_clients)

        sampled_clients = [ np.array( [sample_element_from_cdf(cdf_for_clients, exec_ctx.np_random) for z in range(args.num_clients_per_round)] )
                            for _ in range(args.rounds) ]

        return sampled_clients
    elif args.client_sampling_type == "poisson":
        # Poisson sampling, which allows empty sample in general
        sampled_clients = []
        for i in range(args.rounds):
            collect_clients = []
            for j in range(num_clients):
                rv = exec_ctx.np_random.uniform()
                if rv < args.client_sampling_poisson:
                    collect_clients.append(j)
            sampled_clients.append(np.asarray(collect_clients))
        return sampled_clients
    elif args.client_sampling_type == "poisson-no-empty":
        # Poisson sampling, but exlucing situation with empty sample
        while True:
            sampled_clients = []
            for i in range(args.rounds):
                collect_clients = []
                while len(collect_clients) == 0:
                    for j in range(num_clients):
                        rv = exec_ctx.np_random.uniform()
                        if rv < args.client_sampling_poisson:
                            collect_clients.append(j)
                sampled_clients.append(np.asarray(collect_clients))
            return sampled_clients
    else:
        assert(not "Unknown sampling type!")
        return None


def update_train_dicts(state_dicts, weights):
    logger = Logger.get("default")

    # get dictionary structure
    model_dict = deepcopy(state_dicts[0]['model'])
    optimiser_dict = deepcopy(state_dicts[0]['optimiser'])

    # model state_dict (structure layer key: value)
    logger.info('Aggregating model state dict.')
    for layer in  model_dict:
        layer_vals = torch.stack([state_dict['model'][layer] for state_dict in state_dicts])
        model_dict[layer] = weighted_sum(layer_vals, weights)

    # optimiser state dict (structure: layer key (numeric): buffers for layer: value)
    if 'state' in optimiser_dict:
        logger.info('Aggregating optimiser state dict.')
        for l_key in optimiser_dict['state']:
            layer = optimiser_dict['state'][l_key]
            for buffer in layer:
                buffer_vals = torch.stack([state_dict['optimiser']['state'][l_key][buffer]
                                           for state_dict in state_dicts])
                optimiser_dict['state'][l_key][buffer] = weighted_sum(buffer_vals, weights)
    return model_dict, optimiser_dict

def update_train_dicts_param_based(state_dicts, weights, clients):
    logger = Logger.get("default")

    # get dictionary structure
    state_dicts.waitForItem()
    first_state = state_dicts.popFront()

    optimiser_dict = deepcopy(first_state['optimiser'])
    model_dict = first_state['model'] * weights[0]

    # model state_dict (structure layer key: value)
    logger.info('Aggregating model state dict.')

    for i in range(1, clients):
        state_dicts.waitForItem()
        client_model = state_dicts.popFront()
        model_dict += client_model['model'] * weights[i]

        # optimiser state dict (structure: layer key (numeric): buffers for layer: value)
        if 'state' in optimiser_dict:
            #logger.info('Aggregating optimiser state dict.')
            for l_key in optimiser_dict['state']:
                layer = optimiser_dict['state'][l_key]
                for buffer in layer:
                    buffer_vals = client_model['optimiser']['state'][l_key][buffer]
                    optimiser_dict['state'][l_key][buffer] += buffer_vals * weights[i]
    return model_dict, optimiser_dict

def weighted_sum(tensors, weights):
    # Step-1: create view of tensor with extra artificial axis 1,1,1,1
    # Step-2: exploit broadcasting feature to form one tensor first axis - # weights, another axis correspond to each tensor
    # Step-3: perform summation of all tensors with this trick

    extra_dims = (1,)*(tensors.dim()-1)
    return torch.sum(weights.view(-1, *extra_dims) * tensors, dim=0)
