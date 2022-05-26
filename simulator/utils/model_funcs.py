#!/usr/bin/env python3

# Import PyTorch root package import torch
import torch

from torch import nn
from torch.nn import DataParallel
import time, copy, math, os, json

from collections import OrderedDict
import numpy as np

from copy import deepcopy
import pickle

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import models
from utils.logger import Logger
from utils.buffer import Buffer
from utils.utils import init_metrics_meter, neg_perplexity_from_loss, log_epoch_info
from data_preprocess.data_loader import get_num_classes
from .fl_funcs import update_train_dicts

from utils import algorithms, execution_context, gpu_utils
from models import mutils

CIFAR_MODELS = {
    'resnet': [18, 34, 50, 101, 152],
    'vgg': [11, 13, 16, 19],
    'wideresnet': [282, 284, 288]
}

CLIP_RNN_GRAD = 5

import random
import numpy as np

def robustLinearRegulizers(model, alphaR):
    value = None
    for p in model.parameters():
        xSqr = (p*p)

        if value is None:
            value = ((xSqr) / (xSqr + 1)).sum()
        else:
            value +=((xSqr) / (xSqr + 1)).sum()

    return value * alphaR

def noneRegulizer(model, alphaR):
    return None

def l2NormSqureRegulizers(model, alphaR):
    xsqr = None
    for p in model.parameters():
        if xsqr is None:
            xsqr = (p * p).sum()
        else:
            xsqr += (p * p).sum()

    return xsqr * 0.5 * alphaR

def get_global_regulizer(regul_name):
    if regul_name == "none":
        return noneRegulizer
    elif regul_name == "noncvx_robust_linear_regression":
        return robustLinearRegulizers
    elif regul_name == "cvx_l2norm_square_div_2":
        return l2NormSqureRegulizers

    return None

class MSEWeightedLossWithSumReduction:
    def __call__(self, input, target, weights = None):
        #weights = None
        diff = input - target

        if weights is None:
            loss = ((diff)**2).sum()
        else:
            weights = weights.to(diff.device)
            # strange bug somewhere inside PyTorch weights should be exact shape as diff() (n,) is not good when diff is (n,1)
            loss = (weights.view(diff.shape) * ( (diff) ** 2)).sum()

        return loss

class BCEWeightedLossWithSumReduction:
    def __call__(self, input, target, weights = None):
        loss = 0.0

        if weights is None:
            loss += -(torch.log(input[target > 0.5])).sum()
            loss += -(torch.log(1.0 - input[target < 0.5])).sum()
        else:
            weights = weights.to(target.device).view_as(input)
            loss += -(weights[target > 0.5] * torch.log(input[target > 0.5])).sum()
            loss += -(weights[target < 0.5] * torch.log(1.0 - input[target < 0.5])).sum()

        return loss

class CrossEntropyWeightedLossWithSumReduction:
    def __call__(self, input, target, weights = None):
        # input.shape[0] -- number of samples
        # input.shape[1] -- number of classes with linear scores for each class
        e = input.exp()
        eSum = torch.sum(e, dim = 1)
        eSumRepeat = eSum.repeat((input.shape[1],1)).T
        eSumNomralized = -torch.log(e/eSumRepeat)

        mask = torch.zeros(eSumNomralized.shape, device = input.device)
        mask[range(input.shape[0]), target] = 1.0
        if weights is None:
            loss = (eSumNomralized * mask).sum()
        else:
            weights = weights.to(input.device)
            weightRepeat = weights.repeat((input.shape[1], 1)).T
            loss = (eSumNomralized * mask * weightRepeat).sum()

        return loss

def get_training_elements(model_name, dataset, dataset_ref, args, resume_from, load_best, gpu, loss):
    # Define the model
    execution_context.torch_global_lock.acquire()
    if args.deterministic:
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed(args.manual_init_seed)
            torch.cuda.manual_seed_all(args.manual_init_seed)
        torch.manual_seed(args.manual_init_seed)
        random.seed(args.manual_init_seed)
        np.random.seed(args.manual_init_seed)

    model, current_round = initialise_model(model_name, dataset, dataset_ref, args, resume_from, load_best, use_pretrained = False)
    execution_context.torch_global_lock.release()

    model = model_to_device(model, gpu)

    criterion = None

    if loss == "crossentropy":
        criterion = CrossEntropyWeightedLossWithSumReduction()
    elif loss == "logistic":
        criterion = BCEWeightedLossWithSumReduction()
    elif loss == "mse":
        criterion = MSEWeightedLossWithSumReduction()
    else:
        raise ValueError(f"There are no registered loss for {loss}")

    return model, criterion, current_round


def initialise_model(model_name, dataset, dataset_ref, args, resume_from=None, load_best=None, use_pretrained=False):
    logger = Logger.get(args.run_id)

    model_prefix = [prefix for prefix in CIFAR_MODELS if model_name.startswith(prefix)]

    if len(model_prefix) == 1 and dataset.startswith('cifar') and \
            int(model_name.split(model_prefix[0])[1]) in CIFAR_MODELS[model_prefix[0]]:
        logger.debug("Loading cifar version of model")
        model_name = model_name + '_cifar'

    model = None

    if model_name == "linear":
        # Sample single sample to obtain information about data format
        example, target = dataset_ref[0]

        if not torch.is_tensor(example):
            example = torch.Tensor(example)

        if not torch.is_tensor(target):
            target = torch.Tensor([target])

        model = nn.Sequential( # 0 index reserved for samples in the batch
                               nn.Flatten(1),
                               # Fully connected layer from example.numel() to target.numel() units
                               nn.Linear(in_features = example.numel(), out_features = target.numel(), bias = False)
                              )

        # mutils.set_params_to_zero(model)

    if model_name == "logistic":
        # Sample single sample to obtain information about data format
        example, target = dataset_ref[0]

        if not torch.is_tensor(example):
            example = torch.Tensor(example)

        if not torch.is_tensor(target):
            target = torch.Tensor([target])

        model = nn.Sequential( # 0 index reserved for samples in the batch
                               nn.Flatten(1),
                               # Fully connected layer from example.numel() to target.numel() units
                               nn.Linear(in_features = example.numel(), out_features = target.numel(), bias = False),
                               # Apply sigmoid
                               nn.Sigmoid()
                              )
        # mutils.set_params_to_zero(model)

    if model is None:
        model = getattr(models, model_name)(pretrained=use_pretrained, num_classes=get_num_classes(dataset))

    current_round = 0

    if resume_from:
        # We can not use load_checkpoint(resume_from, load_best) due to circular dependencies
        # model, current_round = load_checkpoint(resume_from, load_best)
        #================================================================================================================
        if load_best:
            model_filename = os.path.join(resume_from, 'model_best.pth.tar')
            metric_filename = os.path.join(resume_from, 'best_metrics.json')
        else:
            model_filename = os.path.join(resume_from, 'model_last.pth.tar')
            metric_filename = os.path.join(resume_from, 'last_metrics.json')
    
        logger.info("Loading checkpoint '{}'".format(model_filename))
        state = torch.load(model_filename)
        logger.info("Loaded checkpoint '{}'".format(model_filename))
        with open(metric_filename, 'r') as f:
            metrics = json.load(f)
        model = state
        current_round = metrics['round']
        #==============================================================================================================

    return model, current_round

def model_to_device(model, device):
    if type(device) == list:          # If to allocate on more than one GPU
        model = model.to(device[0])   # Recursively convert parameters and buffers to device specific tensors
        model = DataParallel(model, device_ids=device)  # Create data-parallel model across devices "device"
                                                        # Details: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
                                                        # Moment: The batch size should be larger than the number of GPUs used.

    else:
        model = model.to(device)      # Recursively convert parameters and buffers to device specific tensors
    return model

def set_model_weights(model, weights, strict=True):
    """
    Sets the weight models in-place. To be used to integrate new updated weights.
    :param model: The model to be updated
    :param weights: (fl.common.Weights) List of np ndarrays representing the model weights
    :param strict: To require 1-to-1 parameter to weights association.s
    """
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=strict)


def get_model_weights(model):
    """Get model weights as a list of NumPy ndarrays."""

    # Move vals' data from GPU to CPU and then convert to numpy
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_lr_scheduler(optimiser, total_epochs, method='static'):
    """
    Implement learning rate scheduler.
    :param optimiser: A reference to the optimiser being usd
    :param total_epochs: The total number of epochs (from the args)
    :param method: The strategy to adjust the learning rate (multistep, cosine or static)
    :returns: scheduler on current step/epoch/policy
    """
    if method == 'cosine':
        return CosineAnnealingLR(optimiser, total_epochs)
    elif method == 'static':
        return MultiStepLR(optimiser, [total_epochs + 1])
    if method == 'cifar_1':  # VGGs + ResNets
        return MultiStepLR(optimiser, [int(0.5 * total_epochs), int(0.75 * total_epochs)], gamma=0.1)
    if method == 'cifar_2':  # WideResNets
        return MultiStepLR(optimiser, [int(0.3 * total_epochs), int(0.6 * total_epochs), int(0.8 * total_epochs)],
                           gamma=0.2)
    raise ValueError(f"{method} is not defined as scheduler name.")

#======================================================================================================================
def local_training(thread, client_state, client_id, msg, model_dict_original, optimiser_dict_original,
                   model, train_loader, criterion, local_optimiser, device, round, run_local_iters,
                   number_of_local_steps, is_rnn, print_freq, state_dicts, run_id):
    logger = Logger.get(run_id)

    # if thread is not None and hasattr(thread, "worker_stream"):
    #    torch.cuda.Stream.wait_stream(torch.cuda.default_stream(device), thread.worker_stream)

    # current_stream = torch.cuda.current_stream(device)
    # blas_handle = torch.cuda.current_blas_handle()
    # logger.info(f"CUDA stream {current_stream} and BLAS handle {blas_handle}")

    logger.info(msg)

    # Use of specified parameters or get them from thread context
    if thread != None:
        model = thread.model_copy
        local_optimiser = thread.local_optimiser_copy
        train_loader = thread.train_loader_copy

    fed_dataset = train_loader.dataset

    # Set current client
    if fed_dataset.num_clients > 1:
        fed_dataset.set_client(client_id)

    # Reconstruct model parameters
    mutils.set_params(model, model_dict_original)

    # Reconstruct optimizer parameters for client via using history
    optimiser_dict_found = algorithms.findRecentRecord(client_state['H'], client_id, "optimiser")

    if optimiser_dict_found is not None:
        local_optimiser.load_state_dict(optimiser_dict_found)
    else:
        local_optimiser.load_state_dict(optimiser_dict_original)

    # Reconstruct buffers in particular needed for batch normalization
    buffers_found = algorithms.findRecentRecord(client_state['H'], client_id, "buffers")
    if buffers_found is not None:
        mutils.set_buffers(model, buffers_found)

    metrics_meter = train_model(client_state, model, train_loader, criterion, local_optimiser, device, round, is_rnn, print_freq, run_local_iters = run_local_iters, iters_to_take = number_of_local_steps, run_id = run_id)

    # save current local model
    local_model_dict = mutils.get_params(model)

    # get local optimizer dict
    local_optimiser_dict = deepcopy(local_optimiser.state_dict())

    # get model buffers (e.g. include running means for BatchNorm)
    local_model_buffers = None
    local_model_buffers = mutils.get_buffers(model)

    # Force all operations to be completed in that worker stream so models are are valid and can be accessed from another threads
    if thread is not None and hasattr(thread, "worker_stream"):
        torch.cuda.Stream.wait_stream(torch.cuda.default_stream(device), thread.worker_stream)

    # Add (model state, optimizer state, client_id, client_id, client_state) into into result
    state_dicts.pushBack({'model': local_model_dict, 'optimiser': copy.deepcopy(local_optimiser_dict), 'buffers' : local_model_buffers, 'client_id' : client_id, 'client_state' : client_state})

    return

def non_local_training(thread, client_state, client_id, msg, model_dict_original, optimiser_dict_original,
                       model, train_loader, criterion, local_optimiser, device, round, run_local_iters,
                       number_of_local_steps, is_rnn, print_freq, state_dicts):

    model_dict_original = model_dict_original.to("cpu")

    args = (client_state, client_id, msg, model_dict_original, optimiser_dict_original,
            model, train_loader, criterion, local_optimiser, device, round, run_local_iters,
            number_of_local_steps, is_rnn, print_freq)

    h = client_state['H']
    client_state['H'] = {}
    for k,v in h.items():
        if k == "execution_context":
            continue

        if torch.is_tensor(v):
            client_state['H'][k] = v.cpu()
            continue

        if k == "history":
            client_state['H'][k] = {}
            for round in v.keys():
                round_info = v[round]
                composed_round_info = {}

                client_state['H'][k][round] = composed_round_info

                for round_key, round_value in round_info.items():
                    if torch.is_tensor(round_value):
                        round_value = round_value.cpu()

                    if round_key != "client_states":
                        composed_round_info[round_key] = round_value
                    elif round_key == "client_states":
                        client_states_values = round_value

                        composed_round_info[round_key] = {}
                        if client_id in client_states_values:
                            composed_round_info[round_key]['client_id'] = client_states_values[client_id]['client_id']
                            composed_round_info[round_key]['client_state'] = {}

                            for hKey, hValue in client_states_values[client_id]['client_state'].items():
                                if torch.is_tensor(hValue): hValue = hValue.cpu()
                                if hKey == 'H':
                                    hValue = None
                                composed_round_info[round_key]['client_state'][hKey] = hValue
            continue

        client_state['H'][k] = v

    serialized_args = pickle.dumps(args)
    client_state['H'] = h

    thread.external_socket.rawSendString("non_local_training")
    thread.external_socket.rawSend(serialized_args)
    description = thread.external_socket.rawRecvString()

    if description == "result_of_local_training":
        state_info = thread.external_socket.rawRecv()
        # {'model': local_model_dict, 'optimiser': copy.deepcopy(local_optimiser_dict), 'buffers' : local_model_buffers, 'client_id': client_id, 'client_state': client_state})
        state_info = pickle.loads(state_info)
        state_info['client_state']['H'] = h
        state_dicts.pushBack(state_info)

def run_one_communication_round(H, model, train_loader, criterion, local_optimiser, optimiser_dict_original, global_optimiser,
                                device, round, run_local_iters, number_of_local_steps,
                                is_rnn = False, sampled_clients = None, print_freq=10):
    logger = Logger.get(H["run_id"])

    metrics_meter = None
    fed_dataset = train_loader.dataset
    model_dict_original = mutils.get_params(model)

    sampled_clients_round = None

    # Feedback from algorithm: Possible information from underlying algorithm to use all clients in that round
    if 'request_use_full_list_of_clients' in H and H['request_use_full_list_of_clients']:
        sampled_clients_round = np.arange(H['total_clients'])
        logger.info(f"Sample all FL clients: {len(sampled_clients_round)}")
    else:
        sampled_clients_round = sampled_clients[round]
        logger.info(f"Sampled FL clients: {len(sampled_clients_round)}")

    # Store sampled clients in round in server state
    H["sampled_clients_in_round"] = sampled_clients_round

    # Information from future regarding clients from a next round
    sampled_clients_per_next_round = list()
    if round + 1 < len(sampled_clients):
        sampled_clients_per_next_round = sampled_clients[round + 1]

    clients_num_data = list()
    for client_id in sampled_clients_round:
        if fed_dataset.num_clients > 1:
            fed_dataset.set_client(client_id)
        clients_num_data.append(len(fed_dataset))
    clients_num_data = np.array(clients_num_data)

    # Result buffer which is created from scratch at each communication round
    state_dicts_thread_safe = Buffer(max_capacity = len(sampled_clients_round))

    exec_ctx = H["execution_context"]

    # Make sure that "model_dict_original" has been obtained correctly
    if gpu_utils.is_target_dev_gpu(device):
        torch.cuda.default_stream(device).synchronize()

    # Wait for all threads
    i = 0
    while i < len(sampled_clients_round):
        # Trainining in a serialized way locally
        if exec_ctx.local_training_threads.workers() == 0:
            client_id = sampled_clients_round[i]
            client_data_samples = clients_num_data[i]

            msg = f'Running local epoch for client: {client_id}, [{i + 1}/{len(sampled_clients_round)}]'
            client_state = algorithms.clientState(H, client_id, client_data_samples, round, device)
            res = local_training(None, client_state, client_id, msg, model_dict_original, optimiser_dict_original,
                                 model, train_loader, criterion, local_optimiser, device, round, run_local_iters, number_of_local_steps,
                                 is_rnn, print_freq, state_dicts_thread_safe, H["run_id"])
            i += 1
        else:
            client_id = sampled_clients_round[i]
            client_data_samples = clients_num_data[i]

            msg = f'Running local epoch for client: {client_id}, [{i + 1}/{len(sampled_clients_round)}]'
            next_device = exec_ctx.local_training_threads.next_dispatched_thread().device
            client_state = algorithms.clientState(H, client_id, client_data_samples, round, next_device)
            args = (client_state, client_id,msg, model_dict_original, optimiser_dict_original,
                    None, None, criterion, None, next_device, round, run_local_iters,
                    number_of_local_steps, is_rnn, print_freq, state_dicts_thread_safe, H["run_id"])
            exec_ctx.local_training_threads.dispatch(local_training, args)
            i += 1

        if exec_ctx.remote_training_threads.workers() > 0:
            # Delegate half of the work for a remote computers
            if i >= len(sampled_clients_round):
                continue

            client_id = sampled_clients_round[i]
            client_data_samples = clients_num_data[i]

            msg = f'Running local epoch for client: {client_id}, [{i + 1}/{len(sampled_clients_round)}]'
            next_device = exec_ctx.remote_training_threads.next_dispatched_thread().device
            client_state = algorithms.clientState(H, client_id, client_data_samples, round, next_device)
            args = (client_state, client_id,msg, model_dict_original, optimiser_dict_original,
                    None, None, criterion, None, next_device, round, run_local_iters,
                    number_of_local_steps, is_rnn, print_freq, state_dicts_thread_safe)
            exec_ctx.remote_training_threads.dispatch(non_local_training, args)
            i += 1

    # Prepapre for aggregation - weights based on the number of local data
    #if len(sampled_clients_round) >= 1:
    #    # weights based on the number of local data
    #    weights = torch.tensor(np.array(clients_num_data)/np.sum(clients_num_data)).to(device)

    # NO AWAIT ONLY BECAUSE THERE IS A QUEUE FOR INTER-THREAD COMMUNCIATION

    # Aggregation
    clients = len(sampled_clients_round)

    if True:
        # aggregate local models to global model
        global_optimiser.zero_grad()
        mutils.set_params(model, model_dict_original)

        # compute gradient from main path
        grad_server = algorithms.serverGradient(state_dicts_thread_safe, clients, model, model_dict_original, H)
        mutils.set_gradient(model, grad_server)

        # Experimental modification for select theoretical steps-size
        if "th_stepsize_cvx" in H["execution_context"].experimental_options or "th_stepsize_noncvx" in H["execution_context"].experimental_options:

            isStepSizeForNonConvexCase = ("th_stepsize_noncvx" in H["execution_context"].experimental_options)

            th_stepsize = algorithms.theoreticalStepSize(model_dict_original, grad_server, H, clients, train_loader, state_dicts_thread_safe, isStepSizeForNonConvexCase)


            for group in global_optimiser.param_groups:
                group['lr'] = th_stepsize
            global_optimiser.step()

            H = algorithms.serverGlobalStateUpdate(state_dicts_thread_safe, model, model_dict_original, round, grad_server, H, clients, sampled_clients_per_next_round)

            if isStepSizeForNonConvexCase:
                H["th_stepsize_noncvx"] = th_stepsize
            else:
                H["th_stepsize_cvx"] = th_stepsize

        else:
            global_optimiser.step()
            H = algorithms.serverGlobalStateUpdate(state_dicts_thread_safe, model, model_dict_original, round, grad_server, H, clients, sampled_clients_per_next_round)
            pass

    # Make sure that "model" has been updated
    if gpu_utils.is_target_dev_gpu(device):
        torch.cuda.default_stream(device).synchronize()

    return metrics_meter, sampled_clients_round

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


def train_model(client_state, model, train_loader, criterion, local_optimiser, device, round, is_rnn=False, print_freq=10, run_local_iters = True, iters_to_take=None, run_id = None):
    logger = Logger.get(run_id)

    metrics_meter = init_metrics_meter(round)
    size_of_dataset_in_samples = len(train_loader.dataset)

    model.train()
    if iters_to_take is None:
        iters_to_take = 1

    experimental_options = client_state['H']['execution_context'].experimental_options

    # Minibatch size for iterated-minibatch and sgd-nice
    tau = len(train_loader.dataset)
    if 'tau' in experimental_options:
        tau = experimental_options['tau']
        if "%" in tau:
            tau = float(tau.replace("%", "")) / 100.0
            tau = math.ceil(tau * len(train_loader.dataset))
        else:
            tau = int(tau)

    # Run_local_iters is False => We are run specific number of epochs
    if run_local_iters == False:
        # Finally calculate how much iterations do we need in case of using SGD
        # Convert epochs into iterations
        iters_to_take *= (math.ceil(len(train_loader.dataset) / float(tau)))

    # Client specific pseudo - random generator
    rndgen = np.random.RandomState(seed = client_state["seed"])

    # ==============================================================================================================
    if experimental_options["internal_sgd"] == "iterated-minibatch":
        if experimental_options['reshuffle'] == "each_round":
            # Reshuffle between for each round
            client_state['iterated-minibatch-indicies-full'] = rndgen.permutation(len(train_loader.dataset))

        elif experimental_options['reshuffle'] == "once_per_client":
            # Reshuffle once per client for whole experiment
            previous_indicies_full = algorithms.findRecentRecord(client_state['H'],
                                                                 client_state['client_id'],
                                                                 'iterated-minibatch-indicies-full')
            if previous_indicies_full is None:
                client_state['iterated-minibatch-indicies-full'] = rndgen.permutation(len(train_loader.dataset))
            else:
                client_state['iterated-minibatch-indicies-full'] = previous_indicies_full

        elif experimental_options['reshuffle'] == "until_exhausted":
            # Reshuffle until permutation will not be exhausted
            previous_indicies_full = algorithms.findRecentRecord(client_state['H'],
                                                                 client_state['client_id'],
                                                                 'iterated-minibatch-indicies-full')
            if previous_indicies_full is None:
                client_state['iterated-minibatch-indicies-full'] = rndgen.permutation(len(train_loader.dataset))
            else:
                client_state['iterated-minibatch-indicies-full'] = previous_indicies_full[ (iters_to_take * 1) * tau : ]
                if len(client_state['iterated-minibatch-indicies-full']) < iters_to_take * tau:
                    client_state['iterated-minibatch-indicies-full'] = rndgen.permutation(len(train_loader.dataset))

        #==============================================================================================================

    logger.info(f"For round #{round} the exact number of iteration derived from iters_to_take is {iters_to_take}")

    # Make local optimization
    for i in range(iters_to_take):
        #===============================================================================================================
        if experimental_options["internal_sgd"] == "iterated-minibatch":
            s = (i*tau)   % len(train_loader.dataset)
            e = ((i+1)*tau) % len(train_loader.dataset)
            client_state['iterated-minibatch-indicies']  = client_state['iterated-minibatch-indicies-full'][s:e]
            client_state['iterated-minibatch-weights'] =  np.array([1.0] * len(client_state['iterated-minibatch-indicies']))
        elif experimental_options["internal_sgd"] == "sgd-us":
            client_state['iterated-minibatch-indicies']  = np.array([rndgen.randint(len(train_loader.dataset))])
            client_state['iterated-minibatch-weights'] =  np.array([1.0] * len(client_state['iterated-minibatch-indicies']))
        elif experimental_options["internal_sgd"] == "sgd-nice":
            client_state['iterated-minibatch-indicies']  = rndgen.choice(len(train_loader.dataset), tau, replace=False)
            client_state['iterated-minibatch-weights'] =  np.array([1.0] * len(client_state['iterated-minibatch-indicies']))

        elif experimental_options["internal_sgd"] == "sgd-multi":
            sample = np.empty(tau, dtype=int)
            sample_weight = np.empty(tau, dtype=float)

            cdf = client_state['qi_cdf']
            inv_len_qi = 1.0 / len(client_state['qi'])

            for i in range(tau):
                sample[i] = sample_element_from_cdf(cdf, rndgen)
                sample_weight[i] = inv_len_qi * client_state['inv_qi'][sample[i]]

            client_state['iterated-minibatch-indicies']  = sample
            client_state['iterated-minibatch-weights'] = sample_weight

        elif experimental_options["internal_sgd"] == "sgd-ind":
            sample = []
            sample_weight = []

            while True:
                for i in range(len(train_loader.dataset)):
                    if rndgen.random() < client_state['pi'][i]:
                        sample.append(i)
                        sample_weight.append(0.0)
                if len(sample) > 0:
                    # Obtain at least one sample with at least one element in it
                    break

            for i in range(len(sample)):
                sample_weight[i] = len(sample) / (len(train_loader.dataset) * client_state['pi'][i])

            client_state['iterated-minibatch-indicies']  = np.array(sample)
            client_state['iterated-minibatch-weights'] = np.array(sample_weight)
            #===============================================================================================================

        # Not nessesary, but for safe
        local_optimiser.zero_grad()
        # Take local gradient
        local_iteration_number = (i, iters_to_take)
        approximate_f_value, local_grad = algorithms.localGradientEvaluation(client_state, model, train_loader, criterion, is_rnn, local_iteration_number)
        # Setup gradient
        mutils.set_gradient(model, local_grad)
        # Case when optimizer should work only after processing whole data
        if is_rnn:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_RNN_GRAD)

        if "local_gradients_per_iteration" in client_state:
            client_state["local_gradients_per_iteration"].append(mutils.get_gradient(model))

        client_state["approximate_f_value"].append(approximate_f_value)
        local_optimiser.step()

    # Make evalution
    if not False:
        for i, (data, label) in enumerate(train_loader):
            batch_size = data.shape[0]
            start_ts = time.time()
            if str(data.device) != device or str(label.device) != device:
                data, label = data.to(device), label.to(device)

            client_state["stats"]["dataload_duration"] += (time.time() - start_ts)
            input, label = get_train_inputs(data, label, model, batch_size, device, is_rnn)
            output_full, _ = forward(client_state, model, criterion, input, label, batch_size, size_of_dataset_in_samples, metrics_meter, is_rnn)

            if i % print_freq == 0:
                log_epoch_info(logger, i, train_loader, metrics_meter, client_state["stats"]["dataload_duration"],
                                                                       client_state["stats"]["inference_duration"],
                                                                       client_state["stats"]["backprop_duration"], train=True)

    return metrics_meter

def get_train_inputs(data, label, model, batch_size, device, is_rnn):
    if not is_rnn:
        input = (data,)
    else:
        hidden = model.init_hidden(batch_size, device)
        input = (data, hidden)
        label = label.reshape(-1)
    return input, label


def evaluate_model(model, val_loader, criterion, device, round,
                   print_freq=10, metric_to_optim='top_1', is_rnn=False):

    logger = Logger.get("default")

    metrics_meter = init_metrics_meter(round)
    if is_rnn:
        hidden = model.init_hidden(val_loader.batch_size, device)

    model.eval()

    # total_number_of_samples = len(val_loader.dataset)
    # Code wrap that stops autograd from tracking tensor 
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            batch_size = data.shape[0]
            start_ts = time.time()
            if str(data.device) != device or str(label.device) != device:
                data = data.to(device)
                label = label.to(device)

            if is_rnn:
                label = label.reshape(-1)

            dataload_duration = time.time() - start_ts
            if is_rnn:
                output, hidden = model(data, hidden)
            else:
                output = model(data)
            inference_duration = time.time() - (start_ts + dataload_duration)

            loss = compute_loss(model, criterion, output, label)
            loss = loss * (1.0 / batch_size)

            update_metrics(metrics_meter, loss, batch_size, output, label)

            if i % print_freq == 0:
                log_epoch_info(logger, i, val_loader, metrics_meter, dataload_duration,
                               inference_duration, backprop_duration=0., train=False)


    # Metric for avg/single model(s)
    logger.info(f'{metric_to_optim}: {metrics_meter[metric_to_optim].get_avg()}')

    return metrics_meter


def accuracy(output, label, topk=(1,)):
    """
    Extract the accuracy of the model.
    :param output: The output of the model
    :param label: The correct target label
    :param topk: Which accuracies to return (e.g. top1, top5)
    :return: The accuracies requested
    """
    maxk = max(topk)           # maximum k value in which we're interested in
    batch_size = label.size(0)

    if output.shape[1] == 1:
        # Case when output of the model is single real value (as output of sigmoid). Pred by rows examples, by (a single) column it contains indicies of most proable class.
        pred = (output > 0.5).int()
    else:
        # Case when output of the model are multiple real values. Pred by rows examples, by columns indicies of most proable class.
        _, pred = output.topk(maxk, 1, True, True)

    if pred.size(0) != 1:
        pred = pred.t()

    if pred.size() == (1,):
        correct = pred.eq(label)
    else:
        correct = pred.eq(label.view(1, -1).expand_as(pred)) # Expand label as prediction. Labels by rows will contain duplicate information, but pred by definition contains different indicies
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)    # As more k - more chance to come into top-k classication statistics
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def forward_backward(client_state, model, criterion, input, label, inference_duration, backprop_duration, batch_size, total_trainset_size, metrics_meter, is_rnn):
    start_ts = time.time()
    outputs = model(*input)
    if not is_rnn:
        hidden = None
        output = outputs
    else:
        output, hidden = outputs

    single_inference = time.time() - start_ts
    inference_duration += single_inference

    loss = compute_loss(model, criterion, output, label)
    loss = loss * (1.0/total_trainset_size)
    loss.backward()
    backprop_duration += time.time() - (start_ts + single_inference)

    # It's cheap to update loss, but because we slighlty move model after each  batch processing - it's not the true train loss, instead of it it's running_loss
    update_metrics(metrics_meter, loss, batch_size, output, label)
    return inference_duration, backprop_duration, output, hidden


def forward(client_state, model, criterion, input, label, batch_size, total_trainset_size, metrics_meter, is_rnn):
    start_ts = time.time()
    outputs = model(*input)
    if not is_rnn:
        hidden = None
        output = outputs
    else:
        output, hidden = outputs
    single_inference = time.time() - start_ts
    client_state["stats"]["inference_duration"] += single_inference

    loss = compute_loss(model, criterion, output, label)
    loss = loss * (1.0/total_trainset_size)

    # It's cheap to update loss, but because we slighlty move model after each  batch processing - it's not the true train loss, instead of it it's running_loss
    update_metrics(metrics_meter, loss, batch_size, output, label)
    return output, hidden

def compute_loss(model, criterion, output, label, weights = None):
    if (type(criterion) is torch.nn.MSELoss or type(criterion) is MSEWeightedLossWithSumReduction) and label.is_floating_point() == False:
        label = label.float()

    if (type(criterion) is torch.nn.BCELoss or type(criterion) is BCEWeightedLossWithSumReduction) and label.is_floating_point() == False:
        label = label.float()

    if (type(criterion) is torch.nn.CrossEntropyLoss or type(criterion) is CrossEntropyWeightedLossWithSumReduction) and label.is_floating_point() == True:
        label = label.long()

    if type(output) == list and len(output) > 1:
        if type(model).__name__.lower() == 'inception3':
            loss = criterion(output[0], label, weights) + 0.4 * criterion(output[1], label, weights)
        else:
            loss = sum([criterion(out, label, weights) for out in output])
    else:
        loss = criterion(output, label, weights)

    return loss


def update_metrics(metrics_meter, loss, batch_size, output, label):
    metrics_meter['loss'].update(loss.item(), batch_size)
    metrics_meter['neq_perplexity'].update(neg_perplexity_from_loss(loss.item()), batch_size)

    if output.shape[1] >= 5:
        acc = accuracy(output, label, (1, 5))
        metrics_meter['top_1_acc'].update(acc[0], batch_size)
        metrics_meter['top_5_acc'].update(acc[1], batch_size)
    else:
        acc = accuracy(output, label, (1,))
        metrics_meter['top_1_acc'].update(acc[0], batch_size)


def get_optimiser(params_to_update, optimiser_name, lr, momentum, weight_decay):
    if optimiser_name == 'sgd':
        optimiser = torch.optim.SGD(params_to_update, lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    elif optimiser_name == 'adam':
        optimiser = torch.optim.Adam(params_to_update, lr, weight_decay=weight_decay)
    elif optimiser_name == 'rmsprop':
        optimiser = torch.optim.RMSprop(params_to_update, lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
    else:
        raise ValueError("optimiser not supported")

    return optimiser
