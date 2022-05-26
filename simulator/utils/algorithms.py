#!/usr/bin/env python3

import random
import time
import copy
import math

# Import PyTorch root package import torch
import torch

import numpy as np

from utils import execution_context
from utils import  model_funcs
from utils import  compressors
from models import mutils

import utils
import argparse
from utils.logger import Logger

#======================================================================================================================
def evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, evaluate_function = False, device = None, args = None):
    """
    Evalute gradient for model at current point and optionally update statistics and return loss value at current point.

    Parameters:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - full_gradient_oracles, samples_gradient_oracles, dataload_duration, inference_duration, backprop_duration
        evaluate_function(bool): if true then returned value is CPU scalar which describes loss function value
    Returns:
        If evaluate_function is True then sclar with local ERM value
    """
    model.train(True)

    if update_statistics:
        client_state['stats']['full_gradient_oracles'] += 1
        client_state['stats']['samples_gradient_oracles'] += len(dataloader.dataset)

    # Zero out previous gradient
    for p in model.parameters():
        p.grad = None

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    function_value = None

    if evaluate_function:
        function_value = torch.Tensor([0.0]).to(device)

    for i, (data, label) in enumerate(dataloader):
        start_ts = time.time()
        batch_size = data.shape[0]
        if str(data.device) != device or str(label.device) != device:
            data, label = data.to(device), label.to(device)

        if update_statistics:
            client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

        input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

        start_ts = time.time()
        outputs = model(*input)

        if not is_rnn:
            hidden = None
            output = outputs
        else:
            output, hidden = outputs

        loss = model_funcs.compute_loss(model, criterion, output, label)
        loss = loss * (1.0/total_number_of_samples)

        if evaluate_function:
            function_value += loss

        if update_statistics:
            client_state["stats"]["inference_duration"] += (time.time() - start_ts)

        start_ts = time.time()
        loss.backward()

        if update_statistics:
            client_state["stats"]["backprop_duration"] += (time.time() - start_ts)

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        R.backward()
        Rvalue = R.item()

    if evaluate_function:
        return function_value.item() + Rvalue
    else:
        return None
#======================================================================================================================
def evaluateSgd(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, evaluate_function = False, device = None, args = None):
    """
    Evalute gradient estimator with using global context for model at current point and optionally update statistics and return loss value at current point.

    Parameters:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - full_gradient_oracles, samples_gradient_oracles, dataload_duration, inference_duration, backprop_duration
        evaluate_function(bool): if true then returned value is CPU scalar which describes loss function value
    Returns:
        If evaluate_function is True then sclar with local ERM value
    """
    exec_ctx = client_state['H']["execution_context"]

    if "internal_sgd" not in exec_ctx.experimental_options:
        return evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics=update_statistics, evaluate_function=evaluate_function, device=device)

    internal_sgd = exec_ctx.experimental_options['internal_sgd']
    if internal_sgd == 'full-gradient':
        return evaluateGradient(client_state, model, dataloader, criterion, is_rnn, update_statistics=update_statistics, evaluate_function=evaluate_function, device=device)

    model.train(True)

    # Zero out previous gradient
    for p in model.parameters():
        p.grad = None

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    function_value = None

    if evaluate_function:
        function_value = torch.Tensor([0.0]).to(device)

    # ==================================================================================================================
    indicies = None
    weights = None
    if internal_sgd == "sgd-nice" or internal_sgd == 'sgd-us' or internal_sgd == 'iterated-minibatch' or internal_sgd == 'sgd-multi' or internal_sgd == 'sgd-ind':
        indicies = client_state['iterated-minibatch-indicies']
        indicies = torch.from_numpy(indicies)

        weights = client_state['iterated-minibatch-weights']
        weights = torch.from_numpy(weights)
    #==================================================================================================================
    batch_size_ds = dataloader.batch_size
    iterations = math.ceil(len(indicies) / float(batch_size_ds))

    sampled_samples = len(indicies)

    for i in range(iterations):
        data = []
        label = []
        batch_weights = []

        for j in range(batch_size_ds):
            index = i * batch_size_ds + j
            if index >= sampled_samples:
                break

            d, t = dataloader.dataset[indicies[index]]
            data.append(d.unsqueeze(0))

            batch_weights.append(weights[index].unsqueeze(0))

            if not torch.is_tensor(t):
                if type(criterion) is torch.nn.MSELoss or type(criterion) is model_funcs.MSEWeightedLossWithSumReduction:
                    label.append(torch.Tensor([t]))
                else:
                    label.append(torch.LongTensor([t]))
            else:
                label.append(t.unsqueeze(0))

        data = torch.cat(data)
        label = torch.cat(label)
        batch_weights = torch.cat(batch_weights)

        start_ts = time.time()
        batch_size = data.shape[0]
        if str(data.device) != device or str(label.device) != device:
            data, label = data.to(device), label.to(device)

        if update_statistics:
            client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

        if update_statistics:
            client_state['stats']['full_gradient_oracles'] += float(batch_size) / total_number_of_samples
            client_state['stats']['samples_gradient_oracles'] += batch_size

        input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

        start_ts = time.time()
        outputs = model(*input)

        if not is_rnn:
            hidden = None
            output = outputs
        else:
            output, hidden = outputs

        loss = model_funcs.compute_loss(model, criterion, output, label, batch_weights)
        loss = loss * (1.0/sampled_samples)

        if evaluate_function:
            function_value += loss

        if update_statistics:
            client_state["stats"]["inference_duration"] += (time.time() - start_ts)

        start_ts = time.time()
        loss.backward()

        if update_statistics:
            client_state["stats"]["backprop_duration"] += (time.time() - start_ts)

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        R.backward()
        Rvalue = R.item()

    if evaluate_function:
        return function_value.item() + Rvalue
    else:
        return None
#======================================================================================================================
def evaluateFunction(client_state, model, dataloader, criterion, is_rnn, update_statistics = True, device = None, args = None):
    """
    Evalute gradient for model at current point and optionally update statistics and return loss value at current point.

    Parameters:
        client_state(dict): information about client. used information - statistics, used device
        model(torch.nn.Module): used model for which trainable variables we will evaluate full gradient
        dataloader: used dataloader for fetch records for evaluate local loss during training
        criterion: used criteria with setuped reduction as sum. After evalute reduction correct scaling is a part of evaluation
        is_rnn(bool): flag which specofy that what we evaluate is rnn
        update_statistics(bool): update the following statistics - dataload_duration, inference_duration
    Returns:
        Scalar with local ERM value
    """
    model.train(False)

    if device is None:
        device = client_state["device"]

    total_number_of_samples = len(dataloader.dataset)
    total_loss = torch.Tensor([0.0]).to(device)

    # code wrap that stops autograd from tracking tensor 
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            start_ts = time.time()
            batch_size = data.shape[0]
            if str(data.device) != device or str(label.device) != device:
                data, label = data.to(device), label.to(device)

            if update_statistics:
                client_state["stats"]["dataload_duration"] += (time.time() - start_ts)

            input, label = model_funcs.get_train_inputs(data, label, model, batch_size, device, is_rnn)

            start_ts = time.time()
            outputs = model(*input)

            if not is_rnn:
                hidden = None
                output = outputs
            else:
                output, hidden = outputs

            loss = model_funcs.compute_loss(model, criterion, output, label)
            loss = loss * (1.0/total_number_of_samples)

            if update_statistics:
                client_state["stats"]["inference_duration"] += (time.time() - start_ts)
            total_loss += loss

    if args is None:
        args = client_state["H"]["args"]

    regulizer_global = model_funcs.get_global_regulizer(args.global_regulizer)
    R = regulizer_global(model, args.global_regulizer_alpha)

    Rvalue = 0.0
    if R is not None:
        Rvalue = R.item()

    return total_loss.item() + Rvalue

def findRecentRecord(H, client_id, field):
    """
    Find in history records recent information about query record in client_states.

    Parameters:
        H(dict): information about client. used information - statistics, used device
        client_id(int): integer number for client
        field(str): name of the field which we are trying to find history
    Returns:
        Return requested object if it found or None if object is not found
    """
    history = H['history']

    history_keys = [k for k in history.keys()]
    history_keys.sort(reverse=True)

    for r in history_keys:
        clients_history = history[r]
        if client_id in clients_history['client_states']:
            client_prev_state = clients_history['client_states'][client_id]['client_state']
            if field in client_prev_state:
                return_object = client_prev_state[field]
                return return_object

            else:
                # Assumption -- if client has been sampled then field have to be setuped
                return None
    return None

def findRecentRecordAndRemoveFromHistory(H, client_id, field):
    """
    Find in history records recent information about query record in client_states.
    If record has been found return it, but before that remove itself from history.

    Parameters:
        H(dict): information about client. used information - statistics, used device
        client_id(int): integer number for client
        field(str): name of the field which we are trying to find history
    Returns:
        Return requested object if it found or None if object is not found
    """
    history = H['history']
    history_keys = [k for k in history.keys()]
    history_keys.sort(reverse=True)

    for r in history_keys:
        clients_history = history[r]
        if client_id in clients_history['client_states']:
            client_prev_state = clients_history['client_states'][client_id]['client_state']
            if field in client_prev_state:
                return_object = client_prev_state[field]
                client_prev_state[field] = None
                return return_object
            else:
                # Assumption -- if client has been sampled then field have to be setuped
                return None
    return None

#======================================================================================================================
def get_logger(H):
    """
    Help function to get logger
    Parameters:
        H(dict): server state
    Returns:
        Reference to logger
    """

    my_logger = Logger.get(H["args"].run_id)
    return my_logger

def has_experiment_option(H, name):
    """
    Check that experimental option is presented

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        True if option is present
    """
    return name in H["execution_context"].experimental_options

def get_experiment_option_f(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        Value of requested value converted to float
    """
    return float(H["execution_context"].experimental_options[name])

def get_experiment_option_int(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        Value of requested value converted to int
    """
    return int(H["execution_context"].experimental_options[name])

def get_experiment_option_str(H, name):
    """
    Get experimental option to carry experiments with algorithms

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        Value of requested value converted to string
    """
    return str(H["execution_context"].experimental_options[name])

def get_initial_shift(args:argparse.Namespace, D:int, grad_start:torch.Tensor):
    """Help method to get initial shifts"""
    if args.initialize_shifts_policy == "full_gradient_at_start":
        return grad_start.detach().clone().to(args.device)
    else:
        return torch.zeros(D).to(args.device)
#======================================================================================================================
class MarinaAlgorithm:
    '''
    MARINA Algoritm [Gorbunov et al., 2021]: https://arxiv.org/abs/2102.07845
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D: int, total_clients: int,
                              grad_start: torch.Tensor) -> dict:

        compressor = compressors.initCompressor(args.client_compressor, D)

        state = {"x_prev": mutils.get_params(model),  # previous iterate
                 "test_ber_rv": 0.0,  # test_ber_rv = 0.0 will force fisrt iteration be a full gradient evaluation
                 "total_clients": total_clients,
                 "w": compressor.getW(),
                 "compressor_fullname" : compressor.fullName()
                 }
        return state

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses,
                            use_steps_size_for_non_convex_case):
        # Step size for non-convex case
        m = 1.0
        workers = H['total_clients']

        Li_all_clients = getLismoothForClients(H, clients_responses)
        Ltask = (np.mean(max(Li_all_clients) ** 2)) ** 0.5
        w = H["w"]

        p = 1.0 / (1 + w)  # For RAND-K compressor
        step_3 = 1.0 / (Ltask * (1 + ((1 - p) * (w) / (p * workers)) ** 0.5))

        return step_3

    @staticmethod
    def clientState(H:dict, clientId:int, client_data_samples:int, device:str)->dict:
        logger = Logger.get(H["run_id"])

        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)

        p = 1.0 / (1.0 + compressor.getW())

        state = {}
        if H["test_ber_rv"] <= p:
            state.update({"p" : p, "ck": 1, "client_compressor" : compressor})
        else:
            state.update({"p" : p, "ck": 0, "client_compressor": compressor})

        return state

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        if client_state["ck"] == 1:
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)
            client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
            return fApprox, grad_cur
        else:
            client_id = client_state["client_id"]
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            reconstruct_params = mutils.get_params(model)
            mutils.set_params(model, client_state["H"]["x_prev"])
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
            grad_prev = mutils.get_gradient(model)
            mutils.set_params(model, reconstruct_params)

            g_prev = client_state["H"]["g_prev"].to(client_state["device"])
            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - grad_prev)

            # Comments: server knows g_prev
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            return fApprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer, clients: int, model: torch.nn.Module, params_current: torch.Tensor, H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H["g_prev"] = grad_server
        H["x_prev"] = mutils.get_params(model)

        H["test_ber_rv"] = H['execution_context'].np_random.random()
        return H
#======================================================================================================================

class PageAlgorithm:
    '''
    Page Algoritm [Zhize Li et al., 2021]: https://arxiv.org/pdf/2008.10898.pdf
    '''
    @staticmethod
    def getBatchSize(H, total_samples):
        b_size_str = get_experiment_option_str(H, "internal_sgd")
        b_size = 0
        if b_size_str == "full-gradient":
            b_size = total_samples
        elif b_size_str == "sgd-nice" or b_size_str == "sgd-multi" or b_size_str == "sgd-ind" or b_size_str == "sgd-us":
            tau = get_experiment_option_str(H, "tau")
            if tau.find("%") != -1:
                b_size = math.ceil(float(tau.replace("%", ""))/100.0 * total_samples)
            else:
                b_size = int(tau)
        else:
            raise Exception("PAGE can not work with that subsampling strategy for evaluate th. step size")

        return b_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D: int, total_clients: int,
                              grad_start: torch.Tensor) -> dict:

        compressor = compressors.initCompressor(args.client_compressor, D)

        state = {"x_prev": mutils.get_params(model),  # previous iterate
                 "test_ber_rv": 0.0,  # test_ber_rv = 0.0 will force fisrt iteration be a full gradient evaluation
                 "total_clients": total_clients,
                 "w": compressor.getW(),
                 "compressor_fullname" : compressor.fullName(),
                 "request_use_full_list_of_clients":  True,
                 "ck" : 1
                 }

        return state

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses,
                            use_steps_size_for_non_convex_case):

        if "hashed_step_size" in H:
            return H["hashed_step_size"]

        # Step size for non-convex case
        m = 1.0
        workers = H['total_clients']

        Li_all_clients_data_points = getLismoothForClientDataPoints(H, clients_responses)
        Li_all_clients = getLismoothForClients(H, clients_responses)
        L = getLsmoothGlobal(H, clients_responses)
        train_loader.dataset.set_client(None)
        n_client_samples = train_loader.dataset.n_client_samples

        #==============================================================================================================
        # Table C: Table and Calculations
        if H["args"].client_sampling_type == "uniform":
            # Nice
            b_size = H["args"].num_clients_per_round
            client_data_samples = train_loader.dataset.num_clients

            A = (client_data_samples - b_size) / (b_size * (client_data_samples - 1))
            wi = [1.0 / client_data_samples for i in range(client_data_samples)]
            B = (client_data_samples - b_size) / (b_size * (client_data_samples - 1))

            H.update({"A_m": A, "B_m": B, "wi_m": wi})

        elif H["args"].client_sampling_type == "poisson" or H["args"].client_sampling_type == 'poisson-no-empty':
            # Independent
            b_size = H["args"].num_clients_per_round
            client_data_samples = train_loader.dataset.num_clients

            pi = [H["args"].client_sampling_poisson] * client_data_samples
            #[b_size / client_data_samples] * client_data_samples

            # derived quantities
            z = sum([pv / (1.0 - pv) for pv in pi])
            A = 1.0 / z
            wi = [(pv / (1.0 - pv)) / z for pv in pi]
            B = 0.0

            H.update({"A_m": A, "B_m": B, "wi_m": wi, "pi_m": pi})

        elif H["args"].client_sampling_type == "uniform-sampling-with-replacement":
            b_size = H["args"].num_clients_per_round
            client_data_samples = train_loader.dataset.num_clients

            # Important sampling with uniform distribution
            qi = [1 / client_data_samples] * client_data_samples
            wi = [qi_item for qi_item in qi]

            # derived quantities
            A = 1.0 / b_size
            B = 1.0 / b_size

            H.update({"A_m": A, "B_m": B, "wi_m": wi, "qi_m": qi})

        elif H["args"].client_sampling_type == "important-sampling-with-replacement":
            # Important sampling
            b_size = H["args"].num_clients_per_round

            LiDp = getLismoothForClients(H, clients_responses)
            LiSum = sum(LiDp)
            qi = [Li / LiSum for Li in LiDp]

            # derived quantities
            A = 1.0 / b_size
            B = 1.0 / b_size

            wi = [qi_item for qi_item in qi]
            H.update({"A_m": A, "B_m": B, "wi_m": wi, "qi_m": qi})
        #==============================================================================================================

        b_size = PageAlgorithm.getBatchSize(H, n_client_samples)

        if True:
        #if has_experiment_option(H, "page_ab_synthetic"):
            p = 0.0
            if has_experiment_option(H, "use_optimal_p"):
                p = (b_size)/(b_size + n_client_samples)
            else:
                p = get_experiment_option_f(H, "p")

            H["p"] = p

            lambda_ = 0.0
            Lr = 0.0
            if H["args"].global_regulizer == "cvx_l2norm_square_div_2":
                lambda_ = H["args"].global_regulizer_alpha
                Lr = 1.0
            elif H["args"].global_regulizer == "none":
                lambda_ = 0.0
                Lr = 0.0
            elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
                lambda_ = H["args"].global_regulizer_alpha
                Lr = 2.0
            else:
                raise Exception("Selected regulatization is not supported")

            if has_experiment_option(H, "logregression"):
                L_minus = getLsmoothGlobal(H, clients_responses)          # L-smooth constant of objective

                L_plus_minus_w_for_client = []
                L_plus_w_for_clients = []

                num_clients = train_loader.dataset.num_clients
                n_client_samples = train_loader.dataset.n_client_samples

                for c in range(num_clients):
                    L_plus_minus_w_for_client_ = 0.0
                    for i in range(n_client_samples):
                        L_plus_minus_w_for_client_ += (Li_all_clients_data_points[c][i] ** 2) / n_client_samples * (1.0/n_client_samples) * (1.0/clients_responses.get(c)['client_state']['wi'][i])

                    L_plus_minus_w_for_client_ = L_plus_minus_w_for_client_ ** 0.5
                    L_plus_w_ = L_plus_minus_w_for_client_
                    L_plus_minus_w_for_client.append(L_plus_minus_w_for_client_)
                    L_plus_w_for_clients.append(L_plus_w_)
                #================================================================

                L_plus_minus_w = 0.0
                for i in range(num_clients):
                    L_plus_minus_w += (Li_all_clients[i] ** 2) / num_clients * (1.0/num_clients) * (1.0/H['wi_m'][i])

                L_plus_minus_w_all = L_plus_minus_w ** 0.5
                L_plus_w_all = L_plus_minus_w ** 0.5

            else:
                Ai = []
                Bi = []
                num_clients = train_loader.dataset.num_clients
                n_client_samples = train_loader.dataset.n_client_samples
                for i in range(num_clients):
                    d = train_loader.dataset.data[(i) * n_client_samples: (i + 1) * n_client_samples, ...]
                    Ai.append(d)
                    t = train_loader.dataset.targets[(i) * n_client_samples: (i + 1) * n_client_samples]
                    Bi.append(t)

                #L_minus  = torch.linalg.norm(Ai[0].T @ Ai[0], ord = 2) *2/n_client_samples + lambda_ * Lr
                #L_minus = getLismoothForClients(H, None)
                L_minus = getLsmoothGlobal(H, clients_responses)
                #L_plus_w = 0.0
                #for i in range(n_client_samples):
                #    ai = Ai[0][i,:]
                #    ai = ai.view(-1, 1)
                #    Zi = ( (ai @ ai.T) @ (ai @ ai.T) )
                #    ZiNorm = torch.linalg.norm(Zi, ord=2)
                #    L_plus_w += ZiNorm * 4 * 1/n_client_samples * (1.0/clients_responses.get(0)['client_state']['wi'][i])
                #L_plus_w = L_plus_w / n_client_samples
                #L_plus_w += 2 * (lambda_**2) * (Lr**2)

                #tmp = min(torch.real(torch.linalg.eigvals(Ai[0].T @ Ai[0]))) * 2/n_client_samples + lambda_
                #L_plus_minus_w = L_plus_w - tmp

                #=========================================================================================================
                # From https://arxiv.org/pdf/2110.03300.pdf, p.6 (for the case wi)
                #L_plus_minus_w_2_part_a = 0.0
                #L_plus_minus_w_2_part_b = 0.0

                #or i in range(n_client_samples):
                #    ai = Ai[0][i, :]
                #    ai = ai.view(-1, 1)
                #    Zi = 2*((ai @ ai.T))
                #    L_plus_minus_w_2_part_a += Zi @ Zi / n_client_samples
                #    L_plus_minus_w_2_part_b += Zi / n_client_samples

                #L_plus_minus_w_2 = (max(torch.real(torch.linalg.eigvals(L_plus_minus_w_2_part_a - (L_plus_minus_w_2_part_b @ L_plus_minus_w_2_part_b) ) )))
                # =========================================================================================================
                #=========================================================================================================

                # From https://arxiv.org/pdf/2110.03300.pdf, p.6 (for the case wi)
                L_plus_minus_w_for_client = []
                L_plus_w_for_clients = []

                for c in range(num_clients):
                    L_plus_minus_w_part_a = 0.0
                    L_plus_minus_w_part_b = 0.0

                    for i in range(n_client_samples):
                        ai = Ai[c][i, :]
                        ai = ai.view(-1, 1)
                        Zi = 2*((ai @ ai.T)) + (lambda_* Lr) * torch.eye(ai.shape[0]).to(ai.device)

                        L_plus_minus_w_part_a += Zi @ Zi / n_client_samples * (1.0/n_client_samples) * (1.0/clients_responses.get(c)['client_state']['wi'][i])
                        L_plus_minus_w_part_b += Zi / n_client_samples

                    L_plus_minus_w = (max(torch.real(torch.linalg.eigvals(L_plus_minus_w_part_a - (L_plus_minus_w_part_b @ L_plus_minus_w_part_b) ) )))
                    L_plus_w =  (max(torch.real(torch.linalg.eigvals(L_plus_minus_w_part_a))))
                    # =========================================================================================================
                    L_plus_minus_w = L_plus_minus_w ** 0.5
                    L_plus_w = L_plus_w ** 0.5
                    L_plus_minus_w_for_client.append(L_plus_minus_w)
                    L_plus_w_for_clients.append(L_plus_w)
                    # =========================================================================================================

                # From https://arxiv.org/pdf/2110.03300.pdf, p.6 (for the case wi)
                L_plus_minus_w_all = 0.0
                L_plus_w_all = 0.0
                L_plus_minus_w_part_a = 0.0
                L_plus_minus_w_part_b = 0.0

                for c in range(num_clients):
                    ai = Ai[c]
                    Zi = 2*((ai.T @ ai)) / ai.shape[0] + (lambda_* Lr) * torch.eye(ai.shape[1]).to(ai.device)

                    L_plus_minus_w_part_a += Zi @ Zi / num_clients * (1.0/num_clients) * (1.0/H['wi_m'][i])
                    L_plus_minus_w_part_b += Zi / num_clients

                L_plus_minus_w = (max(torch.real(torch.linalg.eigvals(L_plus_minus_w_part_a - (L_plus_minus_w_part_b @ L_plus_minus_w_part_b) ) )))
                L_plus_w =  (max(torch.real(torch.linalg.eigvals(L_plus_minus_w_part_a))))
                # =========================================================================================================
                L_plus_minus_w = L_plus_minus_w ** 0.5
                L_plus_w = L_plus_w ** 0.5
                L_plus_minus_w_all = L_plus_minus_w
                L_plus_w_all = L_plus_w
                # =========================================================================================================
            if has_experiment_option(H, "page_ab_synthetic"):
                A = [clients_responses.get(i)['client_state']['A'] for i in range(num_clients)]
                B = [clients_responses.get(i)['client_state']['B'] for i in range(num_clients)]
                H["hashed_A_compressor"] = A
                H["hashed_B_compressor"] = B
            else:
                A = [clients_responses.get(i)['client_state']['A'] for i in range(num_clients)]
                B = [0.0 for i in range(num_clients)]
                H["hashed_A_compressor"] = A
                H["hashed_B_compressor"] = B

            H["hashed_Li"] = Li_all_clients
            H["hashed_max_Li"] = max(Li_all_clients)
            H["hashed_L"] = L
            H["hashed_L_minus"] = L_minus
            H["hashed_Li_all_clients_data_points"] = Li_all_clients_data_points
            H["L_plus_minus_w_for_client"] = L_plus_minus_w_for_client
            H["L_plus_w_for_clients"] = L_plus_w_for_clients
            H["L_plus_minus_w_all"] = L_plus_minus_w_all
            H["L_plus_w_all"] = L_plus_minus_w_all

            #A_client_sampling =
        #H["hashed_L_plus_minus_w_2"] = L_plus_minus_w_2

            #if np.linalg.norm(clients_responses.get(0)['client_state']['wi'] - clients_responses.get(0)['client_state']['wi'][0]) < 1e-9:
            #    step_inv = L_minus + ( (1 - p) / p * ((A - B) * L_plus_w * L_plus_w + B * L_plus_minus_w_2 * L_plus_minus_w_2)) ** 0.5
            #    H["hashed_used_L_plus_minus_w"] = L_plus_minus_w_2
            #else:
            sinv_part_2_a = [ H['A_m'] / num_clients / H['wi_m'][i] + (1 - H["B_m"]) / num_clients for i in range(num_clients)]
            sinv_part_2_b = [ (A[i] - B[i]) * L_plus_w_for_clients[i]**2 for i in range(num_clients)]
            sinv_part_2_c = [B[i] * L_plus_minus_w_for_client[i]**2 for i in range(num_clients)]
            sinv_part_2_d = [ (H["A_m"] - H["B_m"]) * (L_plus_w_all**2) + H["B_m"]*(L_plus_minus_w_all**2) for i in range(num_clients)]

            step_inv = L_minus + ( (1-p)/p * 1.0/num_clients * sum( [sinv_part_2_a[i] * (sinv_part_2_b[i] + sinv_part_2_c[i]) + sinv_part_2_d[i] for i in range(num_clients)] ) )**0.5

        else:
            p = 0.0

            H["hashed_Li"] = Li_all_clients
            H["hashed_max_Li"] = max(Li_all_clients)
            H["hashed_L"] = L
            H["hashed_Li_all_clients_data_points"] = Li_all_clients_data_points

            Ltask = (np.mean(max(Li_all_clients) ** 2)) ** 0.5

            if has_experiment_option(H, "use_optimal_p"):
                p = (b_size)/(b_size + total_samples)
            else:
                p = get_experiment_option_f(H, "p")

            step_inv = Ltask * (1 + ( (1 - p)/(p*b_size) )**0.5 )


        H["hashed_step_size"] = 1.0/step_inv

        if has_experiment_option(H, 'stepsize_multiplier'):
            H["hashed_step_size"] *= get_experiment_option_f(H, 'stepsize_multiplier')

        return H["hashed_step_size"]

    @staticmethod
    def clientState(H:dict, clientId:int, client_data_samples:int, device:str)->dict:
        logger = Logger.get(H["run_id"])

        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)

        b_size = PageAlgorithm.getBatchSize(H, client_data_samples)

        state = {"ck": H["ck"],
                 "client_compressor" : compressor}

        sampling_schema = get_experiment_option_str(H, "internal_sgd")

        # Table C: Table and Calculations
        if sampling_schema == "sgd-nice":
            A = findRecentRecordAndRemoveFromHistory(H, clientId, "A")
            B = findRecentRecordAndRemoveFromHistory(H, clientId, "B")
            wi = findRecentRecordAndRemoveFromHistory(H, clientId, "wi")

            if A is not None and B is not None and wi is not None:
                state.update({"A": A, "B": B, "wi" : wi})
            else:
                A = (client_data_samples - b_size) / (b_size * (client_data_samples - 1))
                wi = [1.0/client_data_samples for i in range(client_data_samples)]
                B = (client_data_samples - b_size) / (b_size * (client_data_samples - 1))

                state.update({"A": A, "B": B, "wi" : wi})

        elif sampling_schema == "sgd-ind":
            A = findRecentRecordAndRemoveFromHistory(H, clientId, "A")
            B = findRecentRecordAndRemoveFromHistory(H, clientId, "B")
            wi = findRecentRecordAndRemoveFromHistory(H, clientId, "wi")
            pi = findRecentRecordAndRemoveFromHistory(H, clientId, "pi")

            if A is not None and B is not None and wi is not None and pi is not None:
                state.update({"A": A, "B": B, "wi" : wi, "pi" : pi})
            else:
                pi = [b_size/client_data_samples] * client_data_samples

                # derived quantities
                z = sum( [pv/(1.0-pv) for pv in pi])
                A = 1.0 / z
                wi = [(pv/(1.0-pv)) / z  for pv in pi]
                B = 0.0

                state.update({"A": A, "B": B, "wi" : wi, "pi" : pi})

        elif sampling_schema == "sgd-multi":
            A = findRecentRecordAndRemoveFromHistory(H, clientId, "A")
            B = findRecentRecordAndRemoveFromHistory(H, clientId, "B")
            wi = findRecentRecordAndRemoveFromHistory(H, clientId, "wi")
            qi = findRecentRecordAndRemoveFromHistory(H, clientId, "qi")
            qi_cdf = findRecentRecordAndRemoveFromHistory(H, clientId, "qi_cdf")

            inv_qi = findRecentRecordAndRemoveFromHistory(H, clientId, "inv_qi")

            if A is not None and B is not None and wi is not None and qi is not None:
                state.update({"A": A, "B": B, "wi" : wi, "qi" : qi, "qi_cdf" : qi_cdf, "inv_qi" : inv_qi})

            else:
                # Important sampling
                LiDp = getLismoothForClientDataPoints(H, None)
                LiDp = LiDp[clientId]

                if has_experiment_option(H, "sgd_multi_case_1"):
                    qi = [1/client_data_samples] * client_data_samples
                elif has_experiment_option(H, "sgd_multi_case_2"):
                    LiSum = sum(LiDp)
                    qi = [Li/LiSum for Li in LiDp]
                else:
                    raise Exception("Please specify one of the two regimes for SGD-MULTI: sgd_multi_case_1|sgd_multi_case_2")

                inv_qi = [1.0/qi_item for qi_item in qi]
                qi_cdf = model_funcs.construct_cdf(qi)

                # derived quantities
                A = 1.0 / b_size
                B = 1.0 / b_size

                wi = [qi_item for qi_item in qi]
                state.update({"A": A, "B": B, "wi" : wi, "qi" : qi, "qi_cdf" : qi_cdf, "inv_qi" : inv_qi})


        return state

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        if client_state["ck"] == 1:
            fApprox = evaluateGradient(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)
            client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
            return fApprox, grad_cur
        else:
            client_id = client_state["client_id"]
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            reconstruct_params = mutils.get_params(model)
            mutils.set_params(model, client_state["H"]["x_prev"])
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
            grad_prev = mutils.get_gradient(model)
            mutils.set_params(model, reconstruct_params)

            g_prev = client_state["H"]["g_prev"].to(client_state["device"])
            scale_client = 1.0
            sampling_type = client_state["H"]["args"].client_sampling_type

            if sampling_type == "poisson" or sampling_type == "poisson-no-empty":
                # sgd-ind
                clients_in_round = len(client_state["H"]["sampled_clients_in_round"])
                scale_client = clients_in_round / ( client_state["H"]['total_clients'] * client_state["H"]["args"].client_sampling_poisson)

            elif sampling_type == "uniform-sampling-with-replacement":
                scale_client = 1.0

            elif sampling_type == "important-sampling-with-replacement":
                # sgd-multi
                qi_m = client_state["H"]["qi_m"]
                scale_client = 1.0/len(qi_m) * 1/qi_m[client_id]

            g_next = g_prev + scale_client * (client_state["client_compressor"].compressVector(grad_cur - grad_prev))

            # Comments: server knows g_prev
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            return fApprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer, clients: int, model: torch.nn.Module, params_current: torch.Tensor, H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)

        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H["x_prev"] = paramsPrev
        H["g_prev"] = grad_server
        H["test_ber_rv"] = H['execution_context'].np_random.random()

        if "hashed_Li_all_clients_data_points" not in H:
            Li_all_clients_data_points = getLismoothForClientDataPoints(H, clients_responses)
            H["hashed_Li_all_clients_data_points"] = Li_all_clients_data_points

        p = H["p"]

        if H["test_ber_rv"] <= p:
            H["request_use_full_list_of_clients"] = True
            H["ck"] = 1
        else:
            H["request_use_full_list_of_clients"] = False
            H["ck"] = 0

        return H
#======================================================================================================================
def getLismoothForClientDataPoints(H, clients_responses):
    Li = np.array(H['Li_data_samples'])

    if H["args"].global_regulizer == "none":
        pass
    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
        Li = Li + (1.0 * H["args"].global_regulizer_alpha)
    elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
        Li = Li + (2.0 * H["args"].global_regulizer_alpha)

    return Li

def getLismoothForClients(H, clients_responses):
    Li = np.array(H['Li_all_clients'])

    if H["args"].global_regulizer == "none":
        pass
    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
        Li = Li + (1.0 * H["args"].global_regulizer_alpha)
    elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
        Li = Li + (2.0 * H["args"].global_regulizer_alpha)

    return Li

def getLsmoothGlobal(H, clients_responses):
    L = H['L_compute']

    if H["args"].global_regulizer == "none":
        pass
    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
        L = L + (1.0 * H["args"].global_regulizer_alpha)
    elif H["args"].global_regulizer == "noncvx_robust_linear_regression":
        L = L + (2.0 * H["args"].global_regulizer_alpha)

    return L

#======================================================================================================================
class MarinaAlgorithmPP:
    '''
    MARINA Algoritm [Gorbunov et al., 2021]: https://arxiv.org/abs/2102.07845
    '''
    @staticmethod
    def algorithmDescription():
        return { "paper" : "https://arxiv.org/abs/2102.07845" }

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for non-convex case
        m = 1.0
        workers_per_round = clients_in_round
        workers = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)

        # Ltask = (np.mean( (Li_all_clients) **2) )**0.5
        Ltask = (np.mean( max(Li_all_clients) **2) )**0.5 # Maybe hack by /2
        w = H["w"]
        p = (workers_per_round/workers)*1.0/(1+w)            # For RAND-K compressor
        r = workers_per_round

        step_1 = ((1 + 4*(1-p)*(1+w)/(p*workers))**0.5 - 1) / (2* (1-p) * (1+w)/(p*workers) * Ltask)
        step_2 = (-(1 + 4*(1-p)*(1+w)/(p*workers))**0.5 - 1) / (2* (1-p) * (1+w)/(p*workers) * Ltask)
        step_3 = 1.0 / (Ltask * (1 + ((1 - p) * (1 + w) / (p * workers_per_round)) ** 0.5))  # Theorem 4.1, p.37

        return step_3

    @staticmethod
    def initializeServerState(args:argparse.Namespace, model:torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor)->dict:
        compressor = compressors.initCompressor(args.client_compressor, D)

        state = {"x_prev" : mutils.get_params(model),             # previous iterate
                 "test_ber_rv" : 0.0,                             # test_ber_rv = 0.0 will force fisrt iteration be a full gradient evaluation
                 "num_clients_per_round" : args.num_clients_per_round,
                 "total_clients" : total_clients,
                 "w" : compressor.getW(),
                 "compressor_fullname": compressor.fullName()
                 }

        p = 1.0 / (1.0 + compressor.getW())
        p = p * args.num_clients_per_round / total_clients
        state.update({"p" : p})

        if state["test_ber_rv"] <= p:
            state["ck"] = 1
            state["request_use_full_list_of_clients"] = True
        else:
            state["ck"] = 0
            state["request_use_full_list_of_clients"] = False

        return state

    @staticmethod
    def clientState(H:dict, clientId:int, client_data_samples:int, device:str)->dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        state = {"p" : H["p"], "ck": H["ck"], "client_compressor" : compressor}
        return state

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        if client_state["ck"] == 1:
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)
            client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
            return fApprox, grad_cur
        else:
            client_id = client_state["client_id"]
            fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            reconstruct_params = mutils.get_params(model)
            mutils.set_params(model, client_state["H"]["x_prev"])
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
            grad_prev = mutils.get_gradient(model)
            mutils.set_params(model, reconstruct_params)

            g_prev = client_state["H"]["g_prev"].to(client_state["device"])
            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - grad_prev)

            # Comments: server knows g_prev
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            return fApprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer, clients: int, model: torch.nn.Module, params_current: torch.Tensor, H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H["g_prev"] = grad_server
        H["x_prev"] = mutils.get_params(model)

        H["test_ber_rv"] = H['execution_context'].np_random.random()
        if H["test_ber_rv"] <= H["p"]:
            H["ck"] = 1
            H["request_use_full_list_of_clients"] = True
        else:
            H["ck"] = 0
            H["request_use_full_list_of_clients"] = False

        return H
#======================================================================================================================
class SCAFFOLD:
    '''
    SCAFFOLD Algoritm [Karimireddy et al., 2020]: https://arxiv.org/abs/1910.06378
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {"c"  : torch.zeros(D).to(args.device),
                 "c0" : torch.zeros(D).to(args.device)}

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        # Compressors are not part of SCAFFOLD
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_ci = findRecentRecordAndRemoveFromHistory(H, clientId, 'ci')

        if last_ci is None:
            return {"ci" : H['c0'].detach().clone().to(device),
                    "client_compressor": compressor}
        else:
            return {"ci" : last_ci.to(device),                  #last_ci.detach().clone().to(device),
                    "client_compressor": compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        c  = client_state["H"]['c'].to(client_state["device"])
        ci = client_state['ci']

        if local_iteration_number[0] == 0:
            evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = False)
            c_plus = mutils.get_gradient(model)
            client_state['delta_c'] = client_state["client_compressor"].compressVector(c_plus - c)

            # send delta_c and delta_x for model which has the same dimension
            client_state['stats']['send_scalars_to_master'] += client_state['delta_c'].numel() # send change iterates
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        dy = grad_cur - ci + c

        return fAprox, dy

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        # x is updates as a part of general logic
        # here we will update c = c + sum(dc) * |S|/N

        obtained_model = clients_responses.get(0)
        dc = obtained_model['client_state']['delta_c'].to(paramsPrev.device)
        clients_num_in_round = len(clients_responses)

        for i in range(1, clients_num_in_round):
            client_model = clients_responses.get(i)
            dc += client_model['client_state']['delta_c'].to(paramsPrev.device)

        # Make dc is average of detla_c
        dc = dc / clients_num_in_round

        # Construct final delta step for update "c"
        dc = dc * float(clients_num_in_round)/float(H["total_clients"])
        H["c"] += dc
        return H
#=======================================================================================================================
class FRECON:
    '''
    FRECON Algoritm [Haoyu Zhao et al., 2021]: https://arxiv.org/abs/2112.13097
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # FRECON in non-convex case
        S = clients_in_round
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        w = compressor.getW()
        a = 1 / (1.0 + w)
        n = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)
        Lmax = max(Li_all_clients)
        step_size = 1.0/(Lmax * (1 + (10*(1+w)*(1+w)*n/S/S)**0.5))
        return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start) -> dict:

        state = {"h0": get_initial_shift(args, D, grad_start),
                 "g0": grad_start.detach().clone().to(args.device),
                 "h_prev": get_initial_shift(args, D, grad_start),
                 "g_server_prev": grad_start.detach().clone().to(args.device),
                 "x_prev": mutils.get_params(model)
                 }

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')

        #last_qi = findRecentRecordAndRemoveFromHistory(H, clientId, 'qi')
        # Drop qi
        #last_qi = None

        w = compressor.getW()
        alpha = 1 / (1.0 + w)

        if last_hi is None:
            return {"client_compressor" : compressor, "alpha" : alpha, "hi" : H['h0'].detach().clone().to(device)}
        else:
            return {"client_compressor" : compressor, "alpha" : alpha,
                    "hi" : last_hi.to(device)  #last_hi.detach().clone().to(device)
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:

        client_id = client_state["client_id"]
        fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)

        # Please select in GUI SGD-US or another estimator
        grad_cur = mutils.get_gradient(model)
        ui = client_state["client_compressor"].compressVector(grad_cur - client_state['hi'])
        #if client_state["H"]["current_round"] != 0:
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        client_state['hi'] = client_state['hi'] + client_state['alpha'] * ui

        #==============================================================================================================
        reconstruct_params = mutils.get_params(model)
        mutils.set_params(model, client_state["H"]["x_prev"])
        evaluateSgd(client_state, model, dataloader, criterion, is_rnn)
        grad_prev = mutils.get_gradient(model)
        mutils.set_params(model, reconstruct_params)
        #=============================================================================================================
        qi = client_state["client_compressor"].compressVector(grad_cur - grad_prev)
        #if client_state["H"]["current_round"] != 0:
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        client_state['qi'] = qi
        # =============================================================================================================
        return fApprox, ui

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        alpha = obtained_model["client_state"]['alpha']

        gs = wi * gi
        q_avg = wi * obtained_model['client_state']['qi']
        w_total = wi

        del obtained_model['client_state']['qi']

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']
            q_avg += wi * client_model['client_state']['qi']

            w_total += wi
            gs += wi * gi

            del client_model['client_state']['qi']

        gs = gs / w_total
        q_avg = q_avg / w_total
        u = gs

        h_prev = H['h_prev']

        # ===============================================================================================================
        if has_experiment_option(H, "lambda_"):
            lambda_ = get_experiment_option_f(H, "lambda_")
        elif has_experiment_option(H, "th_stepsize_noncvx") or has_experiment_option(H, "th_stepsize_cvx"):
            S = clients
            compressor = compressors.initCompressor(H["client_compressor"], H["D"])
            w = compressor.getW()
            n = H['total_clients']
            H["lambda_th"] = S / (2 * (1 + w) * n)
            lambda_ = S/( 2*(1+w) * n )
            get_logger(H).info(f"Used lambda is {lambda_}")
        #===============================================================================================================
        result = q_avg + (1.0 - lambda_) * H["g_server_prev"] + lambda_ * (u + h_prev)

        multipler_alpha = alpha * (clients/H['total_clients'])
        H['u_avg_update'] =  u
        H['alpha_update'] = multipler_alpha
        return result

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['h_prev'] = H['h_prev'] + H["alpha_update"] * H['u_avg_update']
        H["x_prev"] = paramsPrev
        H["g_server_prev"] = grad_server
        return H
#=======================================================================================================================
class COFIG:
    '''
    COFIG Algoritm [Haoyu Zhao et al., 2021]: https://arxiv.org/abs/2112.13097
    Assumption: \widetilda{S} = S, i.e. they are the same sets
    '''

    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for non-convex case
        S = clients_in_round
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])

        w = compressor.getW()
        a = 1 / (1.0 + w)

        if use_steps_size_for_non_convex_case:
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)
            step_size_1 = 1.0/( Lmax * 2 )
            step_size_2 = S / ( 5 * Lmax * (1 + w) * (H['total_clients'] ** (2.0/3.0)) )
            step_size_3 = S / ( 5 * Lmax * ((1 + w)**3.0/2.0) * (H['total_clients'] ** (0.5)) )

            return min(step_size_1, step_size_2, step_size_3)

        else:
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)

            step_size_1 = 1.0/( Lmax * (2 + 8 * (1 + w) / S) )
            step_size_2 = S / ( (1 + w) * (H['total_clients'] ** 0.5) )

            return min(step_size_1, step_size_2)

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()

        state = {"compressor_master" : cm,
                 "h0": get_initial_shift(args, D, grad_start),
                 "h_prev": get_initial_shift(args, D, grad_start),
                 }

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')
        alpha = 1.0 / (1.0 + compressor.getW())

        if last_hi is None:
            return {"client_compressor" : compressor, "alpha" : alpha, "hi" : H['h0'].detach().clone().to(device)}
        else:
            return {"client_compressor" : compressor, "alpha" : alpha,
                    "hi" : last_hi.to(device) #last_hi.detach().clone().to(device)
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:

        client_id = client_state["client_id"]
        fApprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)

        # Please select in GUI SGD-US or another estimator
        grad_cur = mutils.get_gradient(model)

        ui = client_state["client_compressor"].compressVector(grad_cur - client_state['hi'])
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        # Update hi (Experiment!)
        client_state['hi'] = client_state['hi'] + client_state['alpha'] * ui
        return fApprox, ui

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)

        alpha = obtained_model["client_state"]['alpha']

        gs = wi * gi

        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi

        gs = gs / w_total
        u = gs

        h_prev = H['h_prev']
        result =  u + h_prev

        multipler_alpha = alpha * (clients/H['total_clients'])
        H['u_avg_update'] =  gs
        H['alpha_update'] = multipler_alpha
        return result

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['h_prev'] = H['h_prev'] + H["alpha_update"] * H['u_avg_update']
        return H
#======================================================================================================================
class DIANA:
    '''
    DIANA Algoritm [Mishchenko et al., 2019]: https://arxiv.org/abs/1901.09269, https://arxiv.org/pdf/1904.05115.pdf
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        #===============================================================================================================
        workers = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)
        Lf = getLsmoothGlobal(H, clients_responses)

        if has_experiment_option(H, "dcgd_ab_synthetic"):
            if "hashed_Vpm" in H and "hashed_Vm" in H:
                pass
            else:
                assert has_experiment_option(H, "zero_out_b_in_sythetic")

                d = H["D"]
                n = len(H["sampled_clients_in_round"])

                if d >= n:
                    A_compressor = 1.0
                    B_compressor = 1.0
                else:
                    A_compressor = 1.0 - (n - d) / (n - 1)
                    B_compressor = 1.0 - (n - d) / (n - 1)

                Vm = Lf
                Vp = max(Li_all_clients)

                Ai = []

                num_clients = train_loader.dataset.num_clients
                n_client_samples = train_loader.dataset.n_client_samples

                for i in range(num_clients):
                    d = train_loader.dataset.data[(i) * n_client_samples: (i + 1) * n_client_samples, ...]
                    Ai.append((d.T @ d) * 2 / n_client_samples)

                    if H["args"].global_regulizer == "none":
                        pass
                    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
                        regulizer = H["args"].global_regulizer_alpha * torch.eye(Ai[-1].shape[0]).to(Ai[-1].device)
                        Ai[-1] += regulizer

                A = torch.mean(torch.stack(Ai), dim=0)
                A_avg_ = [((ai - A).T @ (ai - A)) for ai in Ai]
                A_avg = torch.mean(torch.stack(A_avg_), dim=0)

                L, V = torch.linalg.eig(A)
                L = torch.real(L)
                V = torch.real(V)
                L = L ** (-0.5)
                invAHalf = V @ torch.diag(L) @ torch.linalg.inv(V)
                Vpm = max(torch.real(torch.linalg.eigvals(invAHalf @ A_avg @ invAHalf)).tolist())

                A_avg_not_shift_ = [((ai).T @ (ai)) for ai in Ai]
                A_avg_not_shift = torch.mean(torch.stack(A_avg_not_shift_), dim=0)
                Vp_coorect = max(torch.real(torch.linalg.eigvals(invAHalf @ A_avg_not_shift @ invAHalf)).tolist())
                Vm_coorect = max(torch.real(torch.linalg.eigvals(A)).tolist())

                H["hashed_Vpm"] = Vpm
                H["hashed_Vp"] = Vp
                H["hashed_Vm"] = Vm

                H["hashed_Vp"] = Vp_coorect
                H["hashed_Vm"] = Vm_coorect

                H["hashed_V"] = V
                H["hashed_Li"] = Li_all_clients
                H["hashed_max_Li"] = max(Li_all_clients)
                H["hashed_L"] = Lf
                H["hashed_A_compressor"] = A_compressor
                H["hashed_B_compressor"] = B_compressor
        #===============================================================================================================

        # DIANA for non-convex case
        step_size = 0.0

        #===============================================================================================================
        if use_steps_size_for_non_convex_case:
            # For non-convex case
            m = 1.0
            workers_per_round = clients_in_round
            workers = H['total_clients']
            Ltask = getLsmoothGlobal(H, clients_responses)
            step_size = 1.0/(10*Ltask*(1 + H["w"]/workers)**0.5 * (m**(2.0/3.0) + H["w"] + 1))  # Th.4 of https://arxiv.org/pdf/1904.05115.pdf
        else:
            # For convex case
            compressor = compressors.initCompressor(H["client_compressor"], H["D"])
            w = compressor.getW()
            a = 1 / (1.0 + w)
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)
            step_size = 1.0 / (Lmax * (1 + 6*w/clients_in_round))                    # SGD-CTRL analysis for strongly-covnex case
        #===============================================================================================================

        if has_experiment_option(H, 'stepsize_multiplier'):
            step_size = step_size * get_experiment_option_f(H, 'stepsize_multiplier')

        return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        c = compressors.initCompressor(args.client_compressor, D)
        w = c.getW()
        alpha = 1.0 / (1.0 + w)

        state = {"h0"    : get_initial_shift(args, D, grad_start),
                 "h"     : get_initial_shift(args, D, grad_start),
                 "alpha" : alpha,
                 "w"     :w,
                 "compressor_fullname": c.fullName()
                }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)

        last_hi = findRecentRecordAndRemoveFromHistory(H, clientId, 'hi')

        if last_hi is None:
            return {"client_compressor" : compressor,
                    "hi" : H['h0'].detach().clone().to(device)}
        else:
            return {"client_compressor" : compressor,
                    "hi" : last_hi.to(device)          #last_hi.detach().clone().to(device)
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # In theory it's possible to perform compute without accessing "h" from master
        h = client_state['hi']
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        m_i = client_state["client_compressor"].compressVector(grad_cur - h)

        # Comments: server needs only obtain m_i
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance

        client_state['hi'] = client_state['hi'] + client_state['H']['alpha'] * m_i
        return fAprox, m_i

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)

        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total


        # Here gs is final gradient estimator without shift
        H['m'] = gs
        h = H['h']
        return h + gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        mk = H['m']
        H['h'] = H['h'] + H['alpha'] * mk
        return H
#======================================================================================================================
class EF21:
    '''
    EF21 Algoritm: "EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback", https://arxiv.org/abs/2106.05203
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        Li = getLismoothForClients(H, clients_responses)
        Ltask = getLsmoothGlobal(H, clients_responses)
        Ltilda = np.mean(Li**2)**0.5

        theta = 1 - (1 - a)**0.5
        beta = (1.0 - a) / (1 - (1 - a)**0.5)
        gamma = 1.0 / (Ltask + Ltilda * (beta/theta)**0.5)

        if has_experiment_option(H, 'stepsize_multiplier'):
            gamma = gamma * get_experiment_option_f(H, 'stepsize_multiplier')

        return gamma # Th.1, p.40 from EF21

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None,
                    "error_in_gradient": None
                   }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device), #last_g_prev.detach().clone().to(device)
                    "error_in_gradient": None
                   }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur
            # Not take into account communication at first round
            client_state['error_in_gradient'] = (grad_cur - grad_cur).cpu()

            return fAprox, grad_cur
        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - g_prev) * compressor_multiplier
            # In algorithm really we need only to send compressed difference between new gradient and previous gradient estimator
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['g_prev'] = g_next
            client_state['error_in_gradient'] = (grad_cur - g_next).cpu()

            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False
        return H
#======================================================================================================================
class EF21PP:
    '''
    EF21PP: "EF21 with Bells & Whistles: Practical Algorithmic Extensions of Modern Error Feedback" https://arxiv.org/abs/2110.03294, PP with Poisson sampling
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # Theoretical steps size for non-convex case
        p = H['args'].client_sampling_poisson
        assert p > 0.0

        pmax = p
        pmin = p

        rho = 1e-3
        s = 1e-3

        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        theta = 1 - (1 + s)*(1 - a)
        beta = (1.0 + 1.0/s) * (1 - a)
        thetap = rho * pmin + theta * pmax - rho - (pmax - pmin)
        Li = getLismoothForClients(H, clients_responses)

        B = (beta * p + (1 + 1.0/rho) * (1 - p)) * (np.mean(Li**2) )

        Ltask = getLsmoothGlobal(H, clients_responses)

        return 1.0/(Ltask + (B/thetap)**0.5) # Th.7, p.47 from EF21-PP

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None
                    }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device) #last_g_prev.detach().clone().to(device)
                    }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur
            # Not take into account communication at first round
            return fAprox, grad_cur
        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            g_next = g_prev + client_state["client_compressor"].compressVector(grad_cur - g_prev) * compressor_multiplier
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['g_prev'] = g_next
            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False

        error_from_clients = 0.0

        for i in range(len(clients)):
            client_model = clients_responses.get(i)
            error = client_model['client_state']['error_in_gradient']
            error_from_clients += error.norm().item()

        H["error_from_clients_prev"] = error_from_clients

        return H
#======================================================================================================================
class EF21AC:
    '''
    EF21PP with adaptive compressors
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        Li = getLismoothForClients(H, clients_responses)
        Ltask = getLsmoothGlobal(H, clients_responses)
        Ltilda = np.mean(Li**2)**0.5

        theta = 1 - (1 - a)**0.5
        beta = (1.0 - a) / (1 - (1 - a)**0.5)
        gamma = 1.0 / (Ltask + Ltilda * (beta/theta)**0.5)

        if has_experiment_option(H, 'stepsize_multiplier'):
            gamma = gamma * get_experiment_option_f(H, 'stepsize_multiplier')

        return gamma # Th.1, p.40 from EF21

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True,
                 "explicit_k" : 1.0
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressorExplicit(H["client_compressor"], H["D"], math.ceil(H["explicit_k"]))
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None,
                    "error_in_gradient" : None
                    }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device), #last_g_prev.detach().clone().to(device)
                    "error_in_gradient": None
                    }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur
            client_state['error_in_gradient'] = (grad_cur - grad_cur).cpu()
            # Not take into account communication at first round
            return fAprox, grad_cur
        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            alpha = (g_prev * grad_cur).sum().item() / ((g_prev.norm()**2).item())
            g_next = alpha * g_prev + client_state["client_compressor"].compressVector(grad_cur - alpha * g_prev) * compressor_multiplier
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['stats']['send_scalars_to_master'] += 1

            client_state['g_prev'] = g_next
            client_state['error_in_gradient'] = (g_next - grad_cur).cpu()

            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False
        error_from_clients = 0.0

        for i in range(len(clients)):
            client_model = clients_responses.get(i)
            error = client_model['client_state']['error_in_gradient']
            error_from_clients += error.norm().item()

        #cif "error_from_clients_prev" in H:
        #    error_from_clients_prev = H["error_from_clients_prev"]
        #    beta_increase = 1.1
        #    beta_decrease = 0.5
        #    alpha = 0.1

        #    if has_experiment_option(H, "beta_increase"):
        #        beta_increase = get_experiment_option_f(H, "beta_increase")
        #    if has_experiment_option(H, "beta_decrease"):
        #        beta_decrease = get_experiment_option_f(H, "beta_decrease")
        #   if has_experiment_option(H, "alpha"):
        #        alpha = get_experiment_option_f(H, "alpha")
        #
        #    if error_from_clients < alpha * error_from_clients_prev + 1e-6:
        #        # decrease K (increase error_from_clients in next round)
        #        H["explicit_k"] = max(H["explicit_k"] * beta_decrease, 1.0)
        #    else:
        #        # increase K (decrease error_from_clients in next round)
        #        H["explicit_k"] = min(H["explicit_k"] * beta_increase, H["D"])
        #
        H["error_from_clients_prev"] = error_from_clients

        return H
#======================================================================================================================
class EF21ACC:
    '''
    EF21PP with adaptive compressors
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        if compressor.isContractionCompressor():
            a = compressor.getAlphaContraction() # use alpha for contraction compressor
        elif compressor.isUnbiasedCompressor():
            a = 1/(1.0 + compressor.getW())      # use w for scaled unbiased compressor

        Li = getLismoothForClients(H, clients_responses)
        Ltask = getLsmoothGlobal(H, clients_responses)
        Ltilda = np.mean(Li**2)**0.5

        theta = 1 - (1 - a)**0.5
        beta = (1.0 - a) / (1 - (1 - a)**0.5)
        gamma = 1.0 / (Ltask + Ltilda * (beta/theta)**0.5)

        if has_experiment_option(H, 'stepsize_multiplier'):
            gamma = gamma * get_experiment_option_f(H, 'stepsize_multiplier')

        return gamma # Th.1, p.40 from EF21

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()
        state = {"compressor_master" : cm,
                 "x0" : mutils.get_params(model),
                 "request_use_full_list_of_clients": True,
                 "explicit_k" : 1.0
                 }
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressorExplicit(H["client_compressor"], H["D"], math.ceil(H["explicit_k"]))
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        last_g_prev = findRecentRecordAndRemoveFromHistory(H, clientId, 'g_prev')

        max_vectors_per_client = 1

        if has_experiment_option(H, 'max_vectors_per_client'):
            max_vectors_per_client = get_experiment_option_int(H, 'max_vectors_per_client')

        if last_g_prev is None:
            return {"client_compressor" : compressor,
                    "g_prev" : None,
                    "error_in_gradient" : None,
                    "max_vectors_per_client" : max_vectors_per_client
                    }
        else:
            return {"client_compressor" : compressor,
                    "g_prev" : last_g_prev.to(device), #last_g_prev.detach().clone().to(device)
                    "error_in_gradient": None,
                    "max_vectors_per_client" : max_vectors_per_client
                    }

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        # Compute g0 for a first iteration
        g_prev = client_state['g_prev']
        if g_prev is None:
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function=True)
            grad_cur = mutils.get_gradient(model)
            client_state['g_prev'] = grad_cur.unsqueeze(1)
            client_state['error_in_gradient'] = (grad_cur - grad_cur).cpu()
            # Not take into account communication at first round
            return fAprox, grad_cur

        else:
            # In theory it's possible to perform compute without accessing "h" from master
            fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
            grad_cur = mutils.get_gradient(model)

            g_prev = client_state['g_prev']

            G = g_prev.view(client_state['H']['D'], -1)

            compressor_multiplier = 1.0
            if not client_state["client_compressor"].isContractionCompressor():
                compressor_multiplier = 1.0/(1.0 + client_state["client_compressor"].getW())

            # alpha = (g_prev * grad_cur).sum().item() / ((g_prev.norm()**2).item())
            #beta = (g_prev * grad_cur).sum().item() / ((g_prev.norm()**2).item())

            U, S, Vh = torch.linalg.svd(G, full_matrices = False)
            alphaVec = Vh.T @ torch.linalg.inv( torch.diag(S) ) @ U.T @ grad_cur

            #alphaVec =  torch.linalg.inv(G.T @ G + torch.eye(G.shape[1]) * 1) @ G.T @ grad_cur

            g_next = (G @ alphaVec) + client_state["client_compressor"].compressVector(grad_cur - (G @ alphaVec)) * compressor_multiplier
            client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
            client_state['stats']['send_scalars_to_master'] += alphaVec.numel()

            client_state['g_prev'] = torch.cat( ( client_state['g_prev'], g_next.unsqueeze(1) ), dim = 1 )
            client_state['error_in_gradient'] = (g_next - grad_cur).cpu()

            max_vectors_per_client = client_state['max_vectors_per_client']

            if client_state['g_prev'].shape[1] > max_vectors_per_client:
                client_state['g_prev'] = client_state['g_prev'][..., -max_vectors_per_client:]

            return fAprox, g_next

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        # We compute it straightford.
        # In the paper the master uses g^t on server side and combine that with avg. of c_i^t

        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        H["request_use_full_list_of_clients"] = False
        error_from_clients = 0.0

        for i in range(len(clients)):
            client_model = clients_responses.get(i)
            error = client_model['client_state']['error_in_gradient']
            error_from_clients += error.norm().item()

        #cif "error_from_clients_prev" in H:
        #    error_from_clients_prev = H["error_from_clients_prev"]
        #    beta_increase = 1.1
        #    beta_decrease = 0.5
        #    alpha = 0.1

        #    if has_experiment_option(H, "beta_increase"):
        #        beta_increase = get_experiment_option_f(H, "beta_increase")
        #    if has_experiment_option(H, "beta_decrease"):
        #        beta_decrease = get_experiment_option_f(H, "beta_decrease")
        #   if has_experiment_option(H, "alpha"):
        #        alpha = get_experiment_option_f(H, "alpha")
        #
        #    if error_from_clients < alpha * error_from_clients_prev + 1e-6:
        #        # decrease K (increase error_from_clients in next round)
        #        H["explicit_k"] = max(H["explicit_k"] * beta_decrease, 1.0)
        #    else:
        #        # increase K (decrease error_from_clients in next round)
        #        H["explicit_k"] = min(H["explicit_k"] * beta_increase, H["D"])
        #
        H["error_from_clients_prev"] = error_from_clients

        return H



class DCGD:
    '''
    Distributed Compressed Gradient Descent Algoritm [Alistarh et al., 2017, Khirirat et al., 2018, Horvath et al., 2019]: https://arxiv.org/abs/1610.02132, https://arxiv.org/abs/1806.06573, https://arxiv.org/abs/1905.10988
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # Step size for convex case
        workers = H['total_clients']
        Li_all_clients = getLismoothForClients(H, clients_responses)
        Lf = getLsmoothGlobal(H, clients_responses)

        if has_experiment_option(H, "dcgd_ab_synthetic"):
            if "hashed_Vpm" in H and "hashed_Vm" in H:
                pass
            else:
                assert has_experiment_option(H, "zero_out_b_in_sythetic")

                d = H["D"]
                n = len(H["sampled_clients_in_round"])

                if d >= n:
                    A_compressor = 1.0
                    B_compressor = 1.0
                else:
                    A_compressor = 1.0 - (n - d) / (n - 1)
                    B_compressor = 1.0 - (n - d) / (n - 1)

                Vm = Lf
                Vp = max(Li_all_clients)

                Ai = []

                num_clients = train_loader.dataset.num_clients
                n_client_samples = train_loader.dataset.n_client_samples

                for i in range(num_clients):
                    d = train_loader.dataset.data[(i) * n_client_samples: (i + 1) * n_client_samples, ...]
                    Ai.append((d.T @ d) * 2 / n_client_samples)

                    if H["args"].global_regulizer == "none":
                        pass
                    elif H["args"].global_regulizer == "cvx_l2norm_square_div_2":
                        regulizer = H["args"].global_regulizer_alpha * torch.eye(Ai[-1].shape[0]).to(Ai[-1].device)
                        Ai[-1] += regulizer

                A = torch.mean(torch.stack(Ai), dim=0)
                A_avg_ = [((ai - A).T @ (ai - A)) for ai in Ai]
                A_avg = torch.mean(torch.stack(A_avg_), dim=0)

                L, V = torch.linalg.eig(A)
                L = torch.real(L)
                V = torch.real(V)
                L = L ** (-0.5)
                invAHalf = V @ torch.diag(L) @ torch.linalg.inv(V)
                Vpm = max(torch.real(torch.linalg.eigvals(invAHalf @ A_avg @ invAHalf)).tolist())

                A_avg_not_shift_ = [((ai).T @ (ai)) for ai in Ai]
                A_avg_not_shift  = torch.mean(torch.stack(A_avg_not_shift_), dim=0)
                Vp_coorect = max(torch.real(torch.linalg.eigvals(   invAHalf @ A_avg_not_shift @ invAHalf  )).tolist())
                Vm_coorect = max(torch.real(torch.linalg.eigvals(A)).tolist())

                H["hashed_Vpm"] = Vpm
                H["hashed_Vp"] = Vp
                H["hashed_Vm"] = Vm

                H["hashed_Vp"] = Vp_coorect
                H["hashed_Vm"] = Vm_coorect

                H["hashed_V"] = V
                H["hashed_Li"] = Li_all_clients
                H["hashed_max_Li"] = max(Li_all_clients)
                H["hashed_L"] = Lf
                H["hashed_A_compressor"] = A_compressor
                H["hashed_B_compressor"] = B_compressor

        if H['client_compressor'].find('permk') == 0:
            Vm  = H["hashed_Vm"]
            Vpm = H["hashed_Vpm"]
            Vp  = H["hashed_Vp"]
            A_compressor = H["hashed_A_compressor"]
            B_compressor = H["hashed_B_compressor"]

            V = Vm + 2 * (A_compressor - B_compressor) * Vp + 2 * B_compressor * Vpm
            step_size = 1.0 / V

        else:
            w = H["w"]
            wM = H["compressor_master"].getW()
            A = Lf + 2 * (wM + 1) * max(Li_all_clients * w/workers) + Lf * wM
            step_size = 1.0/A

        if has_experiment_option(H, 'stepsize_multiplier'):
            step_size = step_size * get_experiment_option_f(H, 'stepsize_multiplier')

        return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D: int, total_clients:int, grad_start:torch.Tensor) -> dict:
        cm = compressors.Compressor()
        cm.makeIdenticalCompressor()

        c = compressors.initCompressor(args.client_compressor, D)
        state = {"compressor_master" : cm,
                 "w" : c.getW(),
                 "compressor_fullname": c.fullName()
                 }

        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        return {"client_compressor" : compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        grad_compress = client_state["client_compressor"].compressVector(grad_cur)
        client_state['stats']['send_scalars_to_master'] += client_state["client_compressor"].last_need_to_send_advance
        return fAprox, grad_compress

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total
        grad_compress = H["compressor_master"].compressVector(gs)
        return grad_compress

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        compressor = H["compressor_master"]
        compressor.generateCompressPattern(H['execution_context'].np_random, paramsPrev.device, -1, H)
        return H

#=====================================================================================================================
class FedAvg:
    '''
    Algorithm FedAVG [McMahan et al., 2017]: https://arxiv.org/abs/1602.05629 
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {}
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        return {}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        client_state['stats']['send_scalars_to_master'] += grad_cur.numel()

        return fAprox, grad_cur

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        return H
#=======================================================================================================================
class Nastya:
    '''
    Algorithm NASTYA (under development)
    '''
    @staticmethod
    def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case):
        # TODO:
        # COPY FROM DIANA
        # DIANA for non-convex case

        if use_steps_size_for_non_convex_case:
            # For non-convex case
            m = 1.0
            workers_per_round = clients_in_round
            workers = H['total_clients']
            Ltask = getLsmoothGlobal(H, clients_responses)
            step_size = 1.0 / (10 * Ltask * (1 + H["w"] / workers) ** 0.5 * (
                        m ** (2.0 / 3.0) + H["w"] + 1))  # Th.4 of https://arxiv.org/pdf/1904.05115.pdf
            return step_size
        else:
            # For convex case
            compressor = compressors.initCompressor(H["client_compressor"], H["D"])
            w = compressor.getW()
            a = 1 / (1.0 + w)
            Li_all_clients = getLismoothForClients(H, clients_responses)
            Lmax = max(Li_all_clients)
            step_size = 1.0 / (Lmax * (1 + 4 * w / clients_in_round))  # SGD-CTRL analysis for strongly-covnex case
            return step_size

    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {}
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        return {}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)
        client_state['stats']['send_scalars_to_master'] += grad_cur.numel()

        return fAprox, grad_cur

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        return H
#=======================================================================================================================
class FedProx:
    '''
    Algorithm FedProx  [Li et al., 2018]: https://arxiv.org/abs/1812.06127
    '''
    @staticmethod
    def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, D:int, total_clients:int, grad_start:torch.Tensor) -> dict:
        state = {'wt': mutils.get_params(model)}
        return state

    @staticmethod
    def clientState(H: dict, clientId: int, client_data_samples:int, device:str) -> dict:
        compressor = compressors.initCompressor(H["client_compressor"], H["D"])
        compressor.generateCompressPattern(H['execution_context'].np_random, device, clientId, H)
        return {"client_compressor": compressor}

    @staticmethod
    def localGradientEvaluation(client_state: dict,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.dataloader.DataLoader,
                                criterion: torch.nn.modules.loss._Loss,
                                is_rnn: bool,
                                local_iteration_number: tuple) -> torch.Tensor:
        fAprox = evaluateSgd(client_state, model, dataloader, criterion, is_rnn, evaluate_function = True)
        grad_cur = mutils.get_gradient(model)

        opts = client_state['H']['execution_context'].experimental_options
        mu_prox = 1.0
        if "mu_prox" in opts:
            mu_prox = float(opts['mu_prox'])

        x_cur = mutils.get_params(model)
        wt = client_state['H']['wt'].to(client_state["device"])

        grad_cur += mu_prox * (x_cur - wt)

        client_state['stats']['send_scalars_to_master'] += grad_cur.numel()
        # assume sending 'wt' from master to clients is for free

        grad_cur = client_state["client_compressor"].compressVector(grad_cur)

        return fAprox, grad_cur

    @staticmethod
    def serverGradient(clients_responses: utils.buffer.Buffer,
                       clients: int,
                       model: torch.nn.Module,
                       params_current: torch.Tensor,
                       H: dict) -> torch.Tensor:
        clients_responses.waitForItem()
        obtained_model = clients_responses.get(0)
        wi = obtained_model['client_state']['weight']
        gi = params_current - obtained_model["model"].to(params_current.device)
        gs = wi * gi
        w_total = wi

        for i in range(1, clients):
            clients_responses.waitForItem()
            client_model = clients_responses.get(i)
            gi = params_current - client_model["model"].to(params_current.device)
            wi = client_model['client_state']['weight']

            w_total += wi
            gs += wi * gi
        gs = gs / w_total

        return gs

    @staticmethod
    def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, clients:dict, model:torch.nn.Module, paramsPrev:torch.Tensor, grad_server:torch.Tensor, H:dict)->dict:
        H['wt'] = mutils.get_params(model)
        return H
#=======================================================================================================================
def getImplClassForAlgo(algorithm):
    """
    Get imlementation class for specific algorithm

    Parameters:
        algorithm(str): Algorithm implementation

    Returns:
        class type with need interface methods
    """
    classImpl = None
    if algorithm == "page":
        classImpl = PageAlgorithm
    elif algorithm == "nastya":
        classImpl = Nastya
    elif algorithm == "marina":
        classImpl = MarinaAlgorithm
    elif algorithm == "dcgd":
        classImpl = DCGD
    elif algorithm == "fedavg":
        classImpl = FedAvg
    elif algorithm == "diana":
        classImpl = DIANA
    elif algorithm == "scaffold":
        classImpl = SCAFFOLD
    elif algorithm == "fedprox":
        classImpl = FedProx
    elif algorithm == "ef21":
        classImpl = EF21
    elif algorithm == "cofig":
        classImpl = COFIG
    elif algorithm == "frecon":
        classImpl = FRECON
    elif algorithm == "ef21-pp":
        classImpl = EF21PP
    elif algorithm == "ef21-ac":
        classImpl = EF21AC
    elif algorithm == "ef21-acc":
        classImpl = EF21ACC

    elif algorithm == "pp-marina":
        classImpl = MarinaAlgorithmPP
    else:
        raise ValueError(f"Please check algorithm. There is no implementation for '{algorithm}'")

    return classImpl

def getAlgorithmsList():
    """
    Get list of algorithms in order in which sorting happens in GUI
    """
    algoList = ["page", "dcgd", "ef21",
                "ef21-ac", "ef21-acc", "ef21-pp",
                "nastya", "marina", "cofig", "frecon", "fedavg", "diana", "scaffold", "fedprox", "pp-marina"]

    for a in algoList:
        assert getImplClassForAlgo(a) is not None

    return algoList

def initializeServerState(args: argparse.Namespace, model: torch.nn.Module, total_clients:int, grad_start: torch.Tensor) -> dict:
    """
    Initialize server state.

    Server state is a main source of information with various information, including:
     - 'x0' : start iterate. Be default it's a current position where model is
     - 'algorithm': string represenation of the used algorithms
     - 'history': history by rounds

    Parameters:
        args (argparse): Parsed command line for the python shell process
        model (nn.Module): Model under which server operate
        total_clients (int): Total number of clients in the experiment
        grad_start (torch.Tensor): Full gradient at starting point
    Returns:
        Returns initialize server state for specific algorithms args.algorithm
    """
    serverState = {}
    classImpl = getImplClassForAlgo(args.algorithm)

    D = mutils.number_of_params(model)
    serverState = classImpl.initializeServerState(args, model, D, total_clients, grad_start)

    serverState.update( {'algorithm' : args.algorithm,
                         'history' : {},
                         'D' : D,
                         'current_round' : 0,
                         'client_compressor' : args.client_compressor,
                         'run_id' : args.run_id,
                         'start_time': time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime()),
                         'server_state_update_time': time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime())
                         } )

    if 'x0' not in serverState:
        serverState.update( {'x0' : mutils.get_params(model) } )

    return serverState

def clientState(H:dict, clientId:int, client_data_samples:int, round:int, device:str)->dict:
    """
    Initialize client state.

    Clientstate is initialized from the scratch at each round for each selected client which is participates in optimization.
    If you want to initilize client via using it's previous state please use findRecentRecord() to find need information in server state.

     clientstate is a second source of information which helps to operate for clients
     - 'H' : referece to server state
     - 'algorithm': string represenation of the used algorithms
     - 'client_id': id of the client
     - 'device': target device for the client
     - 'round': number of round in which this state has been used
     - 'weight': weight used in the aggregation in serverGradient
     - 'stats': different statistics for a single client
     - 'seed': custom seed for pseudo-random generator for a client

    Parameters:
        H (dict): Server state
        clientId(int): Id of the client
        client_data_samples(int): Number of data points for client
        round(int): Number of the round
        device(str): Target device which should be used by the client for computations and store client state

    Returns:
        Initialized client state
    """
    classImpl = getImplClassForAlgo(H["algorithm"])

    clientState = classImpl.clientState(H, clientId, client_data_samples, device)
    if 'weight' not in clientState:
        clientState.update( {'weight' : 1.0} )


    clientState.update( {'H'         : H,
                         'algorithm' : H["algorithm"],
                         'client_id' : clientId,
                         'device'    : device,
                         'weight'    : 1.0,
                         'round'     : round,
                         'approximate_f_value' : [],
                         'seed' :   H['execution_context'].np_random.randint(2**31)
                         }
                      )

    stats = {"dataload_duration"  : 0.0,
             "inference_duration" : 0.0,
             "backprop_duration"  : 0.0,
             "full_gradient_oracles"    : 0,
             "samples_gradient_oracles" : 0,
             "send_scalars_to_master"   : 0
             }

    clientState.update({"stats" : stats})

    return clientState

def localGradientEvaluation(client_state:dict, model:torch.nn.Module, dataloader:torch.utils.data.dataloader.DataLoader, criterion:torch.nn.modules.loss._Loss, is_rnn:bool, local_iteration_number:tuple)->torch.Tensor:
    """
    Evalute local gradient for client.

    This API should implement optimization schema specific SGD estimator.

    Parameters:
        client_state (dict): state of the client which evaluate local gradient
        model(nn.Module): Initialized computation graph(model) locating currently in the interesting "x" position
        dataloader(torch.utils.data.dataloader.DataLoader): DataLoader mechanism to fectch data
        criterion(class): Loss function for minimization, defined as a "summ" of loss over train samples.
        is_rnn(bool): boolean flag which say that model is RNN

    Returns:
        Flat 1D SGD vector
    """
    classImpl = getImplClassForAlgo(client_state["algorithm"])
    return classImpl.localGradientEvaluation(client_state, model, dataloader, criterion, is_rnn, local_iteration_number)

def serverGradient(clients_responses : utils.buffer.Buffer,
                   clients : int,
                   model   : torch.nn.Module,
                   params_current : torch.Tensor,
                   H : dict)->torch.Tensor:
    """
    Evalute server gradient via analyzing local shifts from the clients.

    Parameters:
            clients_responses (Buffer): client responses. Each client transfers at least:
            'model' field in their response with last iterate for local model
            'optimiser' state of the local optimizer for optimizers with state
            'client_id' id of the client
            'client_state' state of the client

        clients(int): number of clients in that communication round. Invariant len(clients_responses) == clients
        model(torch.nn.Module): model which is locating currently in need server position.
        params_current(torch.Tensor): position where currently model is locating
        H(dict): server state

    Returns:
        Flat 1D SGD vector
    """
    if clients == 0:
        return torch.zeros_like(params_current)

    classImpl = getImplClassForAlgo(H["algorithm"])
    gs = classImpl.serverGradient(clients_responses, clients, model, params_current, H)

    # Need change for global optimizer for some time was gs=-gs. Currently no need.

    return gs

def theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses: utils.buffer.Buffer, use_steps_size_for_non_convex_case:bool):
    """
    Experimental method to evaluate theoretical step-size for non-convex L-smooth case

    Parameters:
        x_cur(torch.Tensor): current iterate
        grad_server(torch.Tensor): global gradient for which we're finding theoretical step-size
        H(dict): server state
        clients_in_round(int): number of clients particiapted in that round
        clients_responses: responses from clients

    Returns:
        Step size
    """
    classImpl = getImplClassForAlgo(H["algorithm"])
    step_size = classImpl.theoreticalStepSize(x_cur, grad_server, H, clients_in_round, train_loader, clients_responses, use_steps_size_for_non_convex_case)

    logger = Logger.get(H["args"].run_id)
    logger.info(f'Computed step size for H["algorithm"] is {step_size}')

    return step_size

def serverGlobalStateUpdate(clients_responses:utils.buffer.Buffer, model:torch.nn.Module, paramsPrev:torch.Tensor, round:int, grad_server:torch.Tensor, H:dict, numClients:int, sampled_clients_per_next_round)->dict:
    """
    Server global state update.

    Default update - include any states from previous round, but excluding model parameters, which maybe huge.

    Parameters:
        clients_responses (Buffer): client responses. Each client transfers at least:
        model(torch.nn.Module): model for sever, initialized with last position.
        paramsPrev(torch.Tensor): previous model parameters at the begininng of round
        round(int): number of the round
        grad_server(torch.Tensor): server's gradient.
        H(dict): server state
        numClients(int): number of clients in that round
        sampled_clients_per_next_round: future clients in a next round

    Returns:
        New server state.
    """
    clients = {}

    while len(clients_responses) < numClients:
        pass

    fvalues = []

    # Prune after communication round
    for item in clients_responses.items:
        assert item is not None
        # if item is None:
        #    continue

        item['client_state'].update({"optimiser" : item['optimiser']})
        item['client_state'].update({"buffers" : item['buffers']})

        del item['optimiser']
        del item['buffers']

        del item['model']

        if 'H' in item['client_state']:
            del item['client_state']['H']

        if 'client_compressor' in item['client_state']:
            del item['client_state']['client_compressor']

        clients[item['client_id']] = item
        fvalues += item['client_state']['approximate_f_value']

        # Remove sampled indicies in case of experiments with another SGD estimators inside optimization algorithms
        if 'iterated-minibatch-indicies' in item['client_state']:
            del item['client_state']['iterated-minibatch-indicies']

        if 'iterated-minibatch-weights' in item['client_state']:
            del item['client_state']['iterated-minibatch-weights']

        #if 'iterated-minibatch-indicies-full' in item['client_state']:
        #    del item['client_state']['iterated-minibatch-indicies-full']

    # This place and serverGlobalStateUpdate() are only place where H['history'] can be updated
    assert round not in H['history']

    fRunAvg = np.nan
    if len(fvalues) > 0:
        fRunAvg = np.mean(fvalues)

    H['history'][round] = {}
    H['history'][round].update({ "client_states" : clients,
                                  "grad_sgd_server_l2" : mutils.l2_norm_of_vec(grad_server),
                                  "approximate_f_avg_value" : fRunAvg,
                                  "x_before_round" : mutils.l2_norm_of_vec(paramsPrev)
                               })

    if has_experiment_option(H, "track_distance_to_solution"):
        xSolutionFileName = H["execution_context"].experimental_options["x_solution"]
        if "used_x_solution" not in H:
            with open(xSolutionFileName, "rb") as f:
                import pickle
                obj = pickle.load(f)
                assert len(obj) == 1
                Hsol = obj[0][1]
                used_x_solution = Hsol['xfinal']
                H["used_x_solution"] = used_x_solution

        H['history'][round]["distance_to_solution"] = mutils.l2_norm_of_vec(paramsPrev - H["used_x_solution"].to(paramsPrev.device))

    classImpl = getImplClassForAlgo(H["algorithm"])
    Hnew = classImpl.serverGlobalStateUpdate(clients_responses, clients, model, paramsPrev, grad_server, H)

    # Update server state
    Hnew.update( {'server_state_update_time' : time.strftime("%d %b %Y %H:%M:%S UTC%z", time.localtime())} )

    # Move various shifts client information into CPU
    move_client_state_to_host = Hnew['args'].store_client_state_in_cpu

    if move_client_state_to_host:
        client_states = Hnew['history'][round]["client_states"]
        for client_id in client_states:
            # Client will be sampled in a next round. It's not worthwhile to copy state to CPU, because in next round we will need back it to GPU (TODO: Maybe still better to move to CPU)
            # if client_id in sampled_clients_per_next_round:
            #    continue

            cs = client_states[client_id]['client_state']
            for k,v in cs.items():
                if torch.is_tensor(v):
                    if v.device != 'cpu':
                        v = v.cpu()
                    client_states[client_id]['client_state'][k] = v

    return  Hnew
#======================================================================================================================
