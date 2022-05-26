#!/usr/bin/env python3
import threading
import wandb
from .logger import Logger

_is_init = False
_locker = threading.Lock()
_init_counter = 0

def initWandbProject(api_key, project, name, args):
    '''Thread safe way to initialize wand library, that is originally is not thread safe. If library has already been initialized this method only increase counter and does not perform any actual initialization.
    Parameters:
        api_key(str): personal api key for the use
        project(str): project name
        name(str): experiment name
        args: command line arguments
    Returns:
        Instance of the project from wandb library if library is initialized and None otherwise
    '''
    logger = Logger.get(args.run_id)

    global _is_init
    global _init_counter

    _locker.acquire()
    if _is_init:
        _init_counter += 1
        _locker.release()
        return None
    else:
        try:
            wandb.login(key = api_key)
            logger.info(f"Wandb login completed successfully")
            run = wandb.init(
                # project="federated_nas",
                project=project,
                name=name,
                config=args,
                reinit = True
            )
            _is_init = True
            _init_counter += 1
            _locker.release()
            return run

        except ValueError as err:
            logger.error(f"Ignore Wandb die tp problems with login into it: {str(err)}")
            _is_init = False
            _locker.release()
            return None

def finishProject(projectRun):
    '''Thread safe way to deinitialize wand library, that is originally is not thread safe. It will decrease counter, and once counter of library users will be zero it will perform final deinitialized.
    Parameters:
        projectRun: Instance of the project from wandb library
    Returns:
        None
    '''

    global _is_init
    global _init_counter

    _locker.acquire()
    _init_counter -= 1
    if _init_counter == 0:
        if projectRun:
            projectRun.finish()
        _is_init = False
    _locker.release()

def logStatistics(H, round):
    """ Log statistics from experiments and publish them via using wandb. To distinguish between experiments thr run_id is embeded into plot names.
    Parameters:
        H(dict): current server state
        round(int): current round
    Returns:
        None
    """

    global _is_init
    if _is_init == False:
        return

    item = H['history'][round]
    full_gradient_oracles = sum([v['client_state']['stats']['full_gradient_oracles'] for k, v in item["client_states"].items()])
    samples_gradient_oracles = sum([v['client_state']['stats']['samples_gradient_oracles'] for k, v in item["client_states"].items()])
    send_scalars_to_master = sum([v['client_state']['stats']['send_scalars_to_master'] for k, v in item["client_states"].items()])

    msg = {f"full_gradient_oracles({H['args'].run_id})": full_gradient_oracles,
           f"samples_gradient_oracles({H['args'].run_id})": samples_gradient_oracles,
           f"send_scalars_to_master({H['args'].run_id})": send_scalars_to_master,
           f"round({H['args'].run_id})": round
    }

    items_elements = ["full_gradient_norm_train", "x_before_round", "approximate_f_avg_value", "grad_sgd_server_l2",
                      "full_objective_value_train", "full_gradient_norm_val", "full_objective_value_val"]

    for elem in items_elements:
        if elem in item.keys():
            msg.update({f"{elem}({H['args'].run_id})" : item[elem]})

    _locker.acquire()
    wandb.log(msg)
    _locker.release()
