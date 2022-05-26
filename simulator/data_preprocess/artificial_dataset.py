#!/usr/bin/env python3

import os

from .read_file_cache import cacheItemThreadUnsafe, cacheMakeKey, cacheGetItem, cacheHasItem
from .fl_dataset import FLDataset
import numpy as np

# Import PyTorch root package import torch
import torch
import math

def has_experiment_option(exec_ctx, name):
    """
    Check that experimental option is presented

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        True if option is present
    """
    return name in exec_ctx.experimental_options

def get_experiment_option_f(exec_ctx, name):
    """
    Get experimental option to carry experiments with algorithms

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        Value of requested value converted to float
    """
    return float(exec_ctx.experimental_options[name])

def get_experiment_option_int(exec_ctx, name):
    """
    Get experimental option to carry experiments with algorithms

    Parameters:
        H(dict): server state
        name(str): variable name
    Returns:
        Value of requested value converted to int
    """
    return int(exec_ctx.experimental_options[name])


class ArificialDataset(FLDataset):
    """
    Based FL class that loads H5 type data_preprocess.
    """
    def __init__(self, exec_ctx, args, train = None, client_id = None, transform = None, target_transform = None):
        """
        The constructor for a synthetic dataset.

        Parameters:
            exec_ctx: execution context from which random number generator should be use
            args: command-line argument with generation specification
            train(bool): True if we're in training mode
            client_id (int): make the view of the dataset as we work from the point of view of client client_id
            transform: input transformation applied to input attributes before feeding input into the computation of loss
            target_transform: output or label transformation applied to response variable before feeding input into the computation of loss
        """
        genSpecList = args.dataset_generation_spec.split(",")
        genSpec = {}
        for item in genSpecList:
            k,v = item.split(':')
            genSpec[k] = v

        self.transform = transform
        self.target_transform = target_transform

        self.num_clients = int(genSpec['clients'])
        self.n_client_samples = int(genSpec['samples_per_client'])

        d = int(genSpec['variables'])

        rows = self.num_clients * self.n_client_samples
        cols = d

        if train is None or train == True:
            pass
        else:
            pass

        xSoltMultiplier = 10.0
        xSolution = np.ones(d) * xSoltMultiplier
        b_perurbation = float(genSpec['b_perturb'])

        if True:
            # Returns a anumpy array filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
            Ai = exec_ctx.np_random.rand(rows, cols)
            U, S, Vt = np.linalg.svd(Ai, full_matrices=True)
            L = float(genSpec['l'])
            mu = float(genSpec['mu'])
            S = np.zeros((U.shape[1], Vt.shape[0]))
            len_s = min(S.shape[0], S.shape[1])
            if len_s > 1:
                for i in range(len_s):
                    S[i][i] = math.sqrt((L - mu) * float(i) / (len_s - 1) + mu)
            else:
                S[0][0] = math.sqrt(L)

            Ai = U @ S @ Vt
            Bi = Ai @xSolution
            Bi = Bi.reshape(-1, 1)
            Bi += b_perurbation * exec_ctx.np_random.rand(*(Bi.shape)) # random perturbation

            if has_experiment_option(exec_ctx, "zero_out_b_in_sythetic"):
                Bi *= 0.0

            # Step-1 construction of basic problem with specific charactersitcs L,\mu is DONE.

            # Extract homogenity level
            h1 = float(genSpec['homogeneous_level_1'])
            h2 = float(genSpec['homogeneous_level_2'])
            #==========================================================================================================
            # Prepare for modification
            # Extract Ai, Bi and compute mean of Ai and Bi
            ai = []
            bi = []

            for i in range(rows):
                ai_local = Ai[ (i) * 1 : (i+1)*1,...]
                bi_local = Bi[ (i) * 1 : (i+1)*1,...]
                ai.append(ai_local)
                bi.append(bi_local)

            a_avg = np.mean(np.stack(ai), axis = 0)
            b_avg = np.mean(np.stack(bi), axis = 0)

            # Contract ai and bi towards the center
            for i in range(rows):
                q = (i+1) * h2
                h_used = (abs(h1)) ** q

                ai[i] = ai[i] * (1.0 - h_used) + a_avg * h_used
                bi[i] = bi[i] * (1.0 - h_used) + b_avg * h_used

            # Reconstitute A and B
            for i in range(rows):
                Ai[ (i) * 1 : (i+1)*1,...] = ai[i]
                Bi[ (i) * 1 : (i+1)*1,...] = bi[i]

            # Extra scaling to have L and \mu specifically for function f(x) = 1/n * sum(a_i * x - b_i)**2
            # Needed for compute Li below
            #Ai *= math.sqrt(Ai.shape[0]/2.0)
            #Bi *= math.sqrt(Bi.shape[0]/2.0)
            # ==========================================================================================================
            # Final stack
            A = []
            B = []

            A.append(Ai)
            B.append(Bi)

            A = np.vstack(A)
            B = np.vstack(B)
            # ==========================================================================================================
            # Scale all Li in such way that max(Li) = L
            #A *= math.sqrt(A.shape[0] / 2.0)
            #B *= math.sqrt(A.shape[0] / 2.0)

            #self.compute_Li_for_linear_regression()
            #self.compute_Li_data_sample_for_linear_regression(args)

            #maxLi = max(self.Li_data_samples)
            #A *= math.sqrt(L/maxLi)
            #B *= math.sqrt(L/maxLi)

            # Project-like back into spaces of matrices with spectal norm between [\sqrt{\mu}, \sqrt{L}]
            if not False:
                U, S_orig, Vt = np.linalg.svd(A, full_matrices=True)
                S = np.zeros((U.shape[1], Vt.shape[0]))
                len_s = min(S.shape[0], S.shape[1])

                S[0][0] = math.sqrt(L)
                for i in range(1, len_s - 1):
                    S[i][i] = S_orig[i]
                    if S[i][i] > math.sqrt(L):
                        S[i][i] = math.sqrt(L)
                    elif S[i][i] < math.sqrt(mu):
                        S[i][i] = math.sqrt(mu)
                S[len_s - 1][len_s - 1] = math.sqrt(mu)

                A = U @ S @ Vt

                A *= math.sqrt(A.shape[0] / 2.0)
                B *= math.sqrt(A.shape[0] / 2.0)

            self.data = A
            self.targets = B
            self.targets = torch.Tensor(self.targets)
            self.data = torch.Tensor(self.data)

        self.set_client(client_id)

    def compute_Li_for_linear_regression(self):
        #=======================================================================================================
        # Compute L, Li for linear regression
        #=======================================================================================================
        A = self.data
        self.L = ((2/A.shape[0]) * torch.linalg.norm(A, 2)**2).item()
        self.Li_all_clients = []

        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples : (int(self.client_id) + 1) * self.n_client_samples, ...]
            Li = ((2/subdata.shape[0]) * torch.linalg.norm(subdata, 2) ** 2).item()
            self.Li_all_clients.append(Li)

        assert max(self.Li_all_clients) + 1.0e+3 >= self.L
        assert max(self.Li_all_clients) - 1.0e+3 <= self.L * self.num_clients

    def compute_Li_data_sample_for_linear_regression(self, args):
        #=======================================================================================================
        # Compute Li for linear regression for datasamples
        #=======================================================================================================
        self.Li_data_samples = []

        for c in range(self.num_clients):
            Li_data_samples_cur_client = []
            self.set_client(c)
            for j in range(self.n_client_samples):
                ai, bi = self[j]
                Li = (2) * (torch.linalg.norm(ai, 2) ** 2).item()
                Li_data_samples_cur_client.append(Li)
            self.Li_data_samples.append(Li_data_samples_cur_client)
            assert max(Li_data_samples_cur_client) + 1.0e+3 >= self.Li_all_clients[c]
            assert max(Li_data_samples_cur_client) - 1.0e+3 <= self.Li_all_clients[c] * self.n_client_samples

    def set_client(self, index=None):
        """
        Set pointer to client's data_preprocess corresponding to index. If index is none complete dataset as union of all datapoint will be observable by higher level
        Parameters:
            index(int): index of client.

        Returns:
            None
        """
        if index is None:
            self.client_id = None
            self.length = len(self.data)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.n_client_samples


    def load_data(self):
        """
        Explicit load all need datasets from the filesystem or cache for specific dataset instance.
        """
        pass

    def __getitem__(self, index):
        """
        Parameters:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = int(self.client_id) * self.n_client_samples + index
        img, target = self.data[actual_index], self.targets[actual_index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # TODO: If __getitem__ will always fetch object from the CPU memory. Suggestion use GPU memory or another GPU as a cache storage
        #return torch.from_numpy(img).float(), torch.from_numpy(target).float()
        # reference to objects from dataset (by reference)
        return img.detach(), target.detach()

    def __len__(self):
        return self.length
