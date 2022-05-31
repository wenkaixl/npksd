#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:30 2021

functions and class objects for constructing various testing procedures for implicit models
"""

from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

import torch.autograd as autograd
import torch
import torch.distributions as dists
import torch.optim as optim
import typing
from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

# import freqopttest.tst as tst
import tst as tst

import time
import kernel
import utils
from data import TSTData

# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# import rpy2.robjects.numpy2ri


def bootstrapper_rademacher(n):
    """
    Produce a sequence of i.i.d {-1, 1} random variables.
    Suitable for boostrapping on an i.i.d. sample.
    """
    return 2.0*np.random.randint(0, 1+1, n)-1.0

def bootstrapper_multinomial(n):
    """
    Produce a sequence of i.i.d Multinomial(n; 1/n,... 1/n) random variables.
    This is described on page 5 of Liu et al., 2016 (ICML 2016).
    """
    import warnings
    warnings.warn('Somehow bootstrapper_multinomial() does not give the right null distribution.')
    M = np.random.multinomial(n, np.ones(n)/float(n), size=1) 
    return M.reshape(-1) - (1.0/float(n))

def bootstrapper_gaussian(n):
    """
    Produce a sequence of i.i.d standard gaussian random variables.
    """
    return np.random.randn(n)


class GofTest(object):
    """
    Abstract class for a goodness-of-fit test.
    """

    def __init__(self, p, alpha):
        """
        p: an UnnormalizedDensity
        alpha: significance level of the test
        """
        self.p = p
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """perform the goodness-of-fit test and return values computed in a dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }
        dat: an instance of Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()

# end of GofTest

class QuadMMDGof(GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with
    the MMD test of Gretton et al., 2012. 
    H0: the sample follows p
    H1: the sample does not follow p
    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, p_size=100, n_permute=400, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        k: an instance of Kernel
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha: significance level 
        seed: random seed
        """
        super(QuadMMDGof, self).__init__(p, alpha)
        # Construct the MMD test
        self.mmdtest = tst.QuadMMDTest(k, n_permute=n_permute, alpha=alpha)
        self.k = k
        self.seed = seed
        self.p_size = p_size
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(QuadMMDGof))


    def perform_test(self, dat):
        """
        dat: an instance of Data
        """
        with utils.ContextTimer() as t:
            seed = self.seed
            mmdtest = self.mmdtest
            p = self.p

            # Draw sample from p. #sample to draw is the same as that of dat
            ds = p.get_datasource()
            p_sample = ds.sample(self.p_size, seed=seed+12)

            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = TSTData(p_sample.data(), dat.data())
            # Test 
            results = mmdtest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        mmdtest = self.mmdtest
        p = self.p
        # Draw sample from p. #sample to draw is the same as that of dat
        ds = p.get_datasource()
        p_sample = ds.sample(dat.sample_size(), seed=self.seed)

        # Make a two-sample test data
        tst_data = TSTData(p_sample.data(), dat.data())
        s = mmdtest.compute_stat(tst_data)
        return s

        
# end QuadMMDGof

class MMDGofOpt(GofTest):
    """
    Goodness-of-fit test by drawing sample from the density p and test with the
    MMD test of Gretton et al., 2012. Optimize the kernel by the power
    criterion as in Jitkrittum et al., 2017. Need to split the data into
    training and test sets.
    H0: the sample follows p
    H1: the sample does not follow p
    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, n_permute=500, alpha=0.01, seed=28):
        """
        p: an instance of UnnormalizedDensity
        k: an instance of Kernel
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha: significance level 
        seed: random seed
        """
        super(MMDGofOpt, self).__init__(p, alpha)
        self.n_permute = n_permute
        self.seed = seed
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('%s test requires a density p which implements get_datasource(', str(QuadMMDGof))


    def perform_test(self, dat, candidate_kernels=None, return_mmdtest=False,
            tr_proportion=0.2, reg=1e-3):
        """
        dat: an instance of Data
        candidate_kernels: a list of Kernel's to choose from
        tr_proportion: proportion of sample to be used to choosing the best
            kernel
        reg: regularization parameter for the test power criterion 
        """
        with utils.ContextTimer() as t:
            seed = self.seed
            p = self.p
            ds = p.get_datasource()
            p_sample = ds.sample(dat.sample_size(), seed=seed+77)
            xtr, xte = p_sample.split_tr_te(tr_proportion=tr_proportion, seed=seed+18)
            # ytr, yte are of type data.Data
            ytr, yte = dat.split_tr_te(tr_proportion=tr_proportion, seed=seed+12)

            # training and test data
            tr_tst_data = TSTData(xtr.data(), ytr.data())
            te_tst_data = TSTData(xte.data(), yte.data())

            if candidate_kernels is None:
                # Assume a Gaussian kernel. Construct a list of 
                # kernels to try based on multiples of the median heuristic
                med = utils.meddistance(tr_tst_data.stack_xy(), 1000)
                list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 10) ) ) )
                list_gwidth.sort()
                candidate_kernels = [kernel.KGauss(gw2) for gw2 in list_gwidth]

            alpha = self.alpha

            # grid search to choose the best Gaussian width
            besti, powers = tst.QuadMMDTest.grid_search_kernel(tr_tst_data,
                    candidate_kernels, alpha, reg=reg)
            # perform test 
            best_ker = candidate_kernels[besti]
            mmdtest = tst.QuadMMDTest(best_ker, self.n_permute, alpha=alpha)
            results = mmdtest.perform_test(te_tst_data)
            if return_mmdtest:
                results['mmdtest'] = mmdtest

        results['time_secs'] = t.secs
        return results

    def compute_stat(self, dat):
        raise NotImplementedError('Not implemented yet.')


# end MMDGofOpt

class KernelSteinTest(GofTest):
    """
    Goodness-of-fit test using kernelized Stein discrepancy test of 
    Chwialkowski et al., 2016 and Liu et al., 2016 in ICML 2016.
    Mainly follow the details in Chwialkowski et al., 2016.
    The test statistic is n*V_n where V_n is a V-statistic.
    H0: the sample follows p
    H1: the sample does not follow p
    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, bootstrapper=bootstrapper_rademacher, alpha=0.01,
            n_simulate=500, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        bootstrapper: a function: (n) |-> numpy array of n weights 
            to be multiplied in the double sum of the test statistic for generating 
            bootstrap samples from the null distribution.
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(KernelSteinTest, self).__init__(p, alpha)
        self.k = k
        self.bootstrapper = bootstrapper
        self.n_simulate = n_simulate
        self.seed = seed

    def perform_test(self, dat, return_simulated_stats=False, return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with utils.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = n*np.mean(H)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with utils.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                   W = self.bootstrapper(n)
                   # n * [ (1/n^2) * \sum_i \sum_j h(x_i, x_j) w_i w_j ]
                   boot_stat = W.dot(H.dot((W/float(n))))
                   # This is a bootstrap version of n*V_n
                   sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results


    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x n
        gram_glogp = grad_logp@(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        # V-statistic
        stat = n*np.mean(H)
        if return_ustat_gram:
            return stat, H
        else:
            return stat

        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 't3: {0}'.format(t3)
        #print 't4: {0}'.format(t4)

# end KernelSteinTest





class SteinMCTest(GofTest):
    """
    Stein Monte Carlo test for goodness-of-fit using 
    kernelized Stein discrepancy test for implicit models
    
    H0: p generates same distribution as the sample
    H1: p does not generate same distribution as the sample
    p is specified to the constructor in the form of an UnnormalizedDensity.
    Monte Carlo samples are generated to simulate the null
    """

    def __init__(self, p, k, alpha=0.01, n_simulate=100, n_gen=None, seed=11, B=None):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object for RKHS kernel
        alpha: significance level 
        n_simulate: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        # super(self).__init__(p, alpha)
        self.p = p
        self.ds = p.get_datasource() #datasource that getting 
        self.k = k
        self.alpha = alpha
        self.n_simulate = n_simulate
        self.n_gen = n_gen
        self.seed = seed
        self.B = B

    def perform_test(self, dat, return_simulated_stats=False):
        """
        dat: a instance of Data
        """
        with utils.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            if self.n_gen is None:
                X = dat.data()
                n_gen = X.shape[0]
            else:
                n_gen = self.n_gen

            
            if self.B is None:
                B_list = None
            else:
                d = X.shape[1]    
                B_list = np.random.choice(d, self.B)
            
            test_stat = self.compute_stat(dat, return_ustat_gram=False, B_list=B_list)
            # monte carlo based test for implicit models 
            sim_stats = np.zeros(n_simulate)
            
            for i in range(n_simulate):
                # with utils.NumpySeedContext(seed=self.seed + i*1310):
                W = self.ds.sample(n_gen, seed=self.seed + i*1310)
                sim_stats[i] = self.compute_stat(W, return_ustat_gram=False, B_list=B_list)
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 'n_gen': n_gen
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
            
        return results


    def compute_stat(self, dat, return_ustat_gram=False, B_list=None):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        return_ustat_gram: If True, then return the n x n matrix used to
            compute the statistic (by taking the mean of all the elements)
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x n
        gram_glogp = grad_logp@(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        stat = n*np.mean(H)
        if B_list is not None:
            H = H[B_list, B_list]
            B = len(B_list)
            stat = B*np.mean(H)
        # V-statistic
        if return_ustat_gram:
            return stat, H
        else:
            return stat


# end SteinMCTest


