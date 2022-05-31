#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:30:33 2021

model class including the explicit density models and approximated implicit model classes 
"""


from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

# import autograd.numpy as np
import numpy as np
import scipy 
import scipy.stats as stats
import math

import torch.autograd as autograd
from autograd import elementwise_grad
# import autograd
import torch
import torch.distributions as dists
import torch.optim as optim


import utils
import data
from objective import *

# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# import rpy2.robjects.numpy2ri

import warnings
import logging




### Explicit models, applicable for the unnormalised ones.
### adapted from kernel-gof code in Jitkrittum et. al, 2017 


class UnnormalizedDensity(object):
    """
    An abstract class of an unnormalized probability density function.  This is
    intended to be used to represent a model of the data for goodness-of-fit
    testing.
    """

    @abstractmethod
    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.
        X: n x d numpy array
        Return a one-dimensional numpy array of length n.
        """
        raise NotImplementedError()

    def log_normalized_den(self, X):
        """
        Evaluate the exact normalized log density. The difference to log_den()
        is that this method adds the normalizer. This method is not
        compulsory. Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this density.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function. Given an
        implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.
        X: n x d numpy array.
        Return an n x d numpy array of gradients.
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X)  
        X.requires_grad = True
        logprob = self.log_den(X)
#         # sum
        slogprob = torch.sum(logprob)
        Gs = torch.autograd.grad(slogprob, X, retain_graph=True, only_inputs=True)
# #        Gs = torch.autograd.grad(slogprob, X, retain_graph=False, only_inputs=True)
        G = Gs[0]
        # g = elementwise_grad(self.log_den)
        # # g = autograd.grad(self.log_den)
        # G = g(X)
        return G.numpy()

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

# end UnnormalizedDensity



class IsotropicNormal(UnnormalizedDensity):
    """
    Unnormalized density of an isotropic multivariate normal distribution.
    The torch implementation
    """
    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        self.mean = mean 
        self.variance = variance

    def log_den(self, X):
        if not torch.is_tensor(X):
            X = torch.Tensor(X)        
        mean = self.mean 
        variance = self.variance
        if not torch.is_tensor(mean):
            mean = torch.Tensor(mean)
        unden = (-torch.sum((X-mean)**2, 1)/(2.0*variance))
        return unden

    def log_normalized_den(self, X):
        d = self.dim()
        return stats.multivariate_normal.logpdf(X, mean=self.mean, cov=self.variance*np.eye(d))

    def get_datasource(self):
        return data.DSIsotropicNormal(self.mean, self.variance)

    def dim(self):
        return len(self.mean)




class TDistribution(UnnormalizedDensity):
    """
    Unnormalized density of multivariate student-t distribution.
    """
    def __init__(self, df):
        """
        df: degree of freedom 
        """
        self.df = df
        
    def log_den(self, X):
        if not torch.is_tensor(X):
            X = torch.Tensor(X)
        unden = -(self.df + 1.)/2. * torch.log (1 + torch.sum(X**2,axis=1))
        return unden[:,np.newaxis]
        

    def log_normalized_den(self, X):
        return stats.t.logpdf(X, df=self.df)

    def get_datasource(self):
        return data.DSTDistribution(self.df)

    def dim(self):
        return 1





class Normal(UnnormalizedDensity):
    """
    A multivariate normal distribution.
    """
    def __init__(self, mean, cov):
        """
        mean: a numpy array of length d.
        cov: d x d numpy array for the covariance.
        """
        self.mean = mean 
        self.cov = cov
        assert mean.shape[0] == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]
        E, V = np.linalg.eigh(cov)
        if np.any(np.abs(E) <= 1e-7):
            raise ValueError('covariance matrix is not full rank.')
        # The precision matrix
        self.prec = np.dot(np.dot(V, np.diag((1.0/E))), V.T)
        #print self.prec

    def log_den(self, X):
        mean = self.mean 
        if not torch.is_tensor(X):
            X = torch.Tensor(X)   
        if not torch.is_tensor(mean):
            mean = torch.Tensor(mean[np.newaxis,:])   
        X0 = X - mean
        prec = torch.Tensor(self.prec)
        X0prec = (X0 @ prec)
        unden = (-torch.sum(X0prec*X0, 1)/2.0)
        if len(unden.shape)==1:
            unden = unden[:,None]
        return unden

    def get_datasource(self):
        return data.DSNormal(self.mean, self.cov)

    def dim(self):
        return len(self.mean)

# end Normal



class IsoGaussianMixture(UnnormalizedDensity):
    """
    UnnormalizedDensity of a Gaussian mixture in R^d where each component 
    is an isotropic multivariate normal distribution.
    Let k be the number of mixture components.
    """
    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a one-dimensional length-k array of variances
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k, d = means.shape
        if k != len(variances):
            raise ValueError('Number of components in means and variances do not match.')

        if pmix is None:
            pmix = (torch.ones(k)/float(k))
            # pmix = (np.ones(k)/float(k))

        if torch.abs(torch.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def log_den(self, X):
        return self.log_normalized_den(X)

    def log_normalized_den(self, X):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        n = X.shape[0]
        den = torch.zeros(n, dtype=float)
        for i in range(k):
            norm_den_i = IsoGaussianMixture.normal_density(means[i],
                    variances[i], X)
            den = den + norm_den_i*pmix[i]
        return torch.log(den)

    @staticmethod
    def normal_density(mean, variance, X):
        """
        Exact density (not log density) of an isotropic Gaussian.
        mean: length-d array
        variance: scalar variances
        X: n x d 2d-array
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X)   
        if not torch.is_tensor(mean):
            mean = torch.Tensor(mean)   
        Z = np.sqrt(2.0*np.pi*variance)
        unden = torch.exp((-torch.sum((X-mean)**2.0, 1)/(2.0*variance)) )
        den = (unden/Z)
        assert len(den) == X.shape[0]
        if len(den.shape)==1:
            den=den[:,None]
        return den

    def get_datasource(self):
        return data.DSIsoGaussianMixture(self.means, self.variances, self.pmix)

    def dim(self):
        k, d = self.means.shape
        return d

# end class IsoGaussianMixture



class GaussianMixture(UnnormalizedDensity):
    """
    UnnormalizedDensity of a Gaussian mixture in R^d where each component 
    can be arbitrary. This is the most general form of a Gaussian mixture.
    Let k be the number of mixture components.
    """
    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a k x d x d numpy array containing a stack of k covariance
            matrices, one for each mixture component.
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k, d = means.shape
        if k != variances.shape[0]:
            raise ValueError('Number of components in means and variances do not match.')

        if pmix is None:
            pmix = (np.ones(k)/float(k))

        if np.abs(np.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def log_den(self, X):
        return self.log_normalized_den(X)

    def log_normalized_den(self, X):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        n = X.shape[0]

        den = np.zeros(n, dtype=float)
        for i in range(k):
            norm_den_i = GaussianMixture.multivariate_normal_density(means[i],
                    variances[i], X)
            den = den + norm_den_i*pmix[i]
        return np.log(den)

    @staticmethod
    def multivariate_normal_density(mean, cov, X):
        """
        Exact density (not log density) of a multivariate Gaussian.
        mean: length-d array
        cov: a dxd covariance matrix
        X: n x d 2d-array
        """
        
        evals, evecs = np.linalg.eigh(cov)
        cov_half_inv = evecs.dot(np.diag(evals**(-0.5))).dot(evecs.T)
    #     print(evals)
        half_evals = np.dot(X-mean, cov_half_inv)
        full_evals = np.sum(half_evals**2, 1)
        unden = np.exp(-0.5*full_evals)
        
        Z = np.sqrt(np.linalg.det(2.0*np.pi*cov))
        den = unden/Z
        assert len(den) == X.shape[0]
        return den

    def get_datasource(self):
        return data.DSGaussianMixture(self.means, self.variances, self.pmix)

    def dim(self):
        k, d = self.means.shape
        return d

# end GaussianMixture


class MixtureGaussian(UnnormalizedDensity):
    """
    p(x) is a mixture density of Gaussian components
    p(x) = \sum_{i=1}^K  \pi_i N(x | \mu_i, cov_i)
    for some \pi, mu, cov.
    """
    def __init__(self, means, precs, pi):
        """
        Let X be an n x d torch tensor.
        """
#        self.n_comps = n_comps
        self.pi = pi
        if not torch.is_tensor(means):
            self.means = torch.Tensor(means)  
        else:
            self.means = means
        if not torch.is_tensor(precs):
            self.precs = torch.Tensor(precs)  
        else:
            self.precs = precs


    def log_den(self, X):
        # unnormalized density
        return self.log_normalized_den(X)

    def log_normalized_den(self, X):
        n, d = X.shape

        # Pi: n x K 
        Pi = self.pi
        K = len(Pi)
        
        # Mu: 1 x K x d 
        Mu = self.means.view(1,K,d)#.repeat(n,1,1)
        assert len(Mu.shape) == 3
#        assert Mu.shape[0] == 1 
#        assert Mu.shape[1] == d
#        assert Mu.shape[2] == K

        
        # Precs: K x d x d 
        Precs = self.precs.to(X.device, X.dtype)
        if torch.any(torch.isnan(Precs)):
            warnings.warn('Var contains nan.')
#        S = torch.sqrt(Precs)


        # print('S.shape: {}'.format(S.shape))
        # print('Mu.shape: {}'.format(Mu.shape))
        # print('Y.shape: {}'.format(Y.shape))
        # broadcasting
        Z = (X.view(n, 1, d)- Mu) # n x  K x d
        if torch.any(torch.isnan(Z)):
            warnings.warn('Z contains nan.')
#        squared = torch.sum(Z**2, dim=2) # n x K 
            
#        import pdb; pdb.set_trace()
        squared = torch.einsum("nki, kij, nkj -> nk", Z, Precs, Z) # n x  K 
        assert squared.shape[0] == n
        assert squared.shape[1] == K
        const = 1.0/math.sqrt(2.0*math.pi)**d
        S = torch.det(Precs).view(1, K)
        if torch.any(torch.isnan(S)):
            warnings.warn('S contains nan.')
#        Sig_prod = torch.prod(S, dim=2) # n x K
        Sig_prod = S.repeat(n, 1) # n x K
        # TODO: make the computation robust to numerical errors. Perhaps use
        # logsumexp trick.
        E = torch.exp(-0.5*squared) * const * Pi * Sig_prod
        # sum over the K dimension
        log_prob = torch.log(torch.sum(E, dim=1))
        assert len(log_prob) == n
        return log_prob

    def dim(self):
        k, d = self.means.shape
        return d
    
    def get_datasource(self):
        return data.DSMixtureGaussian(self.means, self.precs, self.pi)




    
## Implicit model, to construct approximated Stein operators 



class ApproxModel(object):
    """
    An abstract class of an (unnormalized) model with approximated probability-related 
    functions. This is intended to be used to compute relevant terms for Non-parametric Stein operators.
    """

    @abstractmethod
    def log_den(self, X):
        """
        Evaluate this log of the unnormalized density on the n points in X.
        X: n x d numpy array
        Return a one-dimensional numpy array of length n.
        """
        raise NotImplementedError()

    def log_normalized_den(self, X):
        """
        Evaluate the exact normalized log density. The difference to log_den()
        is that this method adds the normalizer. This method is not
        compulsory. Subclasses do not need to override.
        """
        raise NotImplementedError()

    def get_datasource(self):
        """
        Return a DataSource that allows sampling from this density.
        May return None if no DataSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the approximate 
        log density at each of the n points in X. This is the score function. Given an
        implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.
        X: n x d numpy array.
        Return an numpy array of gradients.
        """
        samples = torch.Tensor(X)
        samples.requires_grad = True 
        # logp = -self.log_den_fun(samples).sum() #originally estimating the energy fun
        logp = self.log_den_fun(samples).sum()
        grad1 = autograd.grad(logp, samples, create_graph=False)[0]
        return grad1.detach().numpy()

# end UnnormalizedDensity




class ApproxLogDen(ApproxModel):
    """
    constuct the model class via estimating log density
    """
    def __init__(self, generator, estimator, n=1000, seed=1112):
        self.generator = generator
        self.seed = seed
        self.samples = generator.sample(n, seed)
        self.estimator = estimator
        self.log_den_fun = estimator.fit(self.samples.X)
        
    def log_den(self, X):
        X = torch.Tensor(X)
        return self.log_den_fun(X)
    
    def get_datasource(self):
        return self.generator

    def dim(self):
        return len(self.mean)


