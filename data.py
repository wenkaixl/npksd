#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:50 2021

"""

from abc import ABCMeta, abstractmethod
import scipy.stats as stats
# import torch.autograd as autograd
import torch
# import torch.distributions as dists
import numpy as np
import utils 


# import warnings
# import logging



class Data(object):
    """
    Network Data class
    X: Pytorch tensor of adjacency matrices.
    """

    def __init__(self, X):
        """
        X: n x n x m Pytorch tensor for network adjacency matrices
        """
        self.X = X
        # self.X = X.detach()
        # self.X.requires_grad = False
        
    
    def dim(self):
        """Return the dimension of X."""
        dx = self.X.shape[1]
        return dx
    
    def sample_size(self):
        return self.X.shape[0]

    def n(self):
        return self.X.shape[0]

    def data(self):
        """Return adjacency matrix X"""
        return (self.X)

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. 
        Return (Data for tr and te)"""
        # torch tensors
        X = self.X
        n, dx = X.shape
        Itr, Ite = utils.tr_te_indices(n, tr_proportion, seed)
        # tr_data = Data(X[Itr].detach())
        # te_data = Data(X[Ite].detach())
        tr_data = Data(X[Itr])
        te_data = Data(X[Ite])
        return (tr_data, te_data)

    def subsample(self, n, seed=87, return_ind = False):
        """Subsample without replacement. Return a new Data. """
        if n > self.X.shape[0]:
            raise ValueError('n should not be larger than sizes of X')
        ind_x = utils.subsample_ind( self.X.shape[0], n, seed )
        if return_ind:
            return Data(self.X[ind_x, :]), ind_x
        else:
            return Data(self.X[ind_x, :])
        
    def clone(self):
        """
        Return a new Data object with a separate copy of each internal 
        variable, and with the same content.
        """
        nX = self.X.clone()
        return Data(nX)

    def __add__(self, data2):
        """
        Merge the current Data with another one.
        Create a new Data and create a new copy for all internal variables.
        """
        copy = self.clone()
        copy2 = data2.clone()
        nX = torch.vstack((copy.X, copy2.X))
        return Data(nX)
# end Data class        


class DataSource(object):
    """
    A source of data allowing resampling. Subclasses may prefix 
    class names with DS. 
    """

    @abstractmethod
    def sample(self, n, seed):
        """Return a Data. Returned result should be deterministic given 
        the input (n, seed)."""
        raise NotImplementedError()

    def dim(self):
       """
       Return the dimension of the data.  If possible, subclasses should
       override this. Determining the dimension by sampling may not be
       efficient, especially if the sampling relies on MCMC.
       """
       dat = self.sample(n=1, seed=3)
       return dat.dim()

#  end DataSource


class DSIsotropicNormal(DataSource):
    """
    A DataSource providing samples from a mulivariate isotropic normal
    distribution.
    """
    def __init__(self, mean, variance):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        assert len(mean.shape) == 1
        self.mean = mean 
        self.variance = variance

    def sample(self, n, seed=2):
        with utils.NumpySeedContext(seed=seed):
            d = len(self.mean)
            mean = self.mean
            variance = self.variance
            
            X = np.random.randn(n, d)*np.sqrt(variance) + mean
            return Data(X)



class DSNormal(DataSource):
    """
    A DataSource implementing a multivariate Gaussian.
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

    def sample(self, n, seed=3):
        with utils.NumpySeedContext(seed=seed):
            mvn = stats.multivariate_normal(self.mean, self.cov)
            X = mvn.rvs(size=n)
            if len(X.shape) ==1:
                # This can happen if d=1
                X = X[:, np.newaxis]
            return Data(X)



class DSIsoGaussianMixture(DataSource):
    """
    A DataSource implementing a Gaussian mixture in R^d where each component 
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

        if torch.abs(torch.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.variances = variances

    def sample(self, n, seed=29):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        sam_list = []
        with utils.NumpySeedContext(seed=seed):
            # counts for each mixture component 
            counts = np.random.multinomial(n, pmix, size=1)

            # counts is a 2d array
            counts = counts[0]

            # For each component, draw from its corresponding mixture component.            
            for i, nc in enumerate(counts):
                # Sample from ith component
                sam_i = np.random.randn(nc, d)*np.sqrt(variances[i]) + means[i]
                sam_list.append(sam_i)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        return Data(sample)

# end of class DSIsoGaussianMixture

class DSGaussianMixture(DataSource):
    """
    A DataSource implementing a Gaussian mixture in R^d where each component 
    is an arbitrary Gaussian distribution.
    Let k be the number of mixture components.
    """
    def __init__(self, means, variances, pmix=None):
        """
        means: a k x d 2d array specifying the means.
        variances: a k x d x d numpy array containing k covariance matrices,
            one for each component.
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

    def sample(self, n, seed=29):
        pmix = self.pmix
        means = self.means
        variances = self.variances
        k, d = self.means.shape
        sam_list = []
        with utils.NumpySeedContext(seed=seed):
            # counts for each mixture component 
            counts = np.random.multinomial(n, pmix, size=1)

            # counts is a 2d array
            counts = counts[0]

            # For each component, draw from its corresponding mixture component.            
            for i, nc in enumerate(counts):
                # construct the component
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
                cov = variances[i]
                mnorm = stats.multivariate_normal(means[i], cov)
                # Sample from ith component
                sam_i = mnorm.rvs(size=nc)
                sam_list.append(sam_i)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        return Data(sample)

# end of DSGaussianMixture


class DSMixtureGaussian(DataSource):
    """
   A DataSource implementing a Gaussian mixture in R^d where each component 
    is an arbitrary Gaussian distribution.
    Let k be the number of mixture components.
    """
    def __init__(self, means, precs, pmix=None):
        """
        means: a k x d torch tensor specifying the means.
        precs: a k x d x d numpy torch tensor containing k precision matrices,
            one for each component.
        pmix: a one-dimensional length-k array of mixture weights. Sum to one.
        """
        k, d = means.shape
        if k != precs.shape[0]:
            raise ValueError('Number of components in means and variances do not match.')

        if pmix is None:
            pmix = (np.ones(k)/float(k))

        if np.abs(torch.sum(pmix) - 1) > 1e-8:
            raise ValueError('Mixture weights do not sum to 1.')

        self.pmix = pmix
        self.means = means
        self.precs = precs

    def sample(self, n, seed=29):
        pmix = self.pmix#.detach().cpu().numpy()
        means = self.means#.detach().cpu().numpy()
        precs = self.precs#.detach().cpu().numpy()
        k, d = means.shape
        sam_list = []
        with utils.NumpySeedContext(seed=seed):
            # counts for each mixture component 
            counts = np.random.multinomial(n, pmix, size=1)
            # counts is a 2d array
            counts = counts[0]
            # For each component, draw from its corresponding mixture component.            
            for i, nc in enumerate(counts):
                # construct the component
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
                cov = np.linalg.inv(precs[i])
                mnorm = stats.multivariate_normal(means[i], cov)
                # Sample from ith component
                sam_i = mnorm.rvs(size=nc)
                sam_list.append(sam_i)
            sample = np.vstack(sam_list)
            assert sample.shape[0] == n
            np.random.shuffle(sample)
        # samples = torch.from_numpy(sample)
        samples = (sample)
        return Data(samples)

# end of DSGaussianMixture




class DSTrained(object):
    """
    A DataSource implementing a customised pre-trained models.
    """
    def __init__(self, model, d):
        """
        d: size of the network
        model: the pre-trained model that can generate samples 
        """
        self.d = d
        self.model = model
        
    def sample(self, n, seed=3, return_adj = False):
        with utils.NumpySeedContext(seed=seed):
            X = self.model.sample(n)
            return Data(X)
            

class DSSampled(object):
    """
    A DataSource implementing a customised pre-trained models in sample forms.
    """
    def __init__(self, model_samples):
        """
        d: size of the network
        model: the pre-trained model that can generate samples 
        """
        self.model_samples = model_samples
        
    def sample(self, n, seed=3, replace=True):
        with utils.NumpySeedContext(seed=seed):
            m = len(self.model_samples)        
            n = min(n, m)
            idx = np.random.choice(m, n, replace=replace)
            X = self.model_samples[idx]
            return Data(X)



class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        #if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0) 
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0) 
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        #Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = old_div((stdx + stdy),2.0)
        return mstd
        #xy = self.stack_xy()
        #return np.mean(np.std(xy, 0)**2.0)**0.5
    
    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same 
        for both X, Y. 
        
        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = util.subsample_ind( self.X.shape[0], n, seed )
        ind_y = util.subsample_ind( self.Y.shape[0], n, seed )
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    ### end TSTData class        



class DSGenerator(DataSource):
    """
    A DataSource providing samples from a given generative model
    """
    def __init__(self, generator):
        """
        mean: a numpy array of length d for the mean 
        variance: a positive floating-point number for the variance.
        """
        self.generator = generator

    def sample(self, n, seed=2):
        with utils.NumpySeedContext(seed=seed):
            X = self.generator.sample(n)
            return Data(X)




class DCGANgenerator_mnist:
    def __init__(self, G, latent_size, subsample_size=14):
        self.G = G
        self.latent_size = latent_size
        self.subsample_size = subsample_size
        
    def sample(self, batch_size, seed=12312):
        with utils.NumpySeedContext(seed=seed):
            latent_size = self.latent_size
            fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
            if torch.cuda.is_available():
                fixed_noise = fixed_noise.cuda()
            current = self.G(fixed_noise)
            current = current.detach().cpu().numpy()
            if self.subsample_size is not None:
                s = self.subsample_size
                l = int(28/s)
                n = batch_size
                current = np.reshape(current, (n, 28, 28))
                current = np.reshape(current, (n, s, l, s, l))
                current = current.mean(axis=(2, 4))
            gen_sample = np.reshape(current, (n, -1))
        return gen_sample

from torch.autograd import Variable

class GANgenerator_mnist:
    def __init__(self, G, latent_size, subsample_size=14):
        self.G = G
        self.latent_size = latent_size
        self.subsample_size = subsample_size
        
    def sample(self, batch_size, seed=12312):
        with utils.NumpySeedContext(seed=seed):
            latent_size = self.latent_size
            z = torch.randn(batch_size, latent_size)
            if torch.cuda.is_available():
                z = z.cuda()
            z = Variable(z)
            current = self.G(z)
            current = current.detach().cpu().numpy()
            if self.subsample_size is not None:
                s = self.subsample_size
                l = int(28/s)
                n = batch_size
                current = np.reshape(current, (n, 28, 28))
                current = np.reshape(current, (n, s, l, s, l))
                current = current.mean(axis=(2, 4))
            gen_sample = np.reshape(current, (n, -1))
        return gen_sample
    
    
class DCGANgenerator_cifar:
    def __init__(self, G, latent_size, subsample_size=16):
        self.G = G
        self.latent_size = latent_size
        self.subsample_size = subsample_size
        
    def sample(self, batch_size, seed=12312):
        with utils.NumpySeedContext(seed=seed):
            latent_size = self.latent_size
            fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
            if torch.cuda.is_available():
                fixed_noise = fixed_noise.cuda()
            current = self.G(fixed_noise)
            current = current.detach().cpu().numpy()
            if self.subsample_size is not None:
                s = self.subsample_size
                l = int(32/s)
                n = batch_size
                current = np.reshape(current, (n, 32, 32))
                current = np.reshape(current, (n, s, l, s, l))
                current = current.mean(axis=(2, 4))
            gen_sample = np.reshape(current, (n, -1))
        return gen_sample
