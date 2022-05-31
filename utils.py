#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:03:51 2021

"""
import numpy as np
import time 

class NumpySeedContext(object):
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)

# end NumpySeedContext


class ContextTimer(object):
    """
    A class used to time an execution of a code snippet. 
    Use it with with .... as ...
    For example, 
        with ContextTimer() as t:
            # do something 
        time_spent = t.secs
    From https://www.huyng.com/posts/python-performance-analysis
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start 
        if self.verbose:
            print('elapsed time: %f ms' % (self.secs*1000))


def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*X.dot(Y.T) + sy[np.newaxis, :] 
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D

def dist2_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance **squared** matrix of size
    X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X**2, 1)
    sy = np.sum(Y**2, 1)
    D2 =  sx[:, np.newaxis] - 2.0*np.dot(X, Y.T) + sy[np.newaxis, :] 
    return D2

def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.
    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and 
        there are more slightly more 0 than 1. In this case, the m
    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)



def is_real_num(X):
    """return true if x is a real number. 
    Work for a numpy array as well. Return an array of the same dimension."""
    def each_elem_true(x):
        try:
            float(x)
            return not (np.isnan(x) or np.isinf(x))
        except:
            return False
    f = np.vectorize(each_elem_true)
    return f(X)
    

def tr_te_indices(n, tr_proportion, seed=9282 ):
    """Get two logical vectors for indexing train/test points.
    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=32):
    """
    Return a list of indices to choose k out of n without replacement
    """
    with NumpySeedContext(seed=seed):
        ind = np.random.choice(n, k, replace=False)
    return ind

def subsample_rows(X, k, seed=29):
    """
    Subsample k rows from the matrix X.
    """
    n = X.shape[0]
    if k > n:
        raise ValueError('k exceeds the number of rows.')
    ind = subsample_ind(n, k, seed=seed)
    return X[ind, :]
    

def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to the data X (n x d) and draw J points 
    from the fit. 
    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to construct 
        a new covariance matrix before drawing samples. Useful to shrink the spread 
        of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        mean_x = np.mean(X, 0)
        cov_x = np.cov(X.T)
        if d==1:
            cov_x = np.array([[cov_x]])
        [evals, evecs] = np.linalg.eig(cov_x)
        evals = np.maximum(0, np.real(evals))
        assert np.all(np.isfinite(evals))
        evecs = np.real(evecs)
        shrunk_cov = evecs.dot(np.diag(evals**eig_pow)).dot(evecs.T) + reg*np.eye(d)
        V = np.random.multivariate_normal(mean_x, shrunk_cov, J)
    return V

def bound_by_data(Z, Data):
    """
    Determine lower and upper bound for each dimension from the Data, and project 
    Z so that all points in Z live in the bounds.
    Z: m x d 
    Data: n x d
    Return a projected Z of size m x d.
    """
    n, d = Z.shape
    Low = np.min(Data, 0)
    Up = np.max(Data, 0)
    LowMat = np.repeat(Low[np.newaxis, :], n, axis=0)
    UpMat = np.repeat(Up[np.newaxis, :], n, axis=0)

    Z = np.maximum(LowMat, Z)
    Z = np.minimum(UpMat, Z)
    return Z


def one_of_K_code(arr):
    """
    Make a one-of-K coding out of the numpy array.
    For example, if arr = ([0, 1, 0, 2]), then return a 2d array of the form 
     [[1, 0, 0], 
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 1]]
    """
    U = np.unique(arr)
    n = len(arr)
    nu = len(U)
    X = np.zeros((n, nu))
    for i, u in enumerate(U):
        Ii = np.where( np.abs(arr - u) < 1e-8 )
        #ni = len(Ii)
        X[Ii[0], i] = 1
    return X



def randn(m, n, seed=3):
    with NumpySeedContext(seed=seed):
        return np.random.randn(m, n)

def matrix_inner_prod(A, B):
    """
    Compute the matrix inner product <A, B> = trace(A^T * B).
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    return A.reshape(-1).dot(B.reshape(-1))

def get_classpath(obj):
    """
    Return the full module and class path of the obj. For instance, 
    kgof.density.IsotropicNormal
    Return a string.
    """
    return obj.__class__.__module__ + '.' + obj.__class__.__name__

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def sigmoid(X):
     return 1./(1+np.exp(-(X)))
 

def logit(X):
     return np.log(1./(1./(X) - 1.))




def subsample_mnist(current, resample=7):
    n = current.shape[0]
    r = resample
    s = int(28/resample)
    current = np.reshape(current, (n, r,s,r,s))
    current = current.mean(axis=(2, 4))
    return np.reshape(current, (n, int(r*r)))
