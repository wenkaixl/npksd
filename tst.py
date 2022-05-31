#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 22:18:30 2022

Implemtation for kernel-based two-sample tests via Maximum-mean-discrepancy (MMD)
code adapted and modified from freqopttest
Jitkrittum et al. Interpretable Distribution Features with Maximum Testing Power NIPS 2016
https://github.com/wittawatj/interpretable-test/tree/master/freqopttest

"""

from abc import abstractmethod
import numpy as np



class TwoSampleTest(object):
    """Abstract class for two sample tests."""

    def __init__(self, alpha=0.01):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        raise NotImplementedError()

    #@abstractmethod
    #def visual_test(self, tst_data):
    #    """Perform the test and plot the results. This is suitable for use 
    #    with IPython."""
    #    raise NotImplementedError()
    ##@abstractmethod
    #def pvalue(self):
    #   """Compute and return the p-value of the test"""
    #   raise NotImplementedError()

    #def h0_rejected(self):
    #    """Return true if the null hypothesis is rejected"""
    #    return self.pvalue() < self.alpha

class QuadMMDTest(TwoSampleTest):
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    - Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    - The code is based on a Matlab code of Arthur Gretton from the paper 
    A TEST OF RELATIVE SIMILARITY FOR MODEL SELECTION IN GENERATIVE MODELS
    ICLR 2016
    """

    def __init__(self, kernel, n_permute=400, alpha=0.01, use_1sample_U=False):
        """
        kernel: an instance of Kernel 
        n_permute: number of times to do permutation
        """
        self.kernel = kernel
        self.n_permute = n_permute
        self.alpha = alpha 
        self.use_1sample_U = use_1sample_U

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        alpha = self.alpha
        mmd2_stat = self.compute_stat(tst_data, use_1sample_U=self.use_1sample_U)

        X, Y = tst_data.xy()
        k = self.kernel
        repeats = self.n_permute
        list_mmd2 = QuadMMDTest.permutation_list_mmd2(X, Y, k, repeats)
        # approximate p-value with the permutations 
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat,
                'h0_rejected': pvalue < alpha, 'list_permuted_mmd2': list_mmd2}
        return results

    def compute_stat(self, tst_data, use_1sample_U=True):
        """Compute the test statistic: empirical quadratic MMD^2"""
        X, Y = tst_data.xy()
        nx = X.shape[0]
        ny = Y.shape[0]


        k = self.kernel
        mmd2, var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=False,
                use_1sample_U=self.use_1sample_U)
        return mmd2

    @staticmethod 
    def permutation_list_mmd2(X, Y, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        TODO: This is a naive implementation where the kernel matrix is recomputed 
        for each permutation. We might be able to improve this if needed.
        """
        return QuadMMDTest.permutation_list_mmd2_gram(X, Y, k, n_permute, seed)

    @staticmethod 
    def permutation_list_mmd2_gram(X, Y, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        """
        XY = np.vstack((X, Y))
        Kxyxy = k.eval(XY, XY)

        rand_state = np.random.get_state()
        np.random.seed(seed)

        nxy = XY.shape[0]
        nx = X.shape[0]
        ny = Y.shape[0]
        list_mmd2 = np.zeros(n_permute)

        for r in range(n_permute):
            #print r
            ind = np.random.choice(nxy, nxy, replace=False)
            # divide into new X, Y
            indx = ind[:nx]
            #print(indx)
            indy = ind[nx:]
            Kx = Kxyxy[np.ix_(indx, indx)]
            #print(Kx)
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]

            mmd2r, var = QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)
        return list_mmd2

    @staticmethod
    def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
        """
        Same as h1_mean_var() but takes in Gram matrices directly.
        """

        nx = Kx.shape[0]
        ny = Ky.shape[0]
        xx = ((np.sum(Kx) - np.sum(np.diag(Kx)))/(nx*(nx-1)))
        yy = ((np.sum(Ky) - np.sum(np.diag(Ky)))/(ny*(ny-1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = ((np.sum(Kxy) - np.sum(np.diag(Kxy)))/(nx*(ny-1)))
        else:
            xy = (np.sum(Kxy)/(nx*ny))
        mmd2 = xx - 2*xy + yy

        if not is_var_computed:
            return mmd2, None

        # compute the variance
        Kxd = Kx - np.diag(np.diag(Kx))
        Kyd = Ky - np.diag(np.diag(Ky))
        m = nx 
        n = ny
        v = np.zeros(11)

        Kxd_sum = np.sum(Kxd)
        Kyd_sum = np.sum(Kyd)
        Kxy_sum = np.sum(Kxy)
        Kxy2_sum = np.sum(Kxy**2)
        Kxd0_red = np.sum(Kxd, 1)
        Kyd0_red = np.sum(Kyd, 1)
        Kxy1 = np.sum(Kxy, 1)
        Kyx1 = np.sum(Kxy, 0)

        #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
        v[0] = 1.0/m/(m-1)/(m-2)*( np.dot(Kxd0_red, Kxd0_red ) - np.sum(Kxd**2) )
        #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
        v[1] = -( 1.0/m/(m-1) * Kxd_sum )**2
        #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
        v[2] = -2.0/m/(m-1)/n * np.dot(Kxd0_red, Kxy1)
        #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
        v[3] = 2.0/(m**2)/(m-1)/n * Kxd_sum*Kxy_sum
        #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
        v[4] = 1.0/n/(n-1)/(n-2)*( np.dot(Kyd0_red, Kyd0_red) - np.sum(Kyd**2 ) ) 
        #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...		       
        v[5] = -( 1.0/n/(n-1) * Kyd_sum )**2
        #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
        v[6] = -2.0/n/(n-1)/m * np.dot(Kyd0_red, Kyx1)

        #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
        v[7] = 2.0/(n**2)/(n-1)/m * Kyd_sum*Kxy_sum
        #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
        v[8] = 1.0/n/(n-1)/m * ( np.dot(Kxy1, Kxy1) - Kxy2_sum )
        #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
        v[9] = -2.0*( 1.0/n/m*Kxy_sum )**2
        #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
        v[10] = 1.0/m/(m-1)/n * ( np.dot(Kyx1, Kyx1) - Kxy2_sum )


        #%additional low order correction made to some terms compared with ICLR submission
        #%these corrections are of the same order as the 2nd order term and will
        #%be unimportant far from the null.

        #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
        #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
        varEst1st = 4.0*(m-2)/m/(m-1) * np.sum(v)

        Kxyd = Kxy - np.diag(np.diag(Kxy))
        #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
        #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
        varEst2nd = 2.0/m/(m-1) * 1/n/(n-1) * np.sum( (Kxd + Kyd - Kxyd - Kxyd.T)**2)

        #   varEst = varEst + varEst2nd;
        varEst = varEst1st + varEst2nd

        #   %use only 2nd order term if variance estimate negative
        if varEst<0:
            varEst =  varEst2nd
        return mmd2, varEst

    @staticmethod
    def h1_mean_var(X, Y, k, is_var_computed, use_1sample_U=True):
        """
        X: nxd numpy array 
        Y: nxd numpy array
        k: a Kernel object 
        is_var_computed: if True, compute the variance. If False, return None.
        use_1sample_U: if True, use one-sample U statistic for the cross term 
          i.e., k(X, Y).
        Code based on Arthur Gretton's Matlab implementation for
        Bounliphone et. al., 2016.
        return (MMD^2, var[MMD^2]) under H1
        """

        nx = X.shape[0]
        ny = Y.shape[0]

        Kx = k.eval(X, X)
        Ky = k.eval(Y, Y)
        Kxy = k.eval(X, Y)

        return QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha, reg=1e-3):
        """
        Return from the list the best kernel that maximizes the test power criterion.
        
        In principle, the test threshold depends on the null distribution, which 
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically 
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the MMD^2. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel 
        candidate.
        - reg: regularization parameter
        return: (best kernel index, list of test power objective values)
        """
        import time
        X, Y = tst_data.xy()
        n = X.shape[0]
        obj_values = np.zeros(len(list_kernels))
        for ki, k in enumerate(list_kernels):
            start = time.time()
            mmd2, mmd2_var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=True)
            obj = float(mmd2)/((mmd2_var + reg)**0.5)
            obj_values[ki] = obj
            end = time.time()
            print('(%d/%d) %s: mmd2: %.3g, var: %.3g, power obj: %g, took: %s'%(ki+1,
                len(list_kernels), str(k), mmd2, mmd2_var, obj, end-start))
        best_ind = np.argmax(obj_values)
        return best_ind, obj_values