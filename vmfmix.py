# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: vmfmix.py
@Time: 2019-11-06 10:06
@Desc: vmfmix.py
"""
import numpy as np
import random
import scipy.io as scio

from scipy import sparse
from scipy.special import logsumexp


class VMFMixture:

    def __init__(self, n_components=10, maxiter=100, Regularize=1e-3, tol=1e-12):

        self.n_components = n_components
        self.maxiter = maxiter
        self.Regularize = Regularize
        self.tol = tol

    def init(self, x):

        (self.N, self.D) = x.shape
        bestlabel = []
        sumD = np.zeros((self.n_components))
        bCon = False
        maxit = 100
        reps = 10

        # init params by spherical kmeans
        for t in range(1, reps):
            mu = x[randsample(self.N, self.n_components), :]
            last = 0
            label = np.array([1])
            it = 0
            while np.any(label != last) and it < maxit:
                last = label
                simMat = x.dot(mu.T)
                val = np.max(simMat, 1)
                label = np.argmax(simMat, 1)
                ll = np.unique(label)
                if len(ll) < self.n_components:
                    misscluster = [i for i in range(self.n_components)]
                    misscluster = np.delete(misscluster, ll)
                    missNum = len(misscluster)
                    idx = np.argsort(val)
                    label[idx[:missNum]] = misscluster

                E = sparse.coo_matrix((np.ones(self.N), ([i for i in range(self.N)], label)), shape=(self.N, self.n_components))
                mu = E.dot(sparse.spdiags(1/np.sum(E, 0), 0, self.n_components, self.n_components)).T.dot(x)
                centernorm = np.sqrt(np.sum(mu**2, 1))[:, np.newaxis]
                mu = mu / np.repeat(centernorm, self.D, 1)

                it = it + 1

            if it < maxit:
                bCon = True

            if len(bestlabel) == 0:

                bestlabel = label
                bestcenter = mu
                if reps > 1:
                    if np.any(label != last):
                        simMat = x.dot(mu.T)
                    D = 1-simMat
                    for k in range(self.n_components):
                        sumD[k] = np.sum(D[label == k, k], 0)

                    bestsumD = sumD
                    bestD = D
            else:
                if np.any(label != last):
                    simMat = x.dot(mu.T)
                D = 1 - simMat
                for k in range(self.n_components):
                    sumD[k] = np.sum(D[label == k, k], 0)

                if np.sum(sumD) < np.sum(bestsumD):
                    bestlabel = label
                    bestcenter = mu
                    bestsumD = sumD
                    bestD = D

        # init kappa, pi
        self.mu = bestcenter
        kappa = self.Regularize * np.ones((self.n_components))
        pi = np.ones((self.n_components)) * (1/self.N)

        for k in range(self.n_components):
            idx = bestlabel == k
            if np.sum(idx) > 0:
                pi[k] = np.sum(idx) / self.N
                norMu = np.sqrt(mu[k, :].dot(mu[k, :].T))
                rbar = norMu / pi[k]
                mu[k, :] = mu[k, :] / norMu
                kappa[k] = max((rbar*self.D - rbar**3) / (1 - rbar**2), self.Regularize)
            else:
                print(1)

        self.kappa = kappa * self.D
        self.pi = pi / np.sum(pi)

    def predict(self):
        pass

    def fit_predict(self):
        pass

    def fit(self, x):

        self.init(x)
        iteration = 2
        loglike = np.ones((self.maxiter, 1)) * -np.inf
        converged = False

        while converged is False:

            # E-step
            (R, loglike[iteration]) = self.Expectation(x, self.pi, self.mu, self.kappa)
            # M-step
            self.pi, self.mu, self.kappa = self.Maximization(x, self.kappa, R)
            self.kappa = self.kappa + 1 / self.Regularize

            # check convergence
            deltlike = loglike[iteration] - loglike[iteration-1]
            deltlike = np.abs(100 * (deltlike / loglike[iteration-1]))
            if deltlike < self.tol or iteration >= self.maxiter-1:
                if iteration == self.maxiter - 1:
                    print('iteration is {}, please increase maxiter'.format(iteration))
                converged = True
                loglike = loglike[:iteration+1]
            iteration += 1

    def Expectation(self, x, pi, mu, kappa):

        D = self.D
        N = self.N
        log_normalize = np.log(pi) + (D / 2 - 1) * np.log(kappa) - (D / 2) * np.log(2 * pi) - self.log_besseli(D/2 - 1, kappa)
        R = x.dot(mu.T * (np.ones((D, 1)).dot(kappa[np.newaxis, :])))
        R = R + log_normalize
        T = logsumexp(R, 1)[:, np.newaxis]
        loglikelihood = np.sum(T) / N
        R = np.exp(R - T)
        return R, loglikelihood

    def Maximization(self, x, kappa, R):

        N = self.N
        D = self.D
        K = self.n_components

        pi = np.sum(R, 0) / N
        mu = R.T.dot(x)
        for k in range(K):
            norMu = np.sqrt(mu[k, :].dot(mu[k, :].T))
            rbar = norMu / pi[k]
            mu[k, :] = mu[k, :] / norMu
            kappa[k] = (rbar * D - rbar ** 3) / (1 - rbar ** 2)

        return pi, mu, kappa

    def log_besseli(self, nu, x):

        frac = x / nu
        square = 1 + frac**2
        root = np.sqrt(square)
        eta = root + np.log(frac) - np.log(1 + root)
        approx = - np.log(np.sqrt(2 * np.pi * nu)) + nu * eta - 0.25*np.log(square)

        return approx

    def pdf(self):
        pass


def randsample(n, k):

    h = list()
    while (len(h) < k):
        h.append(random.randint(0, n-1))

    return h


test = np.array(scio.loadmat('./demo_data.mat')['data'])
vm = VMFMixture()
vm.fit(test)
