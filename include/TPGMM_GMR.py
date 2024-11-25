from modelClass import model
from init_proposedPGMM_timeBased import init_proposedPGMM_timeBased
from EM_tensorGMM import EM_tensorGMM
from reproduction_DSGMR import reproduction_DSGMR
from plotGMM import plotGMM
import numpy as np
import copy

import rosbag
from tp_gmm.msg import GaussianMixture, Gaussian

class TPGMM_GMR(object):
    def __init__(self, nbStates, nbFrames, nbVar):
        self.model = model(nbStates, nbFrames, nbVar, None, None, None, None, None)

    def fit(self, s):
        self.s = s
        self.model = init_proposedPGMM_timeBased(s, self.model)
        self.model = EM_tensorGMM(s, self.model)

    def reproduce(self, p, currentPosition):
        return reproduction_DSGMR(self.s[0].Data[0,:], self.model, p, currentPosition)

    def plotReproduction(self, r, xaxis, yaxis, ax, showGaussians = True, lw = 7):
        for m in range(r.p.shape[0]):
            ax.plot([r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[xaxis, 0] + r.p[m, 0].A[xaxis, yaxis]],
                     [r.p[m, 0].b[yaxis, 0], r.p[m, 0].b[yaxis, 0] + r.p[m, 0].A[yaxis, yaxis]],
                     lw=lw, color=[0, 1, m])
            ax.plot(r.p[m, 0].b[xaxis, 0], r.p[m, 0].b[yaxis, 0], ms=30, marker='.', color=[0, 1, m])
        ax.plot(r.Data[xaxis, 0], r.Data[yaxis, 0], marker='.', ms=15)
        ax.plot(r.Data[xaxis, :], r.Data[yaxis, :])
        if showGaussians:
            plotGMM(r.Mu[np.ix_([xaxis, yaxis], range(r.Mu.shape[1]), [0])],
                    r.Sigma[np.ix_([xaxis, yaxis], [xaxis, yaxis], range(r.Mu.shape[1]), [0])], [0.5, 0.5, 0.5], 1, ax)

    def getReproductionMatrix(self, r):
        return r.Data

    # Converts the learnded GMM into a format that gmm_rviz_converter node can visualize it in Rviz
    def convertToGM(self, r, frame_id):

        ## converting to GaussianMixture() msg 
        nbGaussians = r.Mu.shape[1]
        gmm = GaussianMixture()
        g = Gaussian()

        # allMeans = np.hsplit(r.Mu[:,:,0], r.Mu.shape[1]) # (4,5) == (4,1)x5  -- each column represents one-unit Gaussian (if nbGaussian = 5)
        # allCovariances = np.ravel(r.Sigma[:,:,:,0]) # (4,4,5) -> (80, 1) --every 16 elements represent one-unit Gaussian 

        for gaus in range(nbGaussians):
            g.means = r.Mu[:,gaus,-1]
            g.covariances = np.ravel(r.Sigma[:,:,gaus,-1])
            gmm.gaussians.append(copy.deepcopy(g))
        gmm.weights = self.model.Priors # or r.H
        gmm.bic = r.Data.shape[1]
        gmm.header.frame_id = frame_id

        ## Writing to rosbag
        wbag = rosbag.Bag("/home/erl/Multicobot-UR10/src/tp_gmm/data/tpgmm_mix.bag", 'w')
        wbag.write("/gmm/mix", gmm)
        wbag.close()

        return gmm