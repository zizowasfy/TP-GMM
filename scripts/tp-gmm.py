#!/usr/bin/env python3
# ROS stuff
import rospy
from geometry_msgs.msg import Pose, PointStamped
from gaussian_mixture_model.msg import GaussianMixture
from tp_gmm.srv import *

# System and directories stuff
import sys
sys.path.append("/home/zizo/haptics-ctrl_ws/src/tp_gmm/include")
data_dir = "/home/zizo/haptics-ctrl_ws/src/tp_gmm/data/"
scripts_dir = "/home/zizo/haptics-ctrl_ws/src/tp_gmm/scripts/"
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pickle

# tpgmm-related stuff
import numpy as np
from math import sqrt
from scipy.spatial.transform import Rotation as R
from sClass import s
from pClass import p
from matplotlib import pyplot as plt
from TPGMM_GMR import TPGMM_GMR
from copy import deepcopy

class TPGMM:
    def __init__(self):

        rospy.init_node("tp_gmm_node")
        print(" --> Node tp_gmm_node is initialized")
        rospy.Service("StartTPGMM_service", StartTPGMM, self.startTPGMM)
        self.tpgmm_pub = rospy.Publisher('/gmm/mix', GaussianMixture, queue_size=1)

        self.demonsToSamples_flag = False   
        self.demonsToSamples()

    def startTPGMM(self, req):
        
        ## Receiving service Request
        self.task_name = req.task_name
        self.frame1_pose = req.start_pose
        self.frame2_pose = req.goal_pose

        ## Fetching Samples and paramters
        self.demonsToSamples_flag = True
        print("startTPGMM")

        rospy.sleep(0.5)
        while not rospy.is_shutdown() and self.demonsToSamples_flag: pass   # wait until demonsToSamples finishes

        with open(scripts_dir + 'demons_info2.pkl', 'rb') as fp:
            self.demons_info2 = pickle.load(fp)
            print("Reference Demon: ", self.demons_info2)

        ## Initialization of parameters and properties------------------------------------------------------------------------- #
        self.nbSamples = self.demons_info2['nbDemons']  # nb of demonstrations
        self.nbVar = 4      # Dim !!
        self.nbFrames = 2 
        self.nbStates = 5   # nb of Gaussians
        self.nbData = self.demons_info2['ref_nbpoints']-1

        self.tpGMM()
        return StartTPGMMResponse(True)
    
    ## Running the demons_to_samples.ipynb-------------------------------------------------------------------------------- #
    def demonsToSamples(self):
        # Waiting for a startTPGMM request - the while was the only way to do that to avoid 'no current event loop in thread' error
        while not rospy.is_shutdown() and not self.demonsToSamples_flag:
            # print("demonsToSamples")
            pass

        # Sending task_name to demons_to_samples.ipynb in .pkl file
        demons_info1 = {"task_name": self.task_name}
        with open(scripts_dir + 'demons_info1.pkl', 'wb') as fp:
            pickle.dump(demons_info1, fp)
            print("demons_info1: ", demons_info1)

        # Running demons_to_samples.ipynb
        with open(scripts_dir + "demons_to_samples.ipynb") as f:
            nb_in = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        nb_out = ep.preprocess(nb_in)
        print("demons_to_samples finished running!")
        self.demonsToSamples_flag = False

    ## Preparing the samples and fit ----------------------------------------------------------------------------------------- #
    def tpGMM(self):
        self.slist = []
        for i in range(self.nbSamples):
            pmat = np.empty(shape=(self.nbFrames, self.nbData), dtype=object)
            # tempData = np.loadtxt('sample' + str(i + 1) + '_Data.txt', delimiter=',')
            tempData = np.loadtxt(data_dir + 'Demon' + str(i+1) + '_sample' + '_Data.txt', delimiter=',')
            print(tempData.shape)
            for j in range(self.nbFrames):
                # tempA = np.loadtxt('sample' + str(i + 1) + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
                # tempB = np.loadtxt('sample' + str(i + 1) + '_frame' + str(j + 1) + '_b.txt', delimiter=',')
                tempA = np.loadtxt(data_dir +'Demon' + str(i+1) + '_sample' + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
                tempB = np.loadtxt(data_dir +'Demon' + str(i+1) + '_sample' + '_frame' + str(j + 1) + '_b.txt', delimiter=',')

                for k in range(self.nbData):
                    pmat[j, k] = p(tempA[:, self.nbVar*k : self.nbVar*k + self.nbVar], tempB[:, k].reshape(len(tempB[:, k]), 1),
                                np.linalg.pinv(tempA[:, self.nbVar*k : self.nbVar*k + self.nbVar]), self.nbStates)                         
            self.slist.append(s(pmat, tempData, tempData.shape[1], self.nbStates))

        # Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
        self.TPGMMGMR = TPGMM_GMR(self.nbStates, self.nbFrames, self.nbVar)

        # Learning the model-------------------------------------------------------------------------------------------------- #
        self.TPGMMGMR.fit(self.slist)

        self.tpGMMGMR()

    def tpGMMGMR(self):
        # # Reproduction for parameters used in demonstration------------------------------------------------------------------- #
        # self.rdemolist = []
        # for n in range(self.nbSamples):
        #     self.rdemolist.append(self.TPGMMGMR.reproduce(self.slist[n].p, self.slist[n].Data[1:self.nbVar,0]))
        #     print(self.rdemolist[n].Mu.shape)
        #     print(self.rdemolist[n].Sigma.shape)

        # # Reproduction with generated parameters------------------------------------------------------------------------------ #
        # self.rnewlist = []
        # for n in range(self.nbSamples):
        #     newP = deepcopy(self.slist[n].p)
        #     for m in range(1, self.nbFrames):
        #         bTransform = np.random.rand(self.nbVar, 1) + 0.5
        #         aTransform = np.random.rand(self.nbVar, self.nbVar) +0.5
        #         for k in range(self.nbData):
        #             newP[m, k].A = newP[m, k].A * aTransform
        #             newP[m, k].b = newP[m, k].b * bTransform
        #     self.rnewlist.append(self.TPGMMGMR.reproduce(newP, self.slist[n].Data[1:self.nbVar, 0]))

        # Reproduction with generated parameters------------------------------------------------------------------------------ #
        self.getFramePoses()
        newP = deepcopy(self.slist[int(self.demons_info2['ref'][-1])-1].p)
        newb1 = np.array([[0], [self.frame1_pose.position.x[0]], [self.frame1_pose.position.y[0]], [self.frame1_pose.position.z[0]]], dtype=object)
        newb2 = np.array([[0], [self.frame2_pose.position.x], [self.frame2_pose.position.y], [self.frame2_pose.position.z]], dtype=object)

        print([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
        rA1 = R.from_quat([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
        rA2 = R.from_quat([self.frame2_pose.orientation.x, self.frame2_pose.orientation.y, self.frame2_pose.orientation.z, self.frame2_pose.orientation.w])
        newA1 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA1.as_matrix() )) )) # TODO: Quat2rotMat
        newA2 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA2.as_matrix() )) )) # TODO: Quat2rotMat
        print(newA1)
        print(newb1)
        for k in range(self.nbData):
            newP[0, k].b = newb1
            newP[1, k].b = newb2            
            newP[0, k].A = newA1
            newP[1, k].A = newA2
            newP[0, k].invA = np.linalg.pinv(newA1)
            newP[1, k].invA = np.linalg.pinv(newA2)

        rnew = self.TPGMMGMR.reproduce(newP, newb1[1:,:])

        # Saving GMM to rosbag ------------------------------------------------------------------------------------------------------------ #
        gmm = self.TPGMMGMR.convertToGM(rnew)

        self.tpgmm_pub.publish(gmm)
        print("GMM is Published!")
        # self.tpGMMPlot()
        rospy.signal_shutdown("TP-GMM Node is Shutting Down!")

    ## Check if frame1_pose and frame2_pose hasn't been requested from startTPGMM rosservice, fill them with these values
    def getFramePoses(self):
        if (sqrt(self.frame1_pose.position.x**2 + self.frame1_pose.position.y**2 + self.frame1_pose.position.z**2) == 0):
            print("... Didn't receive a requested start_pose")
            self.frame1_pose.position.x, self.frame1_pose.position.y, self.frame1_pose.position.z = self.slist[int(self.demons_info2['ref'][-1])-1].p[0,0].b[1:,:]
            # TODO: Fill the .orientation after adding rotMat2Quat() function
            r = R.from_matrix(self.slist[int(self.demons_info2['ref'][-1])-1].p[0,0].A[1:,1:])
            self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w = r.as_quat()

        if (sqrt(self.frame2_pose.position.x**2 + self.frame2_pose.position.y**2 + self.frame2_pose.position.z**2) == 0):
            print(" ... Waiting on goal_pose from /clicked_point topic ... ")
            clkd_point = rospy.wait_for_message("/clicked_point", PointStamped)
            self.frame2_pose.position = clkd_point.point
            # TODO: Add .orientation after adding rotMat2Quat() function
            r = R.from_matrix(self.slist[int(self.demons_info2['ref'][-1])-1].p[1,0].A[1:,1:])
            self.frame2_pose.orientation.x, self.frame2_pose.orientation.y, self.frame2_pose.orientation.z, self.frame2_pose.orientation.w = r.as_quat()


    # def tpGMMPlot(self):
    #     # Plotting------------------------------------------------------------------------------------------------------------ #
    #     xaxis = 1
    #     yaxis = 3
    #     xlim = [-0.3, 0.8]
    #     ylim = [-0.2, 0.8]

    #     # Demos--------------------------------------------------------------------------------------------------------------- #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(131)
    #     ax1.set_xlim(xlim)
    #     ax1.set_ylim(ylim)
    #     ax1.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    #     plt.title('Demonstrations')
    #     for n in range(self.nbSamples):
    #         for m in range(self.nbFrames):
    #             ax1.plot([self.slist[n].p[m,0].b[xaxis,0], self.slist[n].p[m,0].b[xaxis,0] + self.slist[n].p[m,0].A[xaxis,yaxis]], [self.slist[n].p[m,0].b[yaxis,0], self.slist[n].p[m,0].b[yaxis,0] + self.slist[n].p[m,0].A[yaxis,yaxis]], lw = 3, color = [0,1,m])
    #             ax1.plot(self.slist[n].p[m,0].b[xaxis,0], self.slist[n].p[m,0].b[yaxis,0], ms = 10, marker = '.', color = [0,1,m])
    #         ax1.plot(self.slist[n].Data[xaxis,0], self.slist[n].Data[yaxis,0], marker = '.', ms = 15)
    #         ax1.plot(self.slist[n].Data[xaxis,:], self.slist[n].Data[yaxis,:])

    #     # Reproductions with training parameters------------------------------------------------------------------------------ #
    #     ax2 = fig.add_subplot(132)
    #     ax2.set_xlim(xlim)
    #     ax2.set_ylim(ylim)
    #     ax2.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    #     plt.title('Reproductions with same task parameters')
    #     for n in range(self.nbSamples):
    #         self.TPGMMGMR.plotReproduction(self.rdemolist[n], xaxis, yaxis, ax2, showGaussians=True)

    #     # Reproductions with new parameters----------------------------------------------------------------------------------- #
    #     ax3 = fig.add_subplot(133)
    #     ax3.set_xlim(xlim)
    #     ax3.set_ylim(ylim)
    #     ax3.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0]))
    #     plt.title('Reproduction with generated task parameters')
    #     for n in range(self.nbSamples):
    #         self.TPGMMGMR.plotReproduction(self.rnewlist[n], xaxis, yaxis, ax3, showGaussians=True)

    #     print("ProductionMatrix:")
    #     print(self.TPGMMGMR.getReproductionMatrix(self.rnewlist[0]))

    #     plt.show()

if __name__ == "__main__":

    tpgmm = TPGMM()
    rospy.spin()