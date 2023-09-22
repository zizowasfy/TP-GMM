#!/usr/bin/env python3
# ROS stuff
import rospy
from geometry_msgs.msg import Pose, PointStamped
from sensor_msgs.msg import JointState
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
from copy import deepcopy,copy
from data_handle.srv import GetJacobian

class TPGMM:
    def __init__(self):

        rospy.init_node("tp_gmm_node")
        print(" --> Node tp_gmm_node is initialized")
        rospy.Service("StartTPGMM_service", StartTPGMM, self.startTPGMM)
        rospy.Service("ReproduceTPGMM_service", ReproduceTPGMM, self.tpGMMGMR)

        self.tpgmm_pub = rospy.Publisher('/gmm/mix', GaussianMixture, queue_size=1)

        self.demonsToSamples_flag = False   
        self.demonsToSamples()


    def startTPGMM(self, req):

        ## Receiving service Request
        # if rospy.has_param("/task_param"):
        #     self.task_name = rospy.get_param("/task_param")
        #     print("/task_param: ", self.task_name)
        # else:
        # self.task_name = req.task_name
        # self.frame1_pose = req.start_pose
        # self.frame2_pose = req.goal_pose
        self.frame1_pose = Pose()
        self.frame2_pose = Pose()        
        self.task_name = rospy.get_param("/task_param")
        ## Fetching Samples and paramters
        self.demonsToSamples_flag = True
        print("startTPGMM")

        rospy.sleep(0.5)
        while not rospy.is_shutdown() and self.demonsToSamples_flag: pass   # wait until demonsToSamples finishes

        with open(scripts_dir + 'demons_info2.pkl', 'rb') as fp:
            self.demons_info2 = pickle.load(fp)
            print("demons_info2: ", self.demons_info2)

        ## Initialization of parameters and properties------------------------------------------------------------------------- #
        self.nbSamples = self.demons_info2['nbDemons']  # nb of demonstrations
        self.nbVar = 8      # Dim !!
        self.nbFrames = 2 
        self.nbStates = 3  # nb of Gaussians
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
        demons_nums = self.demons_info2['demons_nums']
        self.slist = []
        for i in range(self.nbSamples):
            pmat = np.empty(shape=(self.nbFrames, self.nbData), dtype=object)
            tempData = np.loadtxt(data_dir + demons_nums[i] + '_sample' + '_Data.txt', delimiter=',')
            print(tempData.shape)
            for j in range(self.nbFrames):
                tempA = np.loadtxt(data_dir + demons_nums[i] + '_sample' + '_frame' + str(j + 1) + '_A.txt', delimiter=',')
                tempB = np.loadtxt(data_dir + demons_nums[i] + '_sample' + '_frame' + str(j + 1) + '_b.txt', delimiter=',')

                for k in range(self.nbData):
                    pmat[j, k] = p(tempA[:, self.nbVar*k : self.nbVar*k + self.nbVar], tempB[:, k].reshape(len(tempB[:, k]), 1),
                                np.linalg.pinv(tempA[:, self.nbVar*k : self.nbVar*k + self.nbVar]), self.nbStates)                         
            self.slist.append(s(pmat, tempData, tempData.shape[1], self.nbStates))

        # Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
        self.TPGMMGMR = TPGMM_GMR(self.nbStates, self.nbFrames, self.nbVar)

        # Learning the model-------------------------------------------------------------------------------------------------- #
        self.TPGMMGMR.fit(self.slist)
        
        # self.tpGMMGMR()

    # def tpGMMGMR(self, req):
    #     # Reproduction with generated parameters------------------------------------------------------------------------------ #
    #     self.frame1_pose = req.start_pose.pose
    #     self.frame2_pose = req.goal_pose.pose         
    #     self.getFramePoses()
    #     newP = deepcopy(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p)
    #     print("self.demons_info2['demons_nums'].index(self.demons_info2['ref']) = ", self.demons_info2['demons_nums'].index(self.demons_info2['ref']))
    #     # newP = p(np.zeros((self.nbVar,self.nbVar)), np.zeros((self.nbVar,1)), np.zeros((self.nbVar,self.nbVar)), self.nbStates)
    #     newb1 = np.array([[0], [self.frame1_pose.position.x], [self.frame1_pose.position.y], [self.frame1_pose.position.z]], dtype=object)
    #     newb2 = np.array([[0], [self.frame2_pose.position.x], [self.frame2_pose.position.y], [self.frame2_pose.position.z]], dtype=object)

    #     # print([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
    #     rA1 = R.from_quat([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
    #     rA2 = R.from_quat([self.frame2_pose.orientation.x, self.frame2_pose.orientation.y, self.frame2_pose.orientation.z, self.frame2_pose.orientation.w])
    #     newA1 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA1.as_matrix() )) )) # TODO: Quat2rotMat
    #     newA2 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA2.as_matrix() )) )) # TODO: Quat2rotMat
    #     # print(newA1)
    #     # print(newb1)
    #     for k in range(self.nbData):
    #         newP[0, k].b = newb1
    #         newP[1, k].b = newb2            
    #         newP[0, k].A = newA1
    #         newP[1, k].A = newA2
    #         newP[0, k].invA = np.linalg.pinv(newA1) # TOTRY: with and without invA
    #         newP[1, k].invA = np.linalg.pinv(newA2) # TOTRY: with and without invA

    #     rnew = self.TPGMMGMR.reproduce(newP, newb1[1:,:])

    #     # Saving GMM to rosbag ------------------------------------------------------------------------------------------------------------ #
    #     gmm = self.TPGMMGMR.convertToGM(rnew)

    #     self.tpgmm_pub.publish(gmm)
    #     print("GMM is Published!")
    #     # self.tpGMMPlot()
    #     # rospy.signal_shutdown("TP-GMM Node is Shutting Down!")
    #     return ReproduceTPGMMResponse()

    # Joint Space tpgmm model
    def tpGMMGMR(self, req):
        # Reproduction with generated parameters------------------------------------------------------------------------------ #
        # self.frame1_joints = req.start_joints.position
        # self.frame2_joints = req.goal_joints.position
        # self.frame1_joints = np.expand_dims(np.array(self.frame1_joints), axis=1)
        # self.frame2_joints = np.expand_dims(np.array(self.frame2_joints), axis=1)
        # self.getFramePoses()
        
        newP = deepcopy(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p)
        print("self.demons_info2['demons_nums'].index(self.demons_info2['ref']) = ", self.demons_info2['demons_nums'].index(self.demons_info2['ref']))
        # newP = p(np.zeros((self.nbVar,self.nbVar)), np.zeros((self.nbVar,1)), np.zeros((self.nbVar,self.nbVar)), self.nbStates)
        # newb1 = np.vstack( ([0], self.frame1_joints) )
        # newb2 = np.vstack( ([0], self.frame2_joints) )
        get_jacobian_client = rospy.ServiceProxy("/get_jacobian_service", GetJacobian)
        resp = get_jacobian_client() # Pass the recorded joints at each time-step#
        # Converting all service data into one column numpy array and/or Matrices
        joint_positions_frame1 = np.expand_dims(np.array(resp.joint_positions_frame1.position), axis=1)
        joint_positions_frame2 = np.expand_dims(np.array(resp.joint_positions_frame2.position), axis=1)
        pose_frame1 = np.expand_dims(np.array(resp.pose_frame1), axis=1)
        pose_frame2 = np.expand_dims(np.array(resp.pose_frame2), axis=1)
        jacobian_mat_frame1 = np.reshape(np.array(resp.jacobian_vec_frame1), (7,7))
        jacobian_mat_frame2 = np.reshape(np.array(resp.jacobian_vec_frame2), (7,7))
        jacobian_pinv_frame1 = np.linalg.pinv(jacobian_mat_frame1)
        jacobian_pinv_frame2 = np.linalg.pinv(jacobian_mat_frame2)

        print(joint_positions_frame1.shape)
        print(joint_positions_frame2.shape)
        print(pose_frame1.shape)
        print(pose_frame2.shape)
        print(jacobian_pinv_frame1.shape)
        print(jacobian_pinv_frame2.shape)

        print(jacobian_pinv_frame1@pose_frame1)
        newb1 = np.vstack( ([0], joint_positions_frame1 - jacobian_pinv_frame1@pose_frame1) )
        newb2 = np.vstack( ([0], joint_positions_frame2 - jacobian_pinv_frame2@pose_frame2) )
        # newA1 = jacobian_pinv_frame1
        # newA2 = jacobian_pinv_frame2
        newA1 = np.vstack(( np.array([1,0,0,0,0,0,0,0]), np.hstack(( np.zeros((7,1)), jacobian_pinv_frame1 )) ))
        newA2 = np.vstack(( np.array([1,0,0,0,0,0,0,0]), np.hstack(( np.zeros((7,1)), jacobian_pinv_frame2 )) ))

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
        # rospy.signal_shutdown("TP-GMM Node is Shutting Down!")
        return ReproduceTPGMMResponse()

    ## Check if frame1_pose and frame2_pose hasn't been requested from startTPGMM rosservice, fill them with these values
    def getFramePoses(self):
        if (sqrt(self.frame1_pose.position.x**2 + self.frame1_pose.position.y**2 + self.frame1_pose.position.z**2) == 0):
            print("... Didn't receive a requested start_pose")
            self.frame1_pose.position.x, self.frame1_pose.position.y, self.frame1_pose.position.z = self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p[0,0].b[1:,:]
            self.frame1_pose.position.x, self.frame1_pose.position.y, self.frame1_pose.position.z = self.frame1_pose.position.x[0], self.frame1_pose.position.y[0], self.frame1_pose.position.z[0]
            # TODO: Fill the .orientation after adding rotMat2Quat() function
            r = R.from_matrix(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p[0,0].A[1:,1:])
            self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w = r.as_quat()

        if (sqrt(self.frame2_pose.position.x**2 + self.frame2_pose.position.y**2 + self.frame2_pose.position.z**2) == 0):
            print(" ... Waiting on goal_pose from /clicked_point topic ... ")
            clkd_point = rospy.wait_for_message("/clicked_point", PointStamped)
            self.frame2_pose.position = clkd_point.point
            # TODO: Add .orientation after adding rotMat2Quat() function
            r = R.from_matrix(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p[1,0].A[1:,1:])
            self.frame2_pose.orientation.x, self.frame2_pose.orientation.y, self.frame2_pose.orientation.z, self.frame2_pose.orientation.w = r.as_quat()

if __name__ == "__main__":

    tpgmm = TPGMM()
    rospy.spin()