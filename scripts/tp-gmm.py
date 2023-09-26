#!/usr/bin/env python3
# ROS stuff
import rospy
from geometry_msgs.msg import Pose, PointStamped, PoseArray
from sensor_msgs.msg import JointState
from gaussian_mixture_model.msg import GaussianMixture
from tp_gmm.srv import *
from moveit_msgs.msg import MoveGroupActionResult
from trajectory_msgs.msg import JointTrajectoryPoint

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

        self.move_group_q_viz_pub = rospy.Publisher('/move_group/result', MoveGroupActionResult, queue_size=1)
        self.tpgmm_pub = rospy.Publisher('/gmm/mix', GaussianMixture, queue_size=1)
        self.learned_traj_pub = rospy.Publisher('/gmm/learned_trajectory', PoseArray, queue_size=1)

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
        self.nbVar = 4      # Dim !!
        self.nbFrames = 2 
        self.nbStates = 2  # nb of Gaussians
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

        ## COMMENTING this only to make the code run faster while debugging
        # Running demons_to_samples.ipynb
        # with open(scripts_dir + "demons_to_samples.ipynb") as f:
        #     nb_in = nbformat.read(f, as_version=4)
        # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        # nb_out = ep.preprocess(nb_in)
        # print("demons_to_samples finished running!")
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
                                np.linalg.inv(tempA[:, self.nbVar*k : self.nbVar*k + self.nbVar]), self.nbStates)                         
            self.slist.append(s(pmat, tempData, tempData.shape[1], self.nbStates))

        # Creating instance of TPGMM_GMR-------------------------------------------------------------------------------------- #
        self.TPGMMGMR = TPGMM_GMR(self.nbStates, self.nbFrames, self.nbVar)

        # Learning the model-------------------------------------------------------------------------------------------------- #
        self.TPGMMGMR.fit(self.slist)
        
        # self.tpGMMGMR()

    def tpGMMGMR(self, req):
        # Reproduction with generated parameters------------------------------------------------------------------------------ #
        self.frame1_pose = req.start_pose.pose
        self.frame2_pose = req.goal_pose.pose         
        # self.getFramePoses()
        newP = deepcopy(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p)
        print("self.demons_info2['demons_nums'].index(self.demons_info2['ref']) = ", self.demons_info2['demons_nums'].index(self.demons_info2['ref']))
        # newP = p(np.zeros((self.nbVar,self.nbVar)), np.zeros((self.nbVar,1)), np.zeros((self.nbVar,self.nbVar)), self.nbStates)
        newb1 = np.array([[0], [self.frame1_pose.position.x], [self.frame1_pose.position.y], [self.frame1_pose.position.z]], dtype=object)
        newb2 = np.array([[0], [self.frame2_pose.position.x], [self.frame2_pose.position.y], [self.frame2_pose.position.z]], dtype=object)

        # print([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
        rA1 = R.from_quat([self.frame1_pose.orientation.x, self.frame1_pose.orientation.y, self.frame1_pose.orientation.z, self.frame1_pose.orientation.w])
        rA2 = R.from_quat([self.frame2_pose.orientation.x, self.frame2_pose.orientation.y, self.frame2_pose.orientation.z, self.frame2_pose.orientation.w])
        newA1 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA1.as_matrix() )) )) # TODO: Quat2rotMat
        newA2 = np.vstack(( np.array([1,0,0,0]), np.hstack(( np.zeros((3,1)), rA2.as_matrix() )) )) # TODO: Quat2rotMat
        # print(newA1)
        # print(newb1)
        for k in range(self.nbData):
            newP[0, k].b = newb1
            newP[1, k].b = newb2            
            newP[0, k].A = newA1
            newP[1, k].A = newA2
            newP[0, k].invA = np.linalg.inv(newA1) # TOTRY: with and without invA
            newP[1, k].invA = np.linalg.inv(newA2) # TOTRY: with and without invA

        rnew = self.TPGMMGMR.reproduce(newP, newb1[1:,:])
        
        # Debugging and Visualizing in RViz the trajectory reproduced from regression TP-GMR
        print("rnew ReproductionMatrix: ", self.TPGMMGMR.getReproductionMatrix(rnew).shape)
        print("rnew.H: ", rnew.H[:,-5:-1])
        print("rnew.Data: ", rnew.Data[:,:5])
        posearray = PoseArray()
        pose = Pose()
        for k in range(self.nbData):
            pose.position.x = rnew.Data[1,k]
            pose.position.y = rnew.Data[2,k]
            pose.position.z = rnew.Data[3,k]
            posearray.poses.append(deepcopy(pose))
        posearray.header.frame_id = "panda_link0"
        self.learned_traj_pub.publish(posearray)
        #/ Debugging
        # Saving GMM to rosbag ------------------------------------------------------------------------------------------------------------ #
        gmm = self.TPGMMGMR.convertToGM(rnew)

        # Cartesian Space to Joint Space Transformation ----------------------------------------------------------------------------------- #
        move_group_q_viz = MoveGroupActionResult()
        jointtrajectorypoint_q_viz = JointTrajectoryPoint()
        # solving IK using jacobian-based
        q_jointstate = JointState()
        get_jacobian_client = rospy.ServiceProxy("/get_jacobian_service", GetJacobian)
        resp = get_jacobian_client(True, None)
        # q_0 = np.expand_dims(np.array(resp.q_0.position), axis=1) # start joint configuration
        q_0 = np.array(resp.q_0.position) # start joint configuration
        print("q_0: ", q_0)
        # q_0 = np.vstack(np.zeros(1), q_0)
        q_i = deepcopy(q_0) # q_(t-1)
        q_t = np.zeros((q_0.shape[0],rnew.Data.shape[1]))
        print("q_t.shape: ", q_t.shape)
        print("q_t[:,0].shape: ", q_t[:,0].shape)
        q_t[:,0] = q_0 # Initialize q_t with q_0
        # qData = np.zeros((rnew.Data.shape))

        # Mu and Covariance of the Joint Space
        q_Mu = np.ndarray(shape=(7, rnew.Mu.shape[1], rnew.Mu.shape[2])) # rnew.Mu.shape)
        q_Sigma = np.ndarray(shape=(7, 7, rnew.Sigma.shape[2], rnew.Sigma.shape[3])) # rnew.Sigma.shape)

        inc = 8
        for t in range(inc, rnew.Data.shape[1], inc):
            # getJacobian
            q_jointstate.position = q_t[:,t-inc]
            get_jacobian_client = rospy.ServiceProxy("/get_jacobian_service", GetJacobian)
            resp = get_jacobian_client(False, q_jointstate)
            # print("jacobian_vec = ", resp.jacobian_vec)
            # jacobian_mat = np.reshape(np.array(resp.jacobian_vec), (6,7), order='C') # row-major order
            # print("jacobian_mat_row: ", jacobian_mat)
            jacobian_mat = np.reshape(np.array(resp.jacobian_vec), (7,7), order='F') # col-major order
            jacobian_mat = jacobian_mat[:3,:]
            print("jacobian_mat: ", jacobian_mat)
            jacobian_pinv = np.linalg.pinv(jacobian_mat)
            print("jacobian_pinv.shape: ",jacobian_pinv.shape)
            # compute IK
            x_i = rnew.Data[1:,t-inc]# x_(t-1)
            x_t = rnew.Data[1:,t]
            # q_t[0,t] = t
            print("q_t[:,t].shape: ", q_t[:,t].shape)
            print("q_t[:,t-inc].shape: ", q_t[:,t-inc].shape)
            print("q_t_0: ", q_t[:,t-inc])
            q_t[:,t] = q_t[:,t-inc] + jacobian_pinv @ (np.array(x_t) - np.array(x_i)) #(np.append(x_t, np.zeros(4)) - np.append(x_i, np.zeros(4)))
            print("q_t[:,t]: ", q_t[:,t])
            # For Visualization
            jointtrajectorypoint_q_viz.positions = q_t[:,t]
            move_group_q_viz.result.planned_trajectory.joint_trajectory.points.append(deepcopy(jointtrajectorypoint_q_viz))
            #/ For Visualization
            # Calculating the mean and covariance of the Joint Space after the transformation
            rnew_Mu_i = rnew.Mu[1:,:,t-inc]
            # rnew_Mu_i = np.append(rnew_Mu_i, np.zeros(4))[:,np.newaxis] 
            rnew_Mu_t = rnew.Mu[1:,:,t]
            # rnew_Mu_t = np.append(rnew_Mu_t, np.zeros(4))[:,np.newaxis]
            rnew_Sigma_i = rnew.Sigma[1:,:,:,t-inc]
            rnew_Sigma_t_temp = rnew.Sigma[1:,1:,:,t]; rnew_Sigma_t = np.squeeze(rnew_Sigma_t_temp)
            # rnew_Sigma_t = np.zeros((7,7))
            # rnew_Sigma_t[:rnew_Sigma_t_temp.shape[0], :rnew_Sigma_t_temp.shape[1]] = np.squeeze(rnew_Sigma_t_temp)
            
        #     q_Mu[:,:,t] = q_t[:,t-inc][:,np.newaxis] + jacobian_pinv @ (rnew_Mu_t - rnew_Mu_i)
            
        #     print("rnew_Sigma_t.shape: ", rnew_Sigma_t.shape)            
        #     print("jacobian_pinv.shape: ", jacobian_pinv.shape)
        #     q_Sigma[:,:,:,t] = (jacobian_pinv @ rnew_Sigma_t @ jacobian_pinv.T)[:,:,np.newaxis]

        #     print("t: ", t)


        # q_rnew = deepcopy(rnew)
        # q_rnew.Data = np.vstack((np.zeros((1,q_t.shape[1])), q_t))
        # q_rnew.Mu[1:,:,:] = q_Mu[:3,:,:]           # To keep the values of 1st Var in nbVar (which is time) the same as rnew
        # q_rnew.Sigma[1:,1:,:,:] = q_Sigma[:3,:3,:,:]
        # gmm = self.TPGMMGMR.convertToGM(q_rnew)

        print("move_group size: ", len(move_group_q_viz.result.planned_trajectory.joint_trajectory.points))
        self.move_group_q_viz_pub.publish(move_group_q_viz)

        # Save and Publish reproduced TP-GMM  --------------------------------------------------------------------------------------------- #
        self.tpgmm_pub.publish(gmm)
        print("GMM is Published!")
        # self.tpGMMPlot()
        # rospy.signal_shutdown("TP-GMM Node is Shutting Down!")
        return ReproduceTPGMMResponse()

    # # Joint Space tpgmm model
    # def tpGMMGMR(self, req):
    #     # Reproduction with generated parameters------------------------------------------------------------------------------ #
    #     # self.frame1_joints = req.start_joints.position
    #     # self.frame2_joints = req.goal_joints.position
    #     # self.frame1_joints = np.expand_dims(np.array(self.frame1_joints), axis=1)
    #     # self.frame2_joints = np.expand_dims(np.array(self.frame2_joints), axis=1)
    #     # self.getFramePoses()
        
    #     newP = deepcopy(self.slist[self.demons_info2['demons_nums'].index(self.demons_info2['ref'])].p)
    #     print("self.demons_info2['demons_nums'].index(self.demons_info2['ref']) = ", self.demons_info2['demons_nums'].index(self.demons_info2['ref']))
    #     # newP = p(np.zeros((self.nbVar,self.nbVar)), np.zeros((self.nbVar,1)), np.zeros((self.nbVar,self.nbVar)), self.nbStates)
    #     # newb1 = np.vstack( ([0], self.frame1_joints) )
    #     # newb2 = np.vstack( ([0], self.frame2_joints) )
    #     get_jacobian_client = rospy.ServiceProxy("/get_jacobian_service", GetJacobian)
    #     resp = get_jacobian_client() # Pass the recorded joints at each time-step#
    #     # Converting all service data into one column numpy array and/or Matrices
    #     joint_positions_frame1 = np.expand_dims(np.array(resp.joint_positions_frame1.position), axis=1)
    #     joint_positions_frame2 = np.expand_dims(np.array(resp.joint_positions_frame2.position), axis=1)
    #     pose_frame1 = np.expand_dims(np.array(resp.pose_frame1), axis=1)
    #     pose_frame2 = np.expand_dims(np.array(resp.pose_frame2), axis=1)
    #     jacobian_mat_frame1 = np.reshape(np.array(resp.jacobian_vec_frame1), (7,7))
    #     jacobian_mat_frame2 = np.reshape(np.array(resp.jacobian_vec_frame2), (7,7))
    #     jacobian_pinv_frame1 = np.linalg.pinv(jacobian_mat_frame1)
    #     jacobian_pinv_frame2 = np.linalg.pinv(jacobian_mat_frame2)

    #     print(joint_positions_frame1.shape)
    #     print(joint_positions_frame2.shape)
    #     print(pose_frame1.shape)
    #     print(pose_frame2.shape)
    #     print(jacobian_pinv_frame1.shape)
    #     print(jacobian_pinv_frame2.shape)

    #     print(jacobian_pinv_frame1@pose_frame1)
    #     newb1 = np.vstack( ([0], joint_positions_frame1 - jacobian_pinv_frame1@pose_frame1) )
    #     newb2 = np.vstack( ([0], joint_positions_frame2 - jacobian_pinv_frame2@pose_frame2) )
    #     # newA1 = jacobian_pinv_frame1
    #     # newA2 = jacobian_pinv_frame2
    #     newA1 = np.vstack(( np.array([1,0,0,0,0,0,0,0]), np.hstack(( np.zeros((7,1)), jacobian_pinv_frame1 )) ))
    #     newA2 = np.vstack(( np.array([1,0,0,0,0,0,0,0]), np.hstack(( np.zeros((7,1)), jacobian_pinv_frame2 )) ))

    #     for k in range(self.nbData):
    #         newP[0, k].b = newb1
    #         newP[1, k].b = newb2
    #         newP[0, k].A = newA1
    #         newP[1, k].A = newA2
    #         newP[0, k].invA = np.linalg.pinv(newA1)
    #         newP[1, k].invA = np.linalg.pinv(newA2)
            

    #     rnew = self.TPGMMGMR.reproduce(newP, newb1[1:,:])

    #     # Saving GMM to rosbag ------------------------------------------------------------------------------------------------------------ #
    #     gmm = self.TPGMMGMR.convertToGM(rnew)

    #     self.tpgmm_pub.publish(gmm)
    #     print("GMM is Published!")
    #     # self.tpGMMPlot()
    #     # rospy.signal_shutdown("TP-GMM Node is Shutting Down!")
    #     return ReproduceTPGMMResponse()

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