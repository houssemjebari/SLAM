"""
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology

Description : This file contains EKF SLAM  algorithm with 
known-correspondance tested for the UTIAS Multi-Robot Cooperative 
Localization and Mapping Dataset. 
For details about the algorithm please see Probabilistic Robotics
book: Page 314, Table 10.1
****************************************************************
"""
import numpy as np 
from .gaussian_filter import GaussianFilter

class EkfSlamKnownCorrespondance(GaussianFilter):
    '''
    class that implements the Extended Kalman Filter SLAM
    
    attributes:
        * R: motion covariance matrix 
        * Q: measurement covariance matrix 
        * mu: robot state and map state vector
        * sigma: robot and map state covariance
    '''
        
    def __init__(self):
        GaussianFilter.__init__(self)
        self.landmark_indexes = []
        self.landmark_observed = np.array([])

    def __init__(self,R,Q,mu,sigma,landmark_indexes):
        GaussianFilter.__init__(self,R,Q,mu,sigma)
        self.landmark_indexes = landmark_indexes
        self.mu = np.concatenate((mu,np.zeros((2*len(landmark_indexes),))))
        sigma_values = np.ones(3+2*len(self.landmark_indexes))
        self.sigma = 1e-6 * np.diag(sigma_values) # Warning ! potential bug
        self.sigma[0:3,0:3] = sigma
        self.landmark_observed = np.zeros((3 + 2*len(self.landmark_indexes)),dtype=bool)

    def motion_update(self,control,dt):
        if dt < 1e-3:
            return
        # Compute updated robot state
        update = np.array([control[1] * np.cos(self.mu[2]) * dt,
                           control[1] * np.sin(self.mu[2]) * dt,
                           control[2] * dt])
        self.mu[:3] = self.mu[:3] + update
        # Limit θ within [-pi, pi]
        if (self.mu[2] > np.pi):
            self.mu[2] = self.mu[2] - 2 * np.pi
        elif (self.mu[2] < -np.pi):
            self.mu[2] = self.mu[2] + 2 * np.pi
        # Compute linearized state-transition
        G = np.identity(3+2*len(self.landmark_indexes))
        G[0][2] = - control[2] * dt * np.sin(self.mu[2])
        G[1][2] = control[2] * dt * np.cos(self.mu[2]) 
        # Compute Covariance update
        self.sigma = G.dot(self.sigma).dot(G.T)
        self.sigma[:3,:3] = self.sigma[:3,:3] + self.R

    def measurement_update(self,measurement):
        if not measurement[0] in self.landmark_indexes:
            return 
        # get current measurement and landmark index
        landmark_idx = self.landmark_indexes[measurement[0]]
        # get the state values
        x_t = self.mu[0]
        y_t = self.mu[1]
        theta_t = self.mu[2]
        # update state if landmark was never seen
        if not self.landmark_observed[landmark_idx]:
            x_l = x_t + measurement[1] * np.cos(measurement[2] + self.mu[2])
            y_l = y_t + measurement[1] * np.sin(measurement[2] + self.mu[2])
            self.landmark_observed[landmark_idx] = True
            self.mu[2 * landmark_idx + 1] = x_l
            self.mu[2 * landmark_idx + 2] = y_l
        else:
            x_l = self.mu[2*landmark_idx+1]
            y_l = self.mu[2*landmark_idx+2]
        # get the measurement values
        range_t = measurement[1]
        bearing_t = measurement[2]
        delta_x  = x_l - x_t
        delta_y = y_l - y_t
        q = (delta_x)**2 + (delta_y)**2
        range_expected = np.sqrt(q)
        bearing_expected = np.arctan2(delta_y,delta_x) - theta_t
        # Linearize measurement model
        F_x = np.zeros((5,3 + 2 * len(self.landmark_indexes)))
        F_x[0][0] = 1.0
        F_x[1][1] = 1.0
        F_x[2][2] = 1.0
        F_x[3][2*landmark_idx+1] = 1.0
        F_x[4][2*landmark_idx+2] = 1.0
        H_1 = np.array([-delta_x/np.sqrt(q), -delta_y/np.sqrt(q), 0, delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
        H_2 = np.array([delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q])
        H_3 = np.array([0, 0, 0, 0, 0])
        H = np.array([H_1, H_2, H_3]).dot(F_x)  
        # Kalman gain update
        S_t = H.dot(self.sigma).dot(H.T) + self.Q
        K = self.sigma.dot(H.T).dot(np.linalg.inv(S_t))
        # Mean update
        difference = np.array([range_t - range_expected,bearing_t - bearing_expected,0])
        innovation = K.dot(difference)
        self.mu = self.mu + innovation
        # Covariance update 
        self.sigma = (np.identity(3 + 2 * len(self.landmark_indexes)) - K.dot(H)).dot(self.sigma)
 


