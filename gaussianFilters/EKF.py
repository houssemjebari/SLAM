"""
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology

Description : This file contains EKF localization algorithm
with known-correspondance tested for the UTIAS Multi-Robot Coop-
erative Localization and Mapping Dataset. 
For details about the algorithm please see Probabilistic Robotics
book: Page 204, Table 7.2 
****************************************************************
"""

import numpy as np 
from gaussianFilter import GaussianFilter

class ExtendedKalmanFilter(GaussianFilter):
    '''
    class that implements the Extended Kalman Filter algorithm
    
    attributes:
        * R: motion covariance matrix 
        * Q: measurement covariance matrix 
        * mu: robot state vector
        * sigma: robot state covariance
    '''

    def __init__(self):
        GaussianFilter.__init__(self)
        self.landmark_locations = []
        

    def __init__(self,R,Q,mu,sigma,landmark_locations):
        GaussianFilter.__init__(self,R,Q,mu,sigma)
        self.landmark_locations = landmark_locations
        
    def __str__(self):
        string ='This class contains the state estimation for a mobile robot\nThe current pose estimate of the robot is:  %.2f \nThe current covariance of the estimate is: {}%.2f',self.mu, self.sigma

    
    def motion_update(self,control,dt):
        if dt < 1e-3:
            return
        update = np.array([control[1] * np.cos(self.mu[2]) * dt,
                           control[1] * np.sin(self.mu[2]) * dt,
                           control[2] * dt])
        G_1 = np.array([1, 0, - control[1] * dt * np.sin(self.mu[2])])
        G_2 = np.array([0, 1, control[1] * dt * np.cos(self.mu[2])])
        G_3 = np.array([0, 0, 1])
        G = np.array([G_1, G_2, G_3])
        self.mu = self.mu +  update
        # Limit θ within [-pi, pi]
        if (self.mu[2] > np.pi):
            self.mu[2] -= 2 * np.pi
        elif (self.mu[2] < -np.pi):
            self.mu[2] += 2 * np.pi
        self.sigma = G.dot(self.sigma).dot(G.T) + self.R 


    def measurement_update(self,measurement):
        if not measurement[0] in self.landmark_locations:
            return 
        x_l = self.landmark_locations[measurement[0]][0]
        y_l = self.landmark_locations[measurement[0]][1]
        x_t = self.mu[0]
        y_t = self.mu[1]
        delta = self.mu[:2] - self.landmark_locations[measurement[0]][:2]
        q = delta.T.dot(delta)
        q = (x_l - x_t) * (x_l - x_t) + (y_l - y_t) * (y_l - y_t)
        # Define measurement model jacobian 
        H_1 = np.array([-(x_l - x_t) / np.sqrt(q), -(y_l - y_t) / np.sqrt(q), 0])
        H_2 = np.array([(y_l - y_t) / q, -(x_l - x_t) / q, -1])
        H_3 = np.array([0, 0, 0])
        H = np.array([H_1, H_2, H_3])
        # Calculate predicted measurement and the difference between predicted and true measurement
        z_bar = np.array([np.sqrt(q),np.arctan2(delta[1],delta[0]) - self.mu[2],0.])
        z = np.append(measurement[1:],[0.])
        # Perform mean and covariance update
        S_t = H.dot(self.sigma).dot(H.T) + self.Q
        K = self.sigma.dot(H.T).dot(np.linalg.inv(S_t))
        self.mu += K.dot(z - z_bar)
        self.sigma = (np.identity(3) - K.dot(H)).dot(self.sigma)
        
