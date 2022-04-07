"""
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology

Description : This file contains EKF SLAM  algorithm with 
known-correspondance tested for the UTIAS Multi-Robot Cooperative 
Localization and Mapping Dataset. 
For details about the algorithm please see Probabilistic Robotics
book: Page 321, Table 10.2
****************************************************************
"""

import numpy as np
from .ekf_slam_known_correspondance import EkfSlamKnownCorrespondance

class EkfSlamUnknownCorrespondance(EkfSlamKnownCorrespondance):
    '''
    class that implements the Extended Kalman Filter SLAM
    for unkown data association problem with Manhabolis 
    distance estimation.

    '''

    def data_association(self,measurement):
        if not measurement[0] in self.landmark_indexes:
            return
        
        # Get current robot state, measurement
        x_t = self.mu[0]
        y_t = self.mu[1]
        theta_t = self.mu[2]
        range_t = measurement[1]
        bearing_t = measurement[2]
        # The expected landmark's location based on current robot state 
        landmark_x_expected = x_t + range_t * np.cos(bearing_t + theta_t)
        landmark_y_expected = y_t + range_t * np.sin(bearing_t + theta_t)
        # If the current landmark has not been seen, initialize its location as the expected one 
        landmark_idx = self.landmark_indexes[measurement[0]]
         
        if not self.landmark_observed[landmark_idx] == True:
            self.landmark_observed[landmark_idx] = True
            self.mu[2 * landmark_idx + 1] = landmark_x_expected
            self.mu[2 * landmark_idx + 2] = landmark_y_expected
        # Calculate the Likelihood for each existed landmark
        min_distance = 1e16
        for i in range(1,len(self.landmark_indexes)+1):
            if not self.landmark_observed[i]:
                continue
            # Get current landmark estimate
            x_l = self.mu[2 * i + 1]
            y_l = self.mu[2 * i + 2]
            # calculate expected range and bearing measurement
            delta_x = x_l - x_t
            delta_y = y_l - y_t
            q = delta_x ** 2 + delta_y ** 2
            range_expected = np.sqrt(q)
            bearing_expected = np.arctan2(delta_y, delta_x) - theta_t
            F_x = np.zeros((5,3 + 2 * len(self.landmark_indexes)))
            F_x[0][0] = 1.0
            F_x[1][1] = 1.0
            F_x[2][2] = 1.0
            F_x[3][2*i+1] = 1.0
            F_x[4][2*i+2] = 1.0
            H_1 = np.array([-delta_x/np.sqrt(q), -delta_y/np.sqrt(q), 0, delta_x/np.sqrt(q), delta_y/np.sqrt(q)])
            H_2 = np.array([delta_y/q, -delta_x/q, -1, -delta_y/q, delta_x/q])
            H_3 = np.array([0, 0, 0, 0, 0])
            H = np.array([H_1, H_2, H_3]).dot(F_x)
            # Compute Mahalanobis distance
            Psi = H.dot(self.sigma).dot(H.T) + self.Q
            difference = np.array([range_t - range_expected,bearing_t-bearing_expected,0]) 
            Pi = difference.T.dot(np.linalg.inv(Psi)).dot(difference)
            if Pi < min_distance:
                min_distance = Pi
                # Values for measurement update
                self.landmark_idx = i
                self.H = H 
                self.Psi = Psi 
                self.difference = difference
                self.landmark_expected = np.array([landmark_x_expected, landmark_y_expected])
                self.landmark_current = np.array([x_l, y_l])

    
    def measurement_update(self, measurement):
        if not measurement[0] in self.landmark_indexes:
            return 
        
        # Update mean
        K = self.sigma.dot(self.H.T).dot(np.linalg.inv(self.Psi))
        innovation = K.dot(self.difference)
        self.mu = self.mu + innovation
        # Update covariance 
        self.sigma = (np.identity(3+2*len(self.landmark_indexes)) - K.dot(self.H)).dot(self.sigma)
