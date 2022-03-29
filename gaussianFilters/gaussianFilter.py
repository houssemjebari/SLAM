'''
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology

Description : This file contains Abstract implementation for 
gaussian filters provided in the Probabilistic robotics book
****************************************************************
'''

import numpy as np 

class GaussianFilter():
    '''
    abstract class that implements basic gaussian localization 
    and SLAM algorithms functions
    
    attributes:
        * R: motion covariance matrix 
        * Q: measurement covariance matrix 
        * mu: robot state vector
        * sigma: robot state covariance
    '''

    def __init__(self):
        self.R = np.array([])
        self.Q = np.array([])
        self.mu = np.array([])
        self.sigma = np.array([])
        self.trajectory_estimate = []

    def __init__(self,R,Q,mu,sigma):
        self.R = R
        self.Q = Q 
        self.mu = mu
        self.sigma = sigma
        self.trajectory_estimate = []

    def __str__(self):
        return ()
    

    def get_state(self):
        return self.mu
    
    def get_covariance(self):
        return self.sigma

    def get_trajectory(self):
        return self.trajectory_estimate
    
    def update_trajectory(self):
        self.trajectory_estimate.append(self.mu)
    
    def motion_update(self,control,dt):
        raise NotImplementedError('Abstract class doesn\'t provide implementation for motion update')

    def measurement_update(self,measurement):
        raise NotImplementedError('Abstract class doesn\'t provide implementation for measurement update')
