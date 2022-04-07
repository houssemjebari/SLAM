"""
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology
****************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import *
from ..ekf_slam_known_correspondance import EkfSlamKnownCorrespondance
from ..ekf_slam_unknown_correspondance import EkfSlamUnknownCorrespondance
dataset = 'dataset1/'
R = np.diagflat([100.,100.,100.]) ** 2
Q = np.diagflat([1.,1.,1e16]) ** 2 

groundtruth_data, data, landmark_indexes, landmarks = load_data(dataset)
print(landmarks)
print(landmark_indexes)
##---- Run the EKF ----## 
ekf = EkfSlamUnknownCorrespondance(R,Q,groundtruth_data[0,1:],np.zeros((3,3)),landmark_indexes)
previous_timestep = data[0,0]
for i,d in enumerate(data):
    if d[1] == -1:
        if i == 0:
            ekf.motion_update(d[1:],0)
        else:
            ekf.motion_update(d[1:],d[0]-previous_timestep)
            previous_timestep = d[0]
    else:
        ekf.data_association(d[1:])
        ekf.measurement_update(d[1:])
        previous_timestep = d[0]
    ekf.update_trajectory()
    if(len(ekf.get_trajectory()) % 10 == 0):
        plot_data(groundtruth_data,landmarks,ekf,slam=True,known_correspondance=False)
plt.show()
print("Final State: ",ekf.mu,"\n")
print("Covariance matrix: ",ekf.sigma,"\n")


