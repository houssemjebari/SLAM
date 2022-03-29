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
import matplotlib.pyplot as plt
from EKF import ExtendedKalmanFilter

dataset = '0.Dataset1/'
end_frame = 3200
R = np.diagflat([1.,1.,1.]) ** 2
Q = np.diagflat([350.,350.,1e16]) ** 2 

##---- Load files data ----##
# Barcodes: [Subject#, Barcode#]
barcodes_data = np.loadtxt(dataset + "Barcodes.dat")
# Groundtruth: [Time(s), x[m], y[m], orientation[rad]]
groundtruth_data = np.loadtxt(dataset + "/Groundtruth.dat")
# Landmark Groundtruth: [Subject#, x[m], y[m]]
landmarks_data = np.loadtxt(dataset + "/Landmark_Groundtruth.dat")
# Measurement: [Time(s), Subject#, range[m], bearing[rad]]
measurement_data = np.loadtxt(dataset + "/Measurement.dat")
# Odometry: [Time(s), Subject#, forward_V[m/s], angular_v[rad_s]]
odometry_data = np.loadtxt(dataset + "/Odometry.dat")
# Concatenate data and sort by timestep
odometry_data = np.insert(odometry_data, 1, -1, axis = 1)
data = np.concatenate((odometry_data, measurement_data), axis = 0)
data = data[np.argsort(data[:, 0])]
data = data[:end_frame]
# Remove all groundtruth outside the range
start_frame = 0
while data[start_frame][1] != -1:
    start_frame += 1
# Remove all data before start_frame and after the end_timestamp
data = data[start_frame:end_frame]
start_timestamp = data[0,0]
end_timestamp = data[-1,0]
for i in range(len(groundtruth_data)):
    if (groundtruth_data[i][0] >= end_timestamp):
        break
groundtruth_data = groundtruth_data[:i]
for i in range(len(groundtruth_data)):
    if (groundtruth_data[i,0] >= start_timestamp):
        break
groundtruth_data = groundtruth_data[i:]

##---- Create a landmark locations lookup table ----##
landmarks = {}
for i in range(5, len(barcodes_data), 1):
    landmarks[barcodes_data[i][1]] = landmarks_data[i - 5][1:] 

##---- Create a landmark indexes lookup table ----##
landmark_indexes = {}
for i in range(5,len(barcodes_data),1):
    landmark_indexes[barcodes_data[i][1]] = i - 4 


# Run the EKF 
ekf = ExtendedKalmanFilter(R,Q,groundtruth_data[0,1:],np.zeros((3,3)),landmarks)
previous_timestep = data[0,0]
for i,d in enumerate(data):
    if d[1] == -1:
        if i == 0:
            ekf.motion_update(d[1:],0)
        else:
            ekf.motion_update(d[1:],d[0]-previous_timestep)
            previous_timestep = d[0]
    else:
        ekf.measurement_update(d[1:])
        previous_timestep = d[0]
    ekf.update_trajectory()
print("Final State: ",ekf.mu,"\n")
print("Covariance matrix: ",ekf.sigma,"\n")

# Plot the results 
plt.style.use('seaborn')
plt.plot(groundtruth_data[:, 1], groundtruth_data[:, 2], 'b', label="Robot State Ground truth")
estimated_trajectory = np.array(ekf.get_trajectory())
plt.plot(estimated_trajectory[:,0],estimated_trajectory[:,1],'r',label='Estimated trajectory')
landmark_xs = []
landmark_ys = []
for location in landmarks:
    landmark_xs.append(landmarks[location][0])
    landmark_ys.append(landmarks[location][1])
    plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*')
# Start and end points
plt.plot(groundtruth_data[0, 1], groundtruth_data[0, 2], 'go', label="Start point")
plt.plot(groundtruth_data[-1, 1], groundtruth_data[-1, 2], 'yo', label="End point")
plt.legend()
plt.show()

