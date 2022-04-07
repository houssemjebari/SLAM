"""
****************************************************************
Author : Houssem Jebari

E-mail : jebari.houssem@insat.u-carthage.tn

Institute : National Institute of Applied Science and Technology

Description : This file provides util functions for testing 
localization and SLAM algorithms
****************************************************************
"""

import numpy as np
import matplotlib.pyplot as plt 

END_FRAME = 3200

def load_data(filename):
    # Barcodes: [Subject#, Barcode#]
    barcodes_data = np.loadtxt(filename + "Barcodes.dat")
    # Groundtruth: [Time(s), x[m], y[m], orientation[rad]]
    groundtruth_data = np.loadtxt(filename + "/Groundtruth.dat")
    # Landmark Groundtruth: [Subject#, x[m], y[m]]
    landmarks_data = np.loadtxt(filename + "/Landmark_Groundtruth.dat")
    # Measurement: [Time(s), Subject#, range[m], bearing[rad]]
    measurement_data = np.loadtxt(filename + "/Measurement.dat")
    # Odometry: [Time(s), Subject#, forward_V[m/s], angular_v[rad_s]]
    odometry_data = np.loadtxt(filename + "/Odometry.dat")
    # Concatenate data and sort by timestep
    odometry_data = np.insert(odometry_data, 1, -1, axis = 1)
    data = np.concatenate((odometry_data, measurement_data), axis = 0)
    data = data[np.argsort(data[:, 0])]
    data = data[:END_FRAME]
    # Remove landmark number 20 measurements (outlier rejection test)
    data = np.delete(data,list(np.where(data[:,1] == 90)[0]),0)
    # Remove all groundtruth outside the range
    start_frame = 0
    while data[start_frame][1] != -1:
        start_frame += 1
    # Remove all data before start_frame and after the end_timestamp
    data = data[start_frame:END_FRAME]
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
    return groundtruth_data,data,landmark_indexes,landmarks

def plot_data(groundtruth_data,landmarks,filter,slam=False,known_correspondance=True):
    plt.style.use('seaborn')
    plt.cla()
    # Plot the groundtruth trajectory
    plt.plot(groundtruth_data[:, 1], groundtruth_data[:, 2], 'b', label="Robot State Ground truth")
    # Plot the estimated trajectory
    estimated_trajectory = np.array(filter.get_trajectory())
    plt.plot(estimated_trajectory[:,0],estimated_trajectory[:,1],'r',label='Estimated trajectory')
    # Plot groundtruth landmarks
    landmark_xs = []
    landmark_ys = []
    for location in landmarks:
        landmark_xs.append(landmarks[location][0])
        landmark_ys.append(landmarks[location][1])
        index = filter.landmark_indexes[location] + 5
        plt.text(landmark_xs[-1], landmark_ys[-1], str(index), alpha=0.5, fontsize=10)
    plt.scatter(landmark_xs, landmark_ys, s=200, c='k', alpha=0.2, marker='*')
    # Plot estimated landmarks
    if slam:
        estimate_xs = []
        estimate_ys = []
        for i in range(1, len(filter.landmark_indexes) + 1):
            if filter.landmark_observed[i]:
                estimate_xs.append(estimated_trajectory[-1][2 * i + 1])
                estimate_ys.append(estimated_trajectory[-1][2 * i + 2])
                plt.text(estimate_xs[-1], estimate_ys[-1], str(i+5), fontsize=10)
        plt.scatter(estimate_xs, estimate_ys, s=50, c='k', marker='.', label='Landmark Estimate')
        if not known_correspondance:
            xs = [filter.landmark_expected[0],filter.landmark_current[0]]
            ys = [filter.landmark_expected[1],filter.landmark_current[1]]
            plt.plot(xs,ys,color='c',label='Data association')
            plt.text(filter.landmark_expected[0], filter.landmark_expected[1], str(filter.landmark_idx +5))
            plt.scatter(filter.landmark_expected[0], filter.landmark_expected[1], s=100, c='r', alpha=0.5, marker='P', label='Current Observed Landmark')
    # Start and end points
    plt.plot(groundtruth_data[0, 1], groundtruth_data[0, 2], 'go', label="Start point")
    plt.plot(groundtruth_data[-1, 1], groundtruth_data[-1, 2], 'yo', label="End point")
    plt.legend()
    plt.pause(1e-16)

