# KUKA WAII Robot Model
# Copyright (c) 2021 Rishabh Mukund
# MIT License
#
# Description: This code will generate the Forward and Inverse Kinematics for position and velocity and follow a
# given trajectory.


# Importing all the required header files
from sympy import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

#Defining the lengths s (distance between origins in z direction)
d = [0.360, 0, 0.420, 0, 0.4005, 0, 0.2055]

#Defining the alphas
alpha = [4.7123, 1.5708, 1.5708, 4.7123, 4.7123, 1.5708, 0]

#Defining the lengths a (distance between origins in x direction)
a = [0, 0, 0, 0, 0, 0, 0]

#Defining offset in theta
th = [0, 0, 0, 0, 0, 0, 0]

# Masses of each link
m_i = [3.94781, 4.50275, 2.45520, 2.61155, 3.41, 3.38795, 0.35432]

def create_translation_matrices(theta):
    trasformation_matrices = []
    for i in range(0, len(d)):
        trasformation_matrices.append(sp.Matrix([[cos(theta[i] + th[i]), -sin(theta[i] + th[i])*cos(alpha[i] + th[i]), sin(theta[i] + th[i])*sin(alpha[i] + th[i]), a[i]*cos(theta[i] + th[i])],
                                        [sin(theta[i] + th[i]), cos(theta[i] + th[i])*cos(alpha[i] + th[i]), -cos(theta[i] + th[i])*sin(alpha[i] + th[i]), a[i]*sin(theta[i] + th[i])],
                                        [0, sin(alpha[i] + th[i]), cos(alpha[i] + th[i]), d[i]],
                                        [0, 0, 0, 1]]))
    return trasformation_matrices

# @brief: A function which will generate transformation matrices with respect to the 0th frame
#
# @param: a - A vector of joint angles (1 * 7)
#         d1, d3, d5, d7 - Link lengths of the robot
# @return: A vector of transformation matrices with respect to the 0th frame (8 * 4 * 4)
def calculate_tm(th):
    # Initializing transformation vector with respect to 0th frame
    th0n_temp = [sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

    # Multiplying transformation matrices
    for i in range(0, len(th)):
        th0n_temp.append(th0n_temp[i] * th[i])
    return th0n_temp



# @brief: A function to calculate the jacobian matrix for each joint
#
# @param: A vector of transformation matrices with respect to the 0th frame (8 * 4 * 4)
# @return: Jacbian matrices (6 * 7)
def calculate_jacobian(th0n_temp):
    j_temp = []
    d07 = th0n_temp[-1].extract([0, 1, 2], [-1])                # Extracting dn from T0n matrix
    for i in range(0, len(th0n_temp) - 1):
        ang_vel = th0n_temp[i].extract([0, 1, 2], [2])          # Extracting Zi from T0i matrix
        d0i = th0n_temp[i].extract([0, 1, 2], [-1])             # Extracting di from T0i matrix
        lin_vel = ang_vel.cross(d07 - d0i)                      # Computing linear velcity
        j0n = lin_vel.col_join(ang_vel)                         # Concatinating linear and angular velocities
        j_temp.append(j0n)
    j_1 = j_temp[0]                                             # Concatinating for J matrix
    for xi in range(1, len(j_temp)):
        j_1 = j_1.row_join(j_temp[xi])
    return j_1


# @brief: A function to plot the robotic arm
#
# @param: A vector of transformation matrices with respect to the 0th frame (7 * 4 * 4)
def plot_arm(th0n_temp):
    for i in range(0, len(th0n_temp) - 1):
        plot_line(th0n_temp[i], th0n_temp[i + 1])
        plot_frame(th0n_temp[i])
    plot_frame(th0n_temp[-1])    


# @brief: A function to plot a line in 3D space
#
# @param: two transformation matrices between whoch the line is to be plotted (4 * 4)
def plot_line(f1, f2):
    ax.plot3D([f1[0, 3], f2[0, 3]], [f1[1, 3], f2[1, 3]], [f1[2, 3], f2[2, 3]], 'gray')


# @brief: A function to plot frame at a joint
#
# @param: transformation matrix of the joint (4 * 4)
def plot_frame(f):
    rx = f[:, 0]       # extracting the rotation about X axis
    ry = f[:, 1]       # extracting the rotation about Y axis
    rz = f[:, 2]       # extracting the rotation about Z axis
    tx = f[0, 3]       # extracting the X component of poistion
    ty = f[1, 3]       # extracting the Y component of poistion
    tz = f[2, 3]       # extracting the Z component of poistion
    ax.plot3D([tx, 0.01 * rx[0, 0] + tx], [ty, 0.01 * rx[1, 0] + ty], [tz, 0.01 * rx[2, 0] + tz], 'red')
    ax.plot3D([tx, 0.01 * ry[0, 0] + tx], [ty, 0.01 * ry[1, 0] + ty], [tz, 0.01 * ry[2, 0] + tz], 'green')
    ax.plot3D([tx, 0.01 * rz[0, 0] + tx], [ty, 0.01 * rz[1, 0] + ty], [tz, 0.01 * rz[2, 0] + tz], 'blue')


# @brief: A function to generate and plot a trajectory of a circle in XZ plane
#
# @param: x_off, y_off, z_off - are the center of the circle in 3D plane,
#         r - the raduis of the circle and
#         s - the precision of the circle
# @return: x_val, y_val, z_val - Coordinates of the circle
def circle(x_off, y_off, z_off, r, s):
    th1 = np.linspace(2 * 3.14, 0, s)              # Equally spaced angles of a circle
    x_val = []
    z_val = []
    for i in th1:
        x_val.append(r * sin(i) + x_off)           # Vector of X Cordinates
        z_val.append(r * cos(i) + z_off)           # Vector of Z Cordinates
    y_val = np.ones(s) * y_off
    return x_val, y_val, z_val


# @brief: A function to calculate g(q) matrix
#
# @param: ja_temp - A vector of current joint angles (1 * 7)
#         old_t - A vector of previous joint angles (1 * 7)
# @return: g(q) matrix (1 * 7)
def calculate_g_q(tm0n_temp):
    g = 9.8                                                                              # Gravity in m/s^2
    g_q_temp = sp.Matrix([[0], [0], [0], [0], [0], [0], [0]])                            # Initialising g(q) matrix

    # Height of centre of masses of each link
    z1 = tm0n_temp[1][2, 3]
    z2 = tm0n_temp[2][2, 3]
    z3 = tm0n_temp[3][2, 3]
    z4 = tm0n_temp[4][2, 3]
    z5 = tm0n_temp[5][2, 3]
    z6 = tm0n_temp[6][2, 3]
    z7 = tm0n_temp[7][2, 3]
    h_i = [(z1)/2, (z2 - z1)/2, (z3 - z2)/2, (z4 - z3)/2, (z5 - z4)/2, (z6 - z5)/2, (z7 - z6)/2]

    # Computing g(q) matrix
    for i in range(0, len(h_i)):
        g_q_temp[i] = m_i[i] * g * h_i[i]

    return g_q_temp


if __name__ == '__main__':

    # Initial joint-angles of the arm
    ja = [1.57, 0, 0, 4.71, 0, 0, 0]

    # calculating transformation matrices with respect to 0th frame
    translation = create_translation_matrices(ja)
    tm0n = calculate_tm(translation)

    # initializing torque vector
    tor = []

    # initializing previous angles
    ja_prev = [0, 0, 0, 0, 0, 0, 0]
    F = sp.Matrix([[0], [-5], [0], [0], [0], [0]])

    # calculating the jacobian matrix
    j = calculate_jacobian(tm0n)

    # plotting the arm
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_arm(tm0n)

    # Getting the trajectory of the circle
    x, y, z = circle(0, 0.605, 0.680, 0.100, 200)

    # Plotting the circle
    ax.plot3D(x, y, z, 'yo')

    delta_time = 200 / len(x)                                                 # Total time by number of points to cover
    # Code to follow the trajectory
    for k in range(0, len(x)):
        curr_pos = tm0n[-1].extract([0, 1, 2], [-1])                              # The position of the end effector
        req_pos = sp.Matrix([[x[k]], [y[k]], [z[k]]])                             # The required position
        rate_pos = (req_pos - curr_pos) / delta_time                              # Rate of change in the position

        # Rate of change in angles, Calculating pseudo inverse
        rate_angle = (j.T * ((j * j.T).inv())) * rate_pos.col_join(sp.Matrix([[0], [0], [0]]))
        print(ja)                                                                 # Printing the joint angles
        for m in range(0, 7):                                                     # Updating joint angles
            ja_prev[m] = ja[m]                                                    # Updating previous angles
            ja[m] = ((ja[m] + (rate_angle[m] * delta_time)) % (2 * 3.14))         # Bounding to 0 to 2*pi

        translation = create_translation_matrices(ja)
        tm0n = calculate_tm(translation)                               # FK for new position of the arm
        j = calculate_jacobian(tm0n)                                            # Calculate the new jacobian
        g_q = calculate_g_q(tm0n)                                    # Calculating g(q) matrix
        tor.append(g_q - (j.T * F))                                               # Calculating the torque at each joint
        ax.plot3D(curr_pos[0], curr_pos[1], curr_pos[2], 'ro')
        plt.pause(delta_time/200)

    # Plotting the torque graphs
    tor_1 = []
    tor_2 = []
    tor_3 = []
    tor_4 = []
    tor_5 = []
    tor_6 = []
    tor_7 = []
    for m in range(0, len(tor)):
        tor_1.append(tor[m].extract([0], [0]))
        tor_2.append(tor[m].extract([1], [0]))
        tor_3.append(tor[m].extract([2], [0]))
        tor_4.append(tor[m].extract([3], [0]))
        tor_5.append(tor[m].extract([4], [0]))
        tor_6.append(tor[m].extract([5], [0]))
        tor_7.append(tor[m].extract([6], [0]))

    plt.clf()
    _, axis = plt.subplots(2, 4)
    axis[0, 0].plot(np.linspace(0, 200, len(tor_1)), tor_1)
    axis[0, 0].set_title("Torque of Joint 1")
    axis[0, 1].plot(np.linspace(0, 200, len(tor_1)), tor_2)
    axis[0, 1].set_title("Torque of Joint 2")
    axis[0, 2].plot(np.linspace(0, 200, len(tor_1)), tor_3)
    axis[0, 2].set_title("Torque of Joint 3")
    axis[0, 3].plot(np.linspace(0, 200, len(tor_1)), tor_4)
    axis[0, 3].set_title("Torque of Joint 4")
    axis[1, 0].plot(np.linspace(0, 200, len(tor_1)), tor_5)
    axis[1, 0].set_title("Torque of Joint 5")
    axis[1, 1].plot(np.linspace(0, 200, len(tor_1)), tor_6)
    axis[1, 1].set_title("Torque of Joint 6")
    axis[1, 2].plot(np.linspace(0, 200, len(tor_1)), tor_7)
    axis[1, 2].set_title("Torque of Joint 7")
    plt.show()
