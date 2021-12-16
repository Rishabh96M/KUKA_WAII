# Group 41 Final Project
# Copyright (c) 2021 Rishabh Mukund, Koundinya Vinnakota
#
# Description: To simulate and test inverse and forward kinematics on Fetch and Freight Robot


# Importing all the required header files
from sympy import cos, sin, pprint
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Defining the lengths s (distance between origins in z direction)
d = [0, 0, 0]

#Defining the alphas
alpha = [4.7123, 0, 0]

#Defining the lengths a (distance between origins in x direction)
a = [10, 10, 10]

th = [0, 0, 0]

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
    print("\n\n")
    pprint(th0n_temp[-1])
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
    print("\n\n")
    pprint(j_1)
    print("\n\n")
    return j_1


# @brief: A function to plot the robotic arm
#
# @param: A vector of transformation matrices with respect to the 0th frame (7 * 4 * 4)
def plot_arm(th0n_temp):
    plot_line(th0n_temp[0], th0n_temp[1])
    plot_line(th0n_temp[1], th0n_temp[2])
    plot_line(th0n_temp[2], th0n_temp[3])
    plot_frame(th0n_temp[0])
    plot_frame(th0n_temp[1])
    plot_frame(th0n_temp[2])
    plot_frame(th0n_temp[3])


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
    ax.plot3D([tx, 3 * rx[0, 0] + tx], [ty, 3 * rx[1, 0] + ty], [tz, 3 * rx[2, 0] + tz], 'red')
    ax.plot3D([tx, 3 * ry[0, 0] + tx], [ty, 3 * ry[1, 0] + ty], [tz, 3 * ry[2, 0] + tz], 'green')
    ax.plot3D([tx, 3 * rz[0, 0] + tx], [ty, 3 * rz[1, 0] + ty], [tz, 3 * rz[2, 0] + tz], 'blue')


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
def calculate_g_q(ja_temp, old_t):
    d = [0.036, 0.021, 0.021, 0.0201, 0.01985, 0.01055, 0.01]                            # Link lenghts matrix in m
    m_i = [3.94781, 4.50275, 2.45520, 2.61155, 3.41, 3.38795, 0.35432]                   # Masses of each link
    g = 9.8                                                                              # Gravity in m/s^2
    pe = 0                                                                               # Initializing Potential energy
    a = [ja_temp[1], ja_temp[1] + ja_temp[3], ja_temp[1] + ja_temp[3] + ja_temp[5]]      # Grouping joint angles
    g_q_temp = sp.Matrix([[0], [0], [0], [0], [0], [0], [0]])                            # Initialising g(q) matrix

    # Height of centre of masses of each link
    h_i = [d[0]/2,
           d[0] + (d[1]*cos(a[0])/2),
           d[0] + d[1]*cos(a[0]) + (d[2]*cos(a[0])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]*cos(a[1])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + d[3]*cos(a[1]) + (d[4]*cos(a[1])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]+d[4])*cos(a[1]) + (d[5]*cos(a[2])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]+d[4])*cos(a[1]) + d[5]*cos(a[2]) + (d[6]*cos(a[2])/2)]

    # Calulating potential energy
    for i in range(0, len(m_i)):
        pe += m_i[i] * g * h_i[i]

    # Computing g(q) matrix
    for i in range(0, len(ja_temp)):
        g_q_temp[i] = pe / ((ja_temp[i] - old_t[i])*(180/np.pi))
    return g_q_temp


if __name__ == '__main__':

    # Initial joint-angles of the arm
    ja = [0, 0, 0, 0, 0, 0, 0]


    # calculating transformation matrices with respect to 0th frame
    translation = create_translation_matrices(ja)
    tm0n = calculate_tm(translation)

    # calculating the jacobian matrix
    j = calculate_jacobian(tm0n)

    # plotting the arm
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_arm(tm0n)

    # Getting the trajectory of the circle
    x, y, z = circle(0, 605, 680, 100, 100)

    # Plotting the circle
    ax.plot3D(x, y, z, 'yo')
    plt.show()

    delta_time = 5 / len(x)                                                       # Total time by number of points to cover
    # Code to follow the trajectory
    for k in range(0, 1):
        curr_pos = tm0n[-1].extract([0, 1, 2], [-1])                              # The position of the end effector
        req_pos = sp.Matrix([[x[k]], [y[k]], [z[k]]])                             # The required position
        rate_pos = (req_pos - curr_pos) / delta_time                              # Rate of change in the position

        # Rate of change in angles, Calculating pseudo inverse
        rate_angle = (j.T * ((j * j.T).inv())) * rate_pos.col_join(sp.Matrix([[0], [0], [0]]))
        print(ja)                                                                 # Printing the joint angles
        for m in range(0, 7):                                                     # Updating joint angles
            ja_prev[m] = ja[m]                                                    # Updating previous angles
            ja[m] = ((ja[m] + (rate_angle[m] * delta_time)) % (2 * 3.14))         # Bounding to 0 to 2*pi

        tm0n = calculate_tm(ja, d)                                   # FK for new position of the arm
        j = calculate_jacobian(tm0n)                                              # Calculate the new jacobian
        ax.plot3D(curr_pos[0], curr_pos[1], curr_pos[2], 'ro')
        plt.pause(delta_time)
    plt.show()
