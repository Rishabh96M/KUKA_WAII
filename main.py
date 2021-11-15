# KUKA WAII Robot Model
# Copyright (c) 2021 Rishabh Mukund
#
#  Description: This code will generate the Forward and Inverse Kinematics for position and velocity and follow a
#  given trajectory.


# Importing all the required header files
import time

from sympy import cos, sin
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# @brief: A function which will generate transformation matrices with respect to the 0th frame
#
# @param: A vector of joint angles (1 * 6)
# @return: A vector of transformation matrices with respect to the 0th frame (n * 4 * 4)
def calculate_tm(a, d1, d3, d5, d7):
    th = [sp.Matrix([[cos(a[0]), 0, -sin(a[0]), 0], [sin(a[0]), 0, cos(a[0]), 0], [0, -1, 0, d1], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[1]), 0, sin(a[1]), 0], [sin(a[1]), 0, -cos(a[1]), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[2]), 0, -sin(a[2]), 0], [0, 1, 0, 0], [sin(a[2]), 0, cos(a[2]), d3], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[3]), 0, -sin(a[3]), 0], [sin(a[3]), 0, cos(a[3]), 0], [0, -1, 0, d5], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[4]), 0, sin(a[4]), 0], [sin(a[4]), 0, -cos(a[4]), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[5]), -sin(a[5]), 0, 0], [sin(a[5]), cos(a[5]), 0, 0], [0, 0, 1, d7], [0, 0, 0, 1]])]

    # Initializing transformation vector with respect to 0th frame
    th0n_temp = [sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

    # Multiplying transformation matrices
    for i in range(0, 6):
        th0n_temp.append(th0n_temp[i] * th[i])
    return th0n_temp


# @brief: A function to calculate the jacobian matrix for each joint
#
# @param: A vector of transformation matrices with respect to the 0th frame (6 * 4 * 4)
# @return: Jacbian matrices (6 * 6)
def calculate_jacobian(th0n_temp):
    j_temp = []                                       # To extract the Z component of R matrix
    d06 = th0n_temp[-1].extract([0, 1, 2], [-1])      # Extraxting the distance from Transformation Matrix
    for i in th0n_temp:
        ang_vel = i.extract([0, 1, 2], [2])           # Extracting the Z component
        d0i = i.extract([0, 1, 2], [-1])              # Extracting the o matrix
        lin_vel = ang_vel.cross(d06 - d0i)
        j0n = lin_vel.col_join(ang_vel)               # Concatinating lin and ang velocity for a joint
        j_temp.append(j0n)

    j_1 = j_temp[0]                                   # Concatinating for J matrix
    for x in range(1, 6):
        j_1 = j_1.row_join(j_temp[x])
    return j_1


# @brief: A function to plot the robotic arm
#
# @param: A vector of transformation matrices with respect to the 0th frame (n * 4 * 4)
def plot_arm(th0n_temp):
    plot_line(th0n_temp[0], th0n_temp[1])
    plot_line(th0n_temp[1], th0n_temp[2])
    plot_line(th0n_temp[2], th0n_temp[3])
    plot_line(th0n_temp[3], th0n_temp[4])
    plot_line(th0n_temp[4], th0n_temp[5])
    plot_line(th0n_temp[5], th0n_temp[6])
    plot_frame(th0n_temp[0])
    plot_frame(th0n_temp[1])
    plot_frame(th0n_temp[2])
    plot_frame(th0n_temp[3])
    plot_frame(th0n_temp[4])
    plot_frame(th0n_temp[5])
    plot_frame(th0n_temp[6])


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


# @brief: A function to generate and plot a trajectory of a circle
#
# @param: x_off, y_off, z_off are the center of the circle in 3D plane, r is the raduis of the circle and
# s is the precision of the circle
def circle(x_off, y_off, z_off, r, s):
    th1 = np.linspace(0, 2 * 3.14, s)
    x_val = []
    z_val = []
    for i in th1:
        x_val.append(r * sin(i) + x_off)
        z_val.append(r * cos(i) + z_off)
    y_val = np.ones(s) * y_off
    return x_val, y_val, z_val


if __name__ == '__main__':
    # initializing all the link lengths
    d1 = 400
    d3 = 380
    d5 = 400
    d7 = 205

    # Initial joint-angles of the arm
    ja = [1.5708, 0, -1.5708, 0, 0, 0]

    # calculating transformation matrices with respect to 0th frame
    tm0n = calculate_tm(ja, d1, d3, d5, d7)
    print(tm0n)

    # calculating the jacobian matrix
    j = calculate_jacobian(tm0n)
    print(j)

    # plotting the arm and path
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_arm(tm0n)

    # x, y, z = circle(120, 0, 1015, 100, 100)
    x, y, z = circle(0, 605, 680, 100, 100)

    ax.plot3D(x, y, z, 'yo')

    # Code to follow the trajectory
    for k in range(0, 100):
        curr_pos = tm0n[-1].extract([0, 1, 2], [-1])
        req_pos = sp.Matrix([[x[k]], [y[k]], [z[k]]])
        delta_pos = req_pos - curr_pos
        delta_angle = j.inv() * delta_pos.col_join(sp.Matrix([[0], [0], [0]]))
        for m in range(0,6):
            ja[m] = ja[m] + delta_angle[0]
        print(ja)
        tm0n = calculate_tm(ja, d1, d3, d5, d7)
        j = calculate_jacobian(tm0n)
        plot_arm(tm0n)
        plt.pause(0.001)

    plt.show()



