# KUKA WAII Robot Model
# Copyright (c) 2021 Rishabh Mukund
#
#  Description: This code will generate the Forward and Inverse Kinematics for position and velocity and follow a
#  given trajectory.


# Importing all the required header files
from sympy import cos, sin
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# @brief: A function which will generate transformation matrices with respect to the 0th frame
#
# @param: A vector of transformation matrices with respect to the previous frame (n * 4 * 4)
# @return: A vector of transformation matrices with respect to the 0th frame (n * 4 * 4)
def calculate_tm(th_temp):
    # Initializing transformation vector with respect to 0th frame
    th0n_temp = [sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

    # Multiplying transformation matrices
    for i in range(0, 7):
        th0n_temp.append(th0n_temp[i] * th_temp[i])
    return th0n_temp


# @brief: A function to calculate the jacobian matrix for each joint
#
# @param: A vector of transformation matrices with respect to the 0th frame (n * 4 * 4)
# @return: A vector of Jacbian matrices (n * 6 * 1)
def calculate_jacobian(th0n_temp):
    j_temp = []
    z_temp = sp.Matrix([[0], [0], [1]])               # To extract the Z component of R matrix
    d07 = th0n_temp[-1].extract([0, 1, 2], [-1])      # Extraxting the distance from Transformation Matrix
    for i in range(0, 7):
        if i == 2:                                    # Assuimg joint 3 as fixed joint
            th0n_temp[i + 1] *= th0n_temp[i]
            continue
        ang_vel = th0n_temp[i].extract([0, 1, 2], [0, 1, 2]) * z_temp  # Extracting the Z matrix
        d0i = th0n_temp[i].extract([0, 1, 2], [-1])                    # Extracting the O matrix
        lin_vel = ang_vel.cross(d07 - d0i)
        j0n = lin_vel.col_join(ang_vel)                                # Concatinating lin and ang velocity for a joint
        j_temp.append(j0n)
    return j_temp

# @brief: A function to plot the robotic arm
#
# @param: A vector of transformation matrices with respect to the 0th frame (n * 4 * 4)
def plot_arm(th0n_temp):
    plot_line(th0n_temp[0], th0n_temp[1])
    plot_line(th0n_temp[1], th0n_temp[2])
    plot_line(th0n_temp[2], th0n_temp[4])
    plot_line(th0n_temp[4], th0n_temp[5])
    plot_line(th0n_temp[5], th0n_temp[6])
    plot_line(th0n_temp[6], th0n_temp[7])
    plot_frame(th0n_temp[0])
    plot_frame(th0n_temp[1])
    plot_frame(th0n_temp[2])
    plot_frame(th0n_temp[4])
    plot_frame(th0n_temp[5])
    plot_frame(th0n_temp[6])
    plot_frame(th0n_temp[7])


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
def plot_circle(x_off, y_off, z_off, r, s):
    th1 = np.linspace(0, 2 * 3.14, s)
    x = []
    z = []
    for i in th1:
        x.append(r * cos(i) + x_off)
        z.append(r * sin(i) + z_off)
    y = np.ones(s) * y_off
    ax.plot3D(x, y, z, 'yo')


if __name__ == '__main__':
    # initializing all the link lengths
    d1 = 400
    d3 = 380
    d5 = 400
    d7 = 205

    # Initial postion of the arm
    t1 = 1.5708
    t2 = 0
    t3 = 0
    t4 = -1.5708
    t5 = 0
    t6 = 0
    t7 = 0

    # model of the robot (transformation matrices)
    th = [sp.Matrix([[cos(t1), 0, -sin(t1), 0], [sin(t1), 0, cos(t1), 0], [0, -1, 0, d1], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t2), 0, sin(t2), 0], [sin(t2), 0, -cos(t2), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t3), 0, sin(t3), 0], [sin(t3), 0, -cos(t3), 0], [0, 1, 0, d3], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t4), 0, -sin(t4), 0], [sin(t4), 0, cos(t4), 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t5), 0, -sin(t5), 0], [sin(t5), 0, cos(t5), 0], [0, -1, 0, d5], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t6), 0, sin(t6), 0], [sin(t6), 0, -cos(t6), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(t7), -sin(t7), 0, 0], [sin(t7), cos(t7), 0, 0], [0, 1, 0, d7], [0, 0, 0, 1]])]

    # calculating transformation matrices with respect to 0th frame
    th0n = calculate_tm(th)

    print("\n\nThe transformation matrices are: ")
    for x in th0n:
        print(x)

    # calculating the individual jacobian matrices
    j_indv = calculate_jacobian(th0n)

    # Concatenating and printing J matrix
    j = j_indv[0]
    for x in range(1, 6):
        j = j.row_join(j_indv[x])
    print("\n\nThe J matrix is given by: ")
    print(j)

    # Calculating Inverse J Matrix
    print("\n\nThe J inv matrix is given by: ")
    print(j.inv())

    # plotting the arm and path
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_arm(th0n)
    plot_circle(0, 605, 680, 100, 100)
    plt.show()
