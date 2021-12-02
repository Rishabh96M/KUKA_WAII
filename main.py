# KUKA WAII Robot Model
# Copyright (c) 2021 Rishabh Mukund
# MIT License
#
# Description: This code will generate the Forward and Inverse Kinematics for position and velocity and follow a
# given trajectory.


# Importing all the required header files
from sympy import cos, sin, pprint
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# @brief: A function which will generate transformation matrices with respect to the 0th frame
#
# @param: a - A vector of joint angles (1 * 7)
#         d1, d3, d5, d7 - Link lengths of the robot
# @return: A vector of transformation matrices with respect to the 0th frame (8 * 4 * 4)
def calculate_tm(a, d1_temp, d3_temp, d5_temp, d7_temp):
    # Model of the robot
    th = [sp.Matrix([[cos(a[0]), 0, -sin(a[0]), 0], [sin(a[0]), 0, cos(a[0]), 0], [0, -1, 0, d1_temp], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[1]), 0, sin(a[1]), 0], [sin(a[1]), 0, -cos(a[1]), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[2]), 0, sin(a[2]), 0], [sin(a[2]), 0, -cos(a[2]), 0], [0, 1, 0, d3_temp], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[3]), 0, -sin(a[3]), 0], [sin(a[3]), 0, cos(a[3]), 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[4]), 0, -sin(a[4]), 0], [sin(a[4]), 0, cos(a[4]), 0], [0, -1, 0, d5_temp], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[5]), 0, sin(a[5]), 0], [sin(a[5]), 0, -cos(a[5]), 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
          sp.Matrix([[cos(a[6]), -sin(a[6]), 0, 0], [sin(a[6]), cos(a[6]), 0, 0], [0, 0, 1, d7_temp], [0, 0, 0, 1]])]

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
# @param: ja_temp - A vector of joint angles (1 * 7)
# @return: g(q) matrix (1 * 7)
def calculate_g_q(tm0n_temp, ja_temp):
    d = [0.036, 0.021, 0.021, 0.0201, 0.01985, 0.01055, 0.01]
    m_i = [3.94781, 4.50275, 2.45520, 2.61155, 3.41, 3.38795, 0.35432]
    g = 9.8
    pe = 0
    ja_prev = [1.5708, 0, 0, 4.7123, 0, 0, 0]
    a = [ja_temp[1], ja_temp[1] + ja_temp[3], ja_temp[1] + ja_temp[3] + ja_temp[5]]
    g_q_temp = sp.Matrix([[0], [0], [0], [0], [0], [0], [0]])

    h_i = [d[0]/2,
           d[0] + (d[1]*cos(a[0])/2),
           d[0] + d[1]*cos(a[0]) + (d[2]*cos(a[0])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]*cos(a[1])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + d[3]*cos(a[1]) + (d[4]*cos(a[1])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]+d[4])*cos(a[1]) + (d[5]*cos(a[2])/2),
           d[0] + (d[1]+d[2])*cos(a[0]) + (d[3]+d[4])*cos(a[1]) + d[5]*cos(a[2]) + (d[6]*cos(a[2])/2)]

    for i in range(0, len(m_i) - 1):
        pe += m_i[i] * g * h_i[i]

    for i in range(0, len(ja_temp) - 1):
        g_q_temp[i] = pe / (ja_temp[i] - ja_prev[i])

    ja_prev = ja_temp
    return g_q_temp


if __name__ == '__main__':
    # initializing all the link lengths
    d1 = 360
    d3 = 420
    d5 = 399.5
    d7 = 205.5

    # Initial joint-angles of the arm
    ja = [1.5708, 0, 0, 4.7123, 0, 0, 0]
    tor = []
    F = sp.Matrix([[0], [-5], [0], [0], [0], [0]])

    # calculating transformation matrices with respect to 0th frame
    tm0n = calculate_tm(ja, d1, d3, d5, d7)

    # calculating the jacobian matrix
    j = calculate_jacobian(tm0n)

    # plotting the arm
    plt.figure(0)
    ax = plt.axes(projection='3d')
    plot_arm(tm0n)

    # Getting the trajectory of the circle
    x, y, z = circle(0, 605, 680, 100, 100)

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
            ja[m] = ((ja[m] + (rate_angle[m] * delta_time)) % (2 * 3.14))         # Bounding to 0 to 2*pi
        tm0n = calculate_tm(ja, d1, d3, d5, d7)                                   # FK for new position of the arm
        j = calculate_jacobian(tm0n)                                              # Calculate the new jacobian
        g_q = calculate_g_q(tm0n,ja)
        tor.append(g_q - (j.T * F))
        ax.plot3D(curr_pos[0], curr_pos[1], curr_pos[2], 'ro')
        plt.pause(delta_time/100)

    tor_1 = []
    tor_2 = []
    tor_3 = []
    tor_4 = []
    tor_5 = []
    tor_6 = []
    tor_7 = []
    for m in range(0, len(tor)):
        tor_1.append(tor[m].extract([0],[0]))
        tor_2.append(tor[m].extract([1], [0]))
        tor_3.append(tor[m].extract([2], [0]))
        tor_4.append(tor[m].extract([3], [0]))
        tor_5.append(tor[m].extract([4], [0]))
        tor_6.append(tor[m].extract([5], [0]))
        tor_7.append(tor[m].extract([6], [0]))

    plt.clf()
    fig, axis = plt.subplots(2, 4)
    axis[0, 0].plot(np.linspace(1, 200, 100), tor_1)
    axis[0, 0].set_title("Torque of Joint 1")
    axis[0, 1].plot(np.linspace(1, 200, 100), tor_2)
    axis[0, 1].set_title("Torque of Joint 2")
    axis[0, 2].plot(np.linspace(1, 200, 100), tor_3)
    axis[0, 2].set_title("Torque of Joint 3")
    axis[0, 3].plot(np.linspace(1, 200, 100), tor_4)
    axis[0, 3].set_title("Torque of Joint 4")
    axis[1, 0].plot(np.linspace(1, 200, 100), tor_5)
    axis[1, 0].set_title("Torque of Joint 5")
    axis[1, 1].plot(np.linspace(1, 200, 100), tor_6)
    axis[1, 1].set_title("Torque of Joint 6")
    axis[1, 2].plot(np.linspace(1, 200, 100), tor_7)
    axis[1, 2].set_title("Torque of Joint 7")
    plt.show()

