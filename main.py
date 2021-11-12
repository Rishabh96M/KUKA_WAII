from sympy import cos, sin, Symbol, pprint
import sympy as sp


def calculate_tm(th_temp):
    # Initializing transformation vector with respect to 0th frame
    th0n_temp = [sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])]

    # Multiplying transformation matrices
    for i in range(0, 7):
        th0n_temp.append(th0n_temp[i] * th_temp[i])

    return th0n_temp


def calculate_jacobian(th0n_temp):
    j_temp = []
    z_temp = sp.Matrix([[0], [0], [1]])
    d07 = th0n_temp[-1].extract([0, 1, 2], [-1])
    for i in range(0, 7):
        if i == 2:
            th0n_temp[i+1] *= th0n_temp[i]
            i += 1
            continue
        ang_vel = th0n_temp[i].extract([0, 1, 2], [0, 1, 2]) * z_temp
        d0i = th0n_temp[i].extract([0, 1, 2], [-1])
        lin_vel = ang_vel.cross(d07 - d0i)
        j0n = lin_vel.col_join(ang_vel)
        j_temp.append(j0n)
    return j_temp


if __name__ == '__main__':
    # initializing all variables as symbols
    d1 = Symbol("d1", positive=True)
    d3 = Symbol("d3", positive=True)
    d5 = Symbol("d5", positive=True)
    d7 = Symbol("d7", positive=True)
    t1 = Symbol("t1", positive=True)
    t2 = Symbol("t2", positive=True)
    t3 = Symbol("t3", positive=True)
    t4 = Symbol("t4", positive=True)
    t5 = Symbol("t5", positive=True)
    t6 = Symbol("t6", positive=True)
    t7 = Symbol("t7", positive=True)

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

    print("The FK matrix is given by: ")
    for x in th0n:
        print(x)

    # calculating the jacobian matrix
    j = calculate_jacobian(th0n)

    print("The J matrix is given by: ")
    for x in j:
        print(x)

    # j_lol = j[0]
    # for i in range(1, 6):
    #     j_lol.row_join(j[i])
    #     print(j_lol)