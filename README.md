[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# Kinematic Modeling and Torque Analysis of a KUKA 7 DoF Robotic Arm
Modeling a KUKA 7 DoF robotic arm and analyzing its torque requirements while tracing a given path with a specified velocity profile. The kinematic model was established using DH parameters, enabling the arm to accurately follow the desired path. Torque analysis was conducted to determine the torque graphs for each joint over time, providing insights into the arm's performance and power requirements. The results contribute to optimizing the arm's design and ensuring safe and efficient operation during velocity-controlled path tracing tasks.

Verified on python 3.8.10 and packages used are SymPy, NumPy, Matplotlib

Note: Joint 3 is fixed.
d1, d3, d5 and d7 are the link lengths of the robotic arm.
Total time to trace the circle is 5 seconds.


## Steps to run
To clone the file:
```
git clone https://github.com/rishabh96m/KUKA_WAII.git
```
To run the file:
```
cd KUKA_WAII/
python3 main.py
```

## To install the dependencies
```
sudo pip install matplotlib
sudo pip install sympy
sudo pip install numpy
```
