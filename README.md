[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# KUKA_WAII
Calculating Forward and Inverse Kinematics for KUKA WAII Robot (7 DoF) and making the end effector move as a circle and plotting the movement.

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
