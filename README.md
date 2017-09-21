Motion-planning-walking
=======================

Walking pattern generator for a biped robot (HRP4) based on MPC (Model-Predictive Control).
It includes rotations of the feet and torso. 
Generates trajectories for feet and CoM which can be tracked with a whole-body controller.

Dependencies
============

Needs qpOASES solver python package installed, numpy, scipy and matplotlib.

Examples
========

Walking pattern with feet rotation
![Alt text](media/walking_pattern.png?raw=true "Walking Pattern")

Generated feet and CoM trajectories
![Alt text](media/trajectories.png?raw=true "Trajectories")
