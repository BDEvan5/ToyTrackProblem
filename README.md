ToyTrackProblem

# Files

## main
Includes the functions which can be run. 
Currently functions to:
+ Follow a min curve (optimal) trajectory
+ Train the modification vehicle
+ Test a vehicle on the track

## Agent Mod
A vehicle which plans according to the modification architecture.
Uses a neural network in conjunction with a pure pursuit path follower.

## Agent Optimal
Simply follows the reference trajectory using pure pursuit

## Models RL
Holds implementations of the TD3 and DQN algorithms

## SimMaps
Classes for importing race track maps and random forest maps

## TestingICRA
The script used for generating the results presented in the paper submitted to ICRA

## Simulator
Simulators which represent bicycle car dynamics and a set of range finders

## TrajectoryPlanner
Optimisation code which finds a shortest or minimum curvature trajectory
