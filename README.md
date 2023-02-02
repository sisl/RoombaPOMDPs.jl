# RoombaPOMDPs

[![CI](https://github.com/sisl/RoombaPOMDPs.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sisl/RoombaPOMDPs.jl/actions/workflows/ci.yml)

Roomba is a localization POMDP problem. A robotic vacuum cleaner, also known as Roomba, finds itself in a familiar room but does not know its exact position. It then tries to locate itself with the observations it receives. Roomba can equip two types of sensors, Lidar or Bumper. 
With a Lidar sensor, the robot receives a noisy Lidar measurement indicating the distance to obstacles in the front.
Equipped with a Bumper sensor, it can sense its collision.

An example video showing the robot first localizing itself using its bump sensors, then navigating safely to the goal. The Roomba's belief about where it may be located is represented by the blue regions, and is updated using a particle filter.

<img src="demo.gif" width="450">

## Installation
Run the following commands in Julia:
```julia
import POMDPs
POMDPs.add_registry()
using Pkg
Pkg.add(url="https://github.com/sisl/RoombaPOMDPs.git")
```

## Getting Started
Run the ```escape_roomba.ipynb``` jupyter notebook to become familiar with the Roomba environment. This will walk you through a step-by-step example of how to set up the environment, define a baseline policy, and evaluate the performance of the policy.

Next, familiarize yourself with the source code by examining the files in the ```src``` directory. A brief description of the files is given below:
* ```RoombaPOMDPs.jl``` - defines the package module for this project and includes the necessary import and export statements
* ```roomba_env.jl``` - defines the environment as a POMDPs.jl MDP and POMDP
* ```env_room.jl``` - defines the environment room and rectangles used to define it
* ```line_segment_utils.jl``` - functions for determining whether the Roomba's path interects with a line segment and struct defining line segments
* ```filtering.jl``` - specification of particle filters for the Roomba environments
