module RoombaPOMDPs

using POMDPs
using Distributions
using StaticArrays
using Parameters
using POMDPModelTools
using Statistics
using Graphics
using Cairo
using Random
using Base64
using ParticleFilters
using NearestNeighbors

import POMDPModelTools: render

export
    RoombaState,
    RoombaAct,
    RoombaMDP,
    RoombaPOMDP,
    RoombaModel,
    Bumper,
    BumperPOMDP,
    Lidar,
    LidarPOMDP,
    DiscreteLidar,
    DiscreteLidarPOMDP,
    RoombaParticleFilter,
    ParticleDeathResampler,
    get_goal_xy,
    wrap_to_pi,
    ContinuousRoombaStateSpace,
    DiscreteRoombaStateSpace,
    render

include("line_segment_utils.jl")
include("env_room.jl")
include("roomba_env.jl")
include("filtering.jl")

end
