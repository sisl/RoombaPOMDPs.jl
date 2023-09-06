# Defines the environment as a POMDPs.jl MDP and POMDP
# maintained by {jmorton2,kmenda}@stanford.edu

# Wraps ang to be in (-pi, pi]
function wrap_to_pi(ang::Float64)
    if ang > pi
		ang -= 2*pi
	elseif ang <= -pi
		ang += 2*pi
    end
	ang
end

"""
State of a Roomba.

# Fields
- `x::Float64` x location in meters
- `y::Float64` y location in meters
- `theta::Float64` orientation in radians
- `status::Float64` indicator whether robot has reached goal state or stairs
"""
struct RoombaState <: FieldVector{4, Float64}
    x::Float64
    y::Float64
    theta::Float64
    status::Float64
end

# Struct for a Roomba action
struct RoombaAct <: FieldVector{2, Float64}
    v::Float64     # meters per second
    omega::Float64 # theta dot (rad/s)
end

# action spaces
struct RoombaActions end

function gen_amap(aspace::RoombaActions)
    return nothing
end

function gen_amap(aspace::AbstractVector{RoombaAct})
    return Dict(aspace[i]=>i for i in 1:length(aspace))
end

"""
Define the Roomba MDP.

# Fields
- `v_max::Float64` maximum velocity of Roomba [m/s]
- `om_max::Float64` maximum turn-rate of Roombda [rad/s]
- `dt::Float64` simulation time-step [s]
- `contact_pen::Float64` penalty for wall-contact
- `time_pen::Float64` penalty per time-step
- `goal_reward::Float64` reward for reaching goal
- `stairs_penalty::Float64` penalty for reaching stairs
- `config::Int` specifies room configuration (location of stairs/goal) {1,2,3}
- `room::Room` environment room struct
- `sspace::SS` environment state-space (ContinuousRoombaStateSpace or DiscreteRoombaStateSpace)
- `aspace::AS` environment action-space struct
"""
mutable struct RoombaMDP{SS, AS, S} <: MDP{S, RoombaAct}
    v_max::Float64
    om_max::Float64
    dt::Float64
    contact_pen::Float64
    time_pen::Float64
    goal_reward::Float64
    stairs_penalty::Float64
    discount::Float64
    config::Int
    sspace::SS
    room::Room
    aspace::AS
    _amap::Union{Nothing, Dict{RoombaAct, Int}}
end

function RoombaMDP(;v_max=2.0, om_max=1.0, dt=0.5, contact_pen=-1.0, time_pen=-0.1, goal_reward=10.0, stairs_penalty=-10.0,
                    discount=0.95, config=1, sspace::SS=ContinuousRoombaStateSpace(), room=Room(sspace, configuration=config),
                    aspace::AS=RoombaActions(), _amap=gen_amap(aspace)) where {SS, AS}
        RoombaMDP{SS, AS, eltype(SS)}(v_max, om_max, dt, contact_pen, time_pen, goal_reward, stairs_penalty, discount, config,
                                        sspace, room, aspace, _amap)
end

# state-space definitions
struct ContinuousRoombaStateSpace end
Base.eltype(sspace::Type{ContinuousRoombaStateSpace}) = RoombaState

"""
Specify a DiscreteRoombaStateSpace
- `x_step::Float64` distance between discretized points in x
- `y_step::Float64` distance between discretized points in y
- `th_step::Float64` distance between discretized points in theta
- `XLIMS::Vector` boundaries of room (x-dimension)
- `YLIMS::Vector` boundaries of room (y-dimension)

"""
struct DiscreteRoombaStateSpace
    x_step::Float64
    y_step::Float64
    th_step::Float64
    XLIMS::SVector{2, Float64}
    YLIMS::SVector{2, Float64}
    indices::SVector{2, Int}
    states_num::Int
end

# function to construct DiscreteRoombaStateSpace:
# `num_x_pts::Int` number of points to discretize x range to
# `num_y_pts::Int` number of points to discretize y range to
# `num_th_pts::Int` number of points to discretize th range to
function DiscreteRoombaStateSpace(num_x_pts::Int, num_y_pts::Int, num_theta_pts::Int)

    # hardcoded room-limits
    # watch for consistency with env_room
    XLIMS = SVec2(-25.0, 15.0)
    YLIMS = SVec2(-20.0, 5.0)

    x_step = (XLIMS[2]-XLIMS[1])/(num_x_pts-1)
    y_step = (YLIMS[2]-YLIMS[1])/(num_y_pts-1)

    if x_step != y_step
        @show x_step, y_step
        throw(AssertionError("x_step must equal y_step. In other word, (num_x_pts-1) should equal to 8/5(num_y_pts-1)."))
    end

    # project ROBOT_W.val/2 to nearest multiple of discrete_step
    ROBOT_W.val = 2 * max(1, round(DEFAULT_ROBOT_W/2 / x_step)) * x_step

    return DiscreteRoombaStateSpace(x_step,
                                    y_step,
                                    2*pi/(num_theta_pts-1),
                                    XLIMS,YLIMS,
                                    SVector{2, Int}(num_x_pts, num_x_pts * num_y_pts),
                                    num_x_pts*num_y_pts*num_theta_pts+2)
end

Base.eltype(sspace::Type{DiscreteRoombaStateSpace}) = Int

# round corners to discretized coordinates if necessary
round_corners(sspace::ContinuousRoombaStateSpace, corners) = corners
function round_corners(sspace::DiscreteRoombaStateSpace, corners)
    for i in 1:4
        xi = floor(Int, (corners[i,1] - sspace.XLIMS[1]) / sspace.x_step + 0.5)
        yi = floor(Int, (corners[i,2] - sspace.YLIMS[1]) / sspace.y_step + 0.5)
        corners[i,1] = sspace.XLIMS[1] + xi * sspace.x_step
        corners[i,2] = sspace.YLIMS[1] + yi * sspace.y_step
    end
    return corners
end

"""
Define the Roomba POMDP

Fields:
- `sensor::T` struct specifying the sensor used (Lidar or Bump)
- `mdp::T` underlying RoombaMDP
"""
struct RoombaPOMDP{SS, AS, S, T, O} <: POMDP{S, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP{SS, AS, S}
end

sensor(m::RoombaPOMDP) = m.sensor

# observation models
struct Bumper end
struct FrontBumper 
    max_theta::Float64 # wall must be within [-max_theta, max_theta] of heading
end
FrontBumper() = FrontBumper(π/4)
const AbstractBumper = Union{Bumper,FrontBumper}
POMDPs.obstype(::Type{AbstractBumper}) = Bool
POMDPs.obstype(::AbstractBumper) = Bool

struct Lidar 
    ray_stdev::Float64 # measurement noise: see POMDPs.observation definition
                       # below for usage
end
Lidar() = Lidar(0.1)

POMDPs.obstype(::Type{Lidar}) = Float64
POMDPs.obstype(::Lidar) = Float64 #float64(x)

struct DiscreteLidar
    ray_stdev::Float64
    disc_points::Vector{Float64} # cutpoints: endpoints of (0, Inf) assumed
    _d_disc::Vector{Float64}
end

POMDPs.obstype(::Type{DiscreteLidar}) = Int
POMDPs.obstype(::DiscreteLidar) = Int
DiscreteLidar(disc_points) = DiscreteLidar(Lidar().ray_stdev, disc_points, Vector{Float64}(length(disc_points)+1))

# Shorthands
const RoombaModel{SS, AS} = Union{RoombaMDP{SS, AS}, RoombaPOMDP{SS, AS}}
const BumperPOMDP{SS, AS, S} = RoombaPOMDP{SS, AS, S, Bumper, Bool}
const FrontBumperPOMDP{SS, AS, S} = RoombaPOMDP{SS, AS, S, FrontBumper, Bool}
const AbstractBumperPOMDP{SS, AS, S} = RoombaPOMDP{SS, AS, S, AbstractBumper, Bool}
const LidarPOMDP{SS, AS, S} = RoombaPOMDP{SS, AS, S, Lidar, Float64}
const DiscreteLidarPOMDP{SS, AS, S} = RoombaPOMDP{SS, AS, S, DiscreteLidar, Int}

# access the mdp of a RoombaModel
mdp(e::RoombaMDP) = e
mdp(e::RoombaPOMDP) = e.mdp

# access the room of a RoombaModel
room(m::RoombaMDP) = m.room
room(m::RoombaPOMDP) = room(m.mdp)

# access the state space of a RoombaModel
sspace(m::RoombaMDP{SS}) where SS = m.sspace
sspace(m::RoombaPOMDP{SS}) where SS = sspace(m.mdp)

# RoombaPOMDP Constructor
function RoombaPOMDP(sensor, mdp::RoombaMDP{SS, AS, S}) where {SS, AS, S}
    RoombaPOMDP{SS, AS, S, typeof(sensor), obstype(sensor)}(sensor, mdp)
end

RoombaPOMDP(;sensor=Bumper(), mdp=RoombaMDP()) = RoombaPOMDP(sensor,mdp)

# function to determine if there is contact with a wall
wall_contact(e::RoombaModel, state) = wall_contact(mdp(e).room, SVec2(state[1], state[2]))

POMDPs.actions(m::RoombaModel) = mdp(m).aspace
n_actions(m::RoombaModel) = length(mdp(m).aspace)

# maps a RoombaAct to an index in a RoombaModel with discrete actions
function POMDPs.actionindex(m::RoombaModel{SS, AS}, a::RoombaAct) where {SS, AS <: RoombaActions}
    error("Action index not defined for continuous actions.")
end

POMDPs.actionindex(m::RoombaModel, a::RoombaAct) = mdp(m)._amap[a]

# function to get goal xy location for heuristic controllers
function get_goal_xy(m::RoombaModel)
    grn = mdp(m).room.goal_rect
    gwn = mdp(m).room.goal_wall
    gr = mdp(m).room.rectangles[grn]
    corners = gr.corners
    if gwn == 4
        return (corners[1,:] + corners[4,:]) / 2.
    else
        return (corners[gwn,:] + corners[gwn+1,:]) / 2.
    end
end

# transition Roomba state given curent state and action
POMDPs.transition(m::RoombaPOMDP, s, a::RoombaAct) = transition(m.mdp, s, a)
POMDPs.transition(m::RoombaMDP, s::RoombaState, a::RoombaAct) = Deterministic(get_next_state(m, s, a))
POMDPs.transition(m::RoombaMDP, s::Int, a::RoombaAct) = Deterministic(convert_s(Int, get_next_state(m, convert_s(RoombaState, s, m), a), m))

function get_next_state(m::RoombaMDP, s::RoombaState, a::RoombaAct)
    v, om = a
    v = clamp(v, 0.0, m.v_max)
    om = clamp(om, -m.om_max, m.om_max)

    # propagate dynamics without wall considerations
    x, y, th, _ = s
    dt = m.dt

    # dynamics assume robot rotates and then translates
    next_th = wrap_to_pi(th + om*dt)

    # make sure we arent going through a wall
    p0 = SVec2(x, y)
    heading = SVec2(cos(next_th), sin(next_th))
    des_step = v*dt
    pos = legal_translate(m.room, p0, heading, des_step)

    # Determine whether goal state or stairs have been reached
    r = room(m)
    grn = r.goal_rect
    gwn = r.goal_wall
    srn = r.stair_rect
    swn = r.stair_wall
    gr = r.rectangles[grn]
    sr = r.rectangles[srn]
    next_status = 1.0 * contact_wall(gr, gwn, pos) - 1.0 * contact_wall(sr, swn, pos)

    # define next state
    return RoombaState(pos[1], pos[2], next_th, next_status)
end

# enumerate all possible states in a DiscreteRoombaStateSpace
POMDPs.states(m::RoombaModel{SS}) where SS <: DiscreteRoombaStateSpace = vec(1:n_states(m))
POMDPs.states(m::RoombaModel{SS}) where SS <: ContinuousRoombaStateSpace = sspace(m)

# return the number of states in a DiscreteRoombaStateSpace
n_states(m::RoombaModel{SS}) where SS <: DiscreteRoombaStateSpace = sspace(m).states_num

function n_states(m::RoombaModel{SS}) where SS <: ContinuousRoombaStateSpace
    error("State-space must be DiscreteRoombaStateSpace.")
end

# map a RoombaState to an index in a DiscreteRoombaStateSpace
POMDPs.stateindex(m::RoombaModel{SS}, si::Int) where SS <: DiscreteRoombaStateSpace = si
function POMDPs.stateindex(m::RoombaModel{SS}, s::RoombaState) where SS <: ContinuousRoombaStateSpace
    error("State-space must be DiscreteRoombaStateSpace.")
end

# map an index in a DiscreteRoombaStateSpace to the corresponding RoombaState
function POMDPs.convert_s(::Type{Int}, s::RoombaState, m::RoombaModel{SS}) where SS <: DiscreteRoombaStateSpace
    ss = sspace(m)
    if s.status > 0.0
        return ss.states_num
    elseif s.status < 0.0
        return ss.states_num - 1
    else
        xind = floor(Int, (s[1] - ss.XLIMS[1]) / ss.x_step + 0.5)
        yind = floor(Int, (s[2] - ss.YLIMS[1]) / ss.y_step + 0.5)
        thind = floor(Int, (s[3] - (-pi)) / ss.th_step + 0.5)
        return 1 + xind + ss.indices[1] * yind + ss.indices[2] * thind
    end
end

# map an index in a DiscreteRoombaStateSpace to the corresponding RoombaState
function POMDPs.convert_s(::Type{RoombaState}, si::Int, m::RoombaModel{SS}) where SS <: DiscreteRoombaStateSpace
    ss = sspace(m)
    if si == ss.states_num
        return RoombaState(0.0, 0.0, 0.0, 1.0)
    elseif si == ss.states_num - 1
        return RoombaState(0.0, 0.0, 0.0, -1.0)
    else
        si -= 1
        thi, si = divrem(si, ss.indices[2])
        yi, xi = divrem(si, ss.indices[1])
        x = ss.XLIMS[1] + xi * ss.x_step
        y = ss.YLIMS[1] + yi * ss.y_step
        th = -pi + thi * ss.th_step
        return RoombaState(x, y, th, 0.0)
    end
end

# defines reward function R(s,a,s')
function POMDPs.reward(m::RoombaModel,
                s::RoombaState, 
                a::RoombaAct,
                sp::RoombaState)
    
    # penalty for each timestep elapsed
    cum_reward = mdp(m).time_pen

    # penalty for bumping into wall (not incurred for consecutive contacts)
    previous_wall_contact = wall_contact(m,s)
    current_wall_contact = wall_contact(m,sp)
    if(!previous_wall_contact && current_wall_contact)
        cum_reward += mdp(m).contact_pen
    end

    # terminal rewards
    cum_reward += mdp(m).goal_reward*(sp.status == 1.0)
    cum_reward += mdp(m).stairs_penalty*(sp.status == -1.0)

    return cum_reward  
end

POMDPs.reward(m::RoombaModel, si::Int, a::RoombaAct, spi::Int) = reward(m, convert_s(RoombaState, si, m), a, convert_s(RoombaState, spi, m))

# determine if a terminal state has been reached
POMDPs.isterminal(m::RoombaModel, s::RoombaState) = abs(s.status) > 0.0
POMDPs.isterminal(m::RoombaModel, s::Int) = n_states(m) - s < 2

# simultaneously determine if a roomba is bumping a wall and facing it
function wall_contact_facing(r::Room, b::FrontBumper, state::RoombaState)
    for (i, rect) in enumerate(r.rectangles)
        pos = SVec2(state.x, state.y) # state to SVec2 pos
        wc, _ = wall_contact(rect, pos)
        if wc >= 0
            wall_th = pi - pi/2 * (wc - 1) # wc = {1,2,3,4}, state.th, b.max_theta
            is_facing = wrap_to_pi(abs(wall_th - state.theta)) <= b.max_theta
            return true, is_facing
        end
    end
    return false, false
end

# Bumper POMDP observation
POMDPs.observation(m::BumperPOMDP, sp::RoombaState) = Deterministic(wall_contact(m, sp)) # in {false,true}
function POMDPs.observation(m::FrontBumperPOMDP, sp::RoombaState) 
    # function to determin if facing a wall
    contact, facing = wall_contact_facing(mdp(m).room, sensor(m), sp)
    return Deterministic(contact && facing)
end
POMDPs.observation(m::AbstractBumperPOMDP, sp::Int) = observation(m, convert_s(RoombaState, sp, m))

n_observations(m::AbstractBumperPOMDP) = 2
POMDPs.observations(m::AbstractBumperPOMDP) = [false, true]
POMDPs.obsindex(m::AbstractBumperPOMDP, o::Bool) = o + 1

# Lidar POMDP observation
function lidar_obs_distribution(m::RoombaMDP, ray_stdev::Float64, sp::RoombaState)
    x, y, th = sp
    # determine uncorrupted observation
    rl = ray_length(m.room, SVec2(x, y), SVec2(cos(th), sin(th)))
    # compute observation noise
    sigma = ray_stdev * max(rl, 0.01)
    # disallow negative measurements
    return Truncated(Normal(rl, sigma), 0.0, Inf)
end

POMDPs.observation(m::LidarPOMDP, sp::RoombaState) = lidar_obs_distribution(mdp(m), sensor(m).ray_stdev, sp)
POMDPs.observation(m::LidarPOMDP, sp::Int) = observation(m, convert_s(RoombaState, sp, m))

function n_observations(m::LidarPOMDP)
    error("n_observations not defined for continuous observations.")
end

function POMDPs.observations(m::LidarPOMDP)
    error("LidarPOMDP has continuous observations. Use DiscreteLidarPOMDP for discrete observation spaces.")
end

# DiscreteLidar POMDP observation
function POMDPs.observation(m::DiscreteLidarPOMDP, sp::RoombaState)
    s = sensor(m)
    d = lidar_obs_distribution(mdp(m), s.ray_stdev, sp)

    # discretize observations
    interval_start = 0.0
    d_disc = s._d_disc
    for i in 1:length(s.disc_points)
        interval_end = cdf(d, s.disc_points[i])
        d_disc[i] = interval_end - interval_start
        interval_start = interval_end
    end
    d_disc[end] = 1.0 - interval_start

    return SparseCat(1:length(d_disc), d_disc)
end

POMDPs.observation(m::DiscreteLidarPOMDP, sp::Int) = observation(m, convert_s(RoombaState, sp, m))

n_observations(m::DiscreteLidarPOMDP) = length(sensor(m).disc_points) + 1
POMDPs.observations(m::DiscreteLidarPOMDP) = vec(1:n_observations(m))
POMDPs.obsindex(m::DiscreteLidarPOMDP, o::Int) = o
                        
# define discount factor
POMDPs.discount(m::RoombaModel) = mdp(m).discount

# struct to define an initial distribution over Roomba states
struct RoombaInitialDistribution{M<:RoombaModel}
    m::M
end

# definition of initialstate for Roomba environment
POMDPs.initialstate(m::RoombaModel{SS}) where SS <: ContinuousRoombaStateSpace = RoombaInitialDistribution(m)

function POMDPs.initialstate(m::RoombaModel{SS}) where SS <: DiscreteRoombaStateSpace
    ss = sspace(m)
    r = room(m)
    x_states = range(ss.XLIMS[1], stop=ss.XLIMS[2], step=ss.x_step)
    y_states = range(ss.YLIMS[1], stop=ss.YLIMS[2], step=ss.y_step)
    th_states = range(-pi, stop=pi, step=ss.th_step)
    sup = vec(collect(RoombaState(x,y,th,0.0) for x in x_states, y in y_states if in_room(r, SVec2(x,y)) for th in th_states))
    SparseCat([convert_s(Int, s, m) for s in sup], fill(1.0/length(sup), length(sup)))
end

function Base.rand(rng::AbstractRNG, d::RoombaInitialDistribution)
    x, y = init_pos(mdp(d.m).room, rng)
    th = rand(rng) * 2*pi - pi
    return RoombaState(x, y, th, 0.0)
end

# Render a room and show robot
function render(ctx::CairoContext, m::RoombaModel, step)
    env = mdp(m)
    state = step[:sp]

    radius = ROBOT_W.val*6

    # render particle filter belief
    if haskey(step, :bp)
        bp = step[:bp]
        if bp isa AbstractParticleBelief
            for p in particles(bp)
                x, y = transform_coords(SVec2(p[1],p[2]))
                arc(ctx, x, y, radius, 0, 2*pi)
                set_source_rgba(ctx, 0.6, 0.6, 1, 0.3)
                fill(ctx)
            end
        end
    end

    # Render room
    render(env.room, ctx)

    # Find center of robot in frame and draw circle
    x, y = transform_coords(SVec2(state[1],state[2]))
    arc(ctx, x, y, radius, 0, 2*pi)
    set_source_rgb(ctx, 1, 0.6, 0.6)
    fill(ctx)

    # Draw line indicating orientation
    move_to(ctx, x, y)
    end_point = SVec2(state[1] + ROBOT_W.val*cos(state[3])/2, state[2] + ROBOT_W.val*sin(state[3])/2)
    end_x, end_y = transform_coords(end_point)
    line_to(ctx, end_x, end_y)
    set_source_rgb(ctx, 0, 0, 0)
    stroke(ctx)
    return ctx
end

# this object should have show methods for a variety of mimes
# in particular, for now it has both png and html-like
# it would also give us a ton of hacker cred to make an ascii rendering
struct RoombaVis
    m::RoombaModel
    step::Any
    text::String
end

render(m::RoombaModel, step; text::String="") = RoombaVis(m, step, text)

function Base.show(io::IO, mime::Union{MIME"text/html", MIME"image/svg+xml"}, v::RoombaVis)
    c = CairoSVGSurface(io, 800, 600)
    ctx = CairoContext(c)
    render(ctx, v.m, v.step)
    finish(c)
end

function Base.show(io::IO, mime::MIME"image/png", v::RoombaVis)
    c = CairoRGBSurface(800, 600)
    ctx = CairoContext(c)
    render(ctx, v.m, v.step)
    # finish(c) # doesn't work with this; I wonder why
    write_to_png(c, io)
end