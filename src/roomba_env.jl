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
- `status::Bool` indicator whether robot has reached goal state or stairs
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
- `aspace::AS` environment action-space struct
"""
@with_kw mutable struct RoombaMDP{AS} <: MDP{RoombaState, RoombaAct}
    v_max::Float64  = 10.0  # m/s
    om_max::Float64 = 1.0   # rad/s
    dt::Float64     = 0.5   # s
    contact_pen::Float64 = -1.0 
    time_pen::Float64 = -0.1
    goal_reward::Float64 = 10
    stairs_penalty::Float64 = -10
    config::Int = 1
    room::Room  = Room(configuration=config)
    aspace::AS = RoombaActions()
end

"""
Define the Roomba POMDP

Fields:
- `sensor::T` struct specifying the sensor used (Lidar or Bump)
- `mdp::T` underlying RoombaMDP
"""
struct RoombaPOMDP{T, O} <: POMDP{RoombaState, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP
end

# observation models
struct Bumper end
POMDPs.obstype(::Type{Bumper}) = Bool
POMDPs.obstype(::Bumper) = Bool

struct Lidar 
    ray_stdev::Float64 # measurement noise: see POMDPs.observation definition
                       # below for usage
end
Lidar() = Lidar(0.1)

POMDPs.obstype(::Type{Lidar}) = Float64
POMDPs.obstype(::Lidar) = Float64 #float64(x)


# Shorthands
const RoombaModel = Union{RoombaMDP, RoombaPOMDP}
const BumperPOMDP = RoombaPOMDP{Bumper, Bool}
const LidarPOMDP = RoombaPOMDP{Lidar, Float64}

# access the mdp of a RoombaModel
mdp(e::RoombaMDP) = e
mdp(e::RoombaPOMDP) = e.mdp


# RoombaPOMDP Constructor
function RoombaPOMDP(sensor, mdp)
    RoombaPOMDP{typeof(sensor), obstype(sensor)}(sensor, mdp)
end

RoombaPOMDP(;sensor=Bumper(), mdp=RoombaMDP()) = RoombaPOMDP(sensor,mdp)

# function to determine if there is contact with a wall
wall_contact(e::RoombaModel, state) = wall_contact(mdp(e).room, state[1:2])

POMDPs.actions(m::RoombaModel) = mdp(m).aspace
POMDPs.n_actions(m::RoombaModel) = length(mdp(m).aspace)

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

# initializes x,y,th of Roomba in the room
function POMDPs.initialstate(m::RoombaModel, rng::AbstractRNG)
    e = mdp(m)
    x, y = init_pos(e.room, rng)
    th = rand() * 2*pi - pi
    return RoombaState(x, y, th, 0.0)
end

# transition Roomba state given curent state and action
function POMDPs.transition(env::RoombaModel,
                           s::AbstractVector{Float64},
                           a::AbstractVector{Float64})

    e = mdp(env)
    v, om = a
    v = clamp(v, 0.0, e.v_max)
    om = clamp(om, -e.om_max, e.om_max)

    # propagate dynamics without wall considerations
    x, y, th, _ = s
    dt = e.dt

    # dynamics assume robot rotates and then translates
    next_th = wrap_to_pi(th + om*dt)

    # make sure we arent going through a wall
    p0 = SVector(x, y)
    heading = SVector(cos(next_th), sin(next_th))
    des_step = v*dt
    next_x, next_y = legal_translate(e.room, p0, heading, des_step)

    # Determine whether goal state or stairs have been reached
    grn = mdp(env).room.goal_rect
    gwn = mdp(env).room.goal_wall
    srn = mdp(env).room.stair_rect
    swn = mdp(env).room.stair_wall
    gr = mdp(env).room.rectangles[grn]
    sr = mdp(env).room.rectangles[srn]
    next_status = 1.0*contact_wall(gr, gwn, [next_x, next_y]) - 1.0*contact_wall(sr, swn, [next_x, next_y])

    # define next state
    sp = RoombaState(next_x, next_y, next_th, next_status)

    return Deterministic(sp)
end

# defines reward function R(s,a,s')
function POMDPs.reward(m::RoombaModel,
                s::AbstractVector{Float64}, 
                a::AbstractVector{Float64},
                sp::AbstractVector{Float64})
    
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

# determine if a terminal state has been reached
POMDPs.isterminal(m::RoombaModel, s::AbstractVector{Float64}) = abs(s.status) > 0.0

# Bumper POMDP observation
function POMDPs.observation(m::BumperPOMDP, 
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64})
    return Deterministic(wall_contact(m, sp)) # in {0.0,1.0}
end

# Lidar POMDP observation
function POMDPs.observation(m::LidarPOMDP, 
                            a::AbstractVector{Float64},
                            sp::AbstractVector{Float64})
    x, y, th = sp

    # determine uncorrupted observation
    rl = ray_length(mdp(m).room, [x, y], [cos(th), sin(th)])

    # compute observation noise
    sigma = m.sensor.ray_stdev * max(rl, 0.01)

    # disallow negative measurements
    return Truncated(Normal(rl, sigma), 0.0, Inf)
end
                        
# define discount factor
POMDPs.discount(m::RoombaModel) = 1.0

# struct to define an initial distribution over Roomba states
struct RoombaInitialDistribution{M<:RoombaModel}
    m::M
end

# definition of initialstate and initialstate_distribution for Roomba environment
POMDPs.rand(rng::AbstractRNG, d::RoombaInitialDistribution) = initialstate(d.m, rng)
POMDPs.initialstate_distribution(m::RoombaModel) = RoombaInitialDistribution(m)

# Render a room and show robot
function render(ctx::CairoContext, m::RoombaModel, step)
    env = mdp(m)
    state = step[:sp]

    radius = ROBOT_W*6

    # render particle filter belief
    if haskey(step, :bp)
        bp = step[:bp]
        if bp isa AbstractParticleBelief
            for p in particles(bp)
                x, y = transform_coords(p[1:2])
                arc(ctx, x, y, radius, 0, 2*pi)
                set_source_rgba(ctx, 0.6, 0.6, 1, 0.3)
                fill(ctx)
            end
        end
    end

    # Render room
    render(env.room, ctx)

    # Find center of robot in frame and draw circle
    x, y = transform_coords(state[1:2])
    arc(ctx, x, y, radius, 0, 2*pi)
    set_source_rgb(ctx, 1, 0.6, 0.6)
    fill(ctx)

    # Draw line indicating orientation
    move_to(ctx, x, y)
    end_point = [state[1] + ROBOT_W*cos(state[3])/2, state[2] + ROBOT_W*sin(state[3])/2]
    end_x, end_y = transform_coords(end_point)
    line_to(ctx, end_x, end_y)
    set_source_rgb(ctx, 0, 0, 0)
    stroke(ctx)
    return ctx
end

function render(m::RoombaModel, step)
    io = IOBuffer()
    c = CairoSVGSurface(io, 800, 600)
    ctx = CairoContext(c)
    render(ctx, m, step)
    finish(c)
    return HTML(String(take!(io)))
end
