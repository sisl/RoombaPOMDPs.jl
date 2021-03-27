# import necessary packages
using AA228FinalProject
using POMDPs
using POMDPPolicies
using ParticleFilters
using POMDPSimulators
using Cairo
using Gtk
using Random
using Test

sensor = Lidar() # or Bumper() for the bumper version of the environment
config = 3 # 1,2, or 3
m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config));

num_particles = 2000
v_noise_coefficient = 2.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient)

# Define the policy to test
mutable struct ToEnd <: Policy
    ts::Int64 # to track the current time-step.
end

# extract goal for heuristic controller
goal_xy = get_goal_xy(m)

# define a new function that takes in the policy struct and current belief,
# and returns the desired action
function POMDPs.action(p::ToEnd, b::ParticleCollection{RoombaState})
    
    # spin around to localize for the first 25 time-steps
    if p.ts < 25
        p.ts += 1
        return RoombaAct(0.,1.0) # all actions are of type RoombaAct(v,om)
    end
    p.ts += 1

    # after 25 time-steps, we follow a proportional controller to navigate
    # directly to the goal, using the mean belief state
    
    # compute mean belief of a subset of particles
    s = mean(b)
    
    # compute the difference between our current heading and one that would
    # point to the goal
    goal_x, goal_y = goal_xy
    x,y,th = s[1:3]
    ang_to_goal = atan(goal_y - y, goal_x - x)
    del_angle = wrap_to_pi(ang_to_goal - th)
    
    # apply proportional control to compute the turn-rate
    Kprop = 1.0
    om = Kprop * del_angle
    
    # always travel at some fixed velocity
    v = 5.0
    
    return RoombaAct(v, om)
end

# run simulation

Random.seed!(0)

p = ToEnd(0)

for step in stepthrough(m,p,belief_updater, max_steps=100)
    @show step.a
end

step = first(stepthrough(m,p,belief_updater, max_steps=100))

@show fbase = tempname()

v = render(m, step)
for (ext, mime) in ["html"=>MIME("text/html"), "svg"=>MIME("image/svg+xml"), "png"=>MIME("image/png")]
    fname = fbase*"."*ext
    open(fname, "w") do f
        show(f, mime, v)
    end
    @test filesize(fname) > 0
end

m = RoombaPOMDP(sensor=sensor, mdp=RoombaMDP(config=config, aspace=vec([RoombaAct(v, om) for v in range(0, stop=2, length=2) for om in range(-2, stop=2, length=3)]), sspace=DiscreteRoombaStateSpace(41, 26, 20)));

@testset "DiscreteRoomba Test" begin
    d = initialstate(m)
    @test sum(pdf(d, s) for s in support(d)) ≈ 1.0
    @test sum(pdf(d, s) for s in states(m)) ≈ 1.0
    s = rand(states(m))
    for a in actions(m)
        T = transition(m, s, a)
        p = 0.0
        for sp in states(m)
            p += pdf(T, sp)
        end
        @test p ≈ 1.0
    end

    for s in states(m)
        @test s == convert_s(Int, convert_s(RoombaState, s, m), m)
    end
end

belief_updater = RoombaParticleFilter(m, num_particles, v_noise_coefficient, om_noise_coefficient)

for step in stepthrough(m,RandomPolicy(m),belief_updater, max_steps=100)
    @show convert_s(RoombaState, step.s, m), step.a, step.o, step.r
end