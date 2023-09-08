# specification of particle filters for the bumper and lidar Roomba environments
# maintained by {jmorton2,kmenda}@stanford.edu

"""
Definition of the particle filter for the Roomba environment
Fields:
- `v_noise_coeff::Float64` coefficient to scale particle-propagation noise in velocity
- `om_noise_coeff::Float64`coefficient to scale particle-propagation noise in turn-rate
"""
mutable struct RoombaParticleFilter{M<:RoombaModel,RM,RNG<:AbstractRNG,PMEM} <: Updater
    model::M
    resampler::RM
    n_init::Int
    v_noise_coeff::Float64
    om_noise_coeff::Float64
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
end

function RoombaParticleFilter(model, n::Integer, v_noise_coeff, om_noise_coeff, resampler=LowVarianceResampler(n), rng::AbstractRNG=Random.GLOBAL_RNG)
    return RoombaParticleFilter(model,
                               resampler,
                               n,
                               v_noise_coeff,
                               om_noise_coeff,
                               rng,
                               sizehint!(particle_memory(model), n),
                               sizehint!(Float64[], n)
                              )
end

# Modified Update function adds noise to the actions that propagate particles
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    empty!(pm)
    empty!(wm)
    all_terminal = true
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            # noise added here:
            a_pert = a + SVec2(up.v_noise_coeff * (rand(up.rng) - 0.5), up.om_noise_coeff * (rand(up.rng) - 0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        end
    end
    # if all particles are terminal, issue an error
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end
    resampler = up.resampler
    return ParticleFilters.resample(resampler,
                    WeightedParticleBelief(pm, wm, sum(wm), nothing),
                    up.model,
                    up.model,
                    b, a, o,
                    up.rng)
end

# initialize belief state
ParticleFilters.initialize_belief(up::RoombaParticleFilter, d) = ParticleCollection([rand(up.rng, d) for i in 1:up.n_init])

### SearchParticleFilter
# To handle proper costs for terminal particles

mutable struct RoombaSearchParticleFilter{M<:RoombaModel,RNG<:AbstractRNG,PMEM} <: Updater
    model::M
    n_init::Int
    v_noise_coeff::Float64
    om_noise_coeff::Float64
    rng::RNG
    _particle_memory::PMEM
    _weight_memory::Vector{Float64}
    _dead_particles::PMEM
end

function RoombaSearchParticleFilter(model, n::Integer, v_noise_coeff, om_noise_coeff, rng::AbstractRNG=Random.GLOBAL_RNG)
    return RoombaSearchParticleFilter(model,
                               # resampler, #hand make as LowVarianceResampler(n)
                               n,
                               v_noise_coeff,
                               om_noise_coeff,
                               rng,
                               sizehint!(particle_memory(model), n),
                               sizehint!(Float64[], n),
                               sizehint!(particle_memory(model), n),
                              )
end

# Modified Update function adds noise to the actions that propagate particles
function POMDPs.update(up::RoombaSearchParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    dp = up._dead_particles
    empty!(pm)
    empty!(wm)
    empty!(dp)
    all_terminal = true
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            # noise added here:
            a_pert = a + SVec2(up.v_noise_coeff * (rand(up.rng) - 0.5), up.om_noise_coeff * (rand(up.rng) - 0.5))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            push!(pm, sp)
            push!(wm, obs_weight(up.model, s, a_pert, sp, o))
        else
            push!(dp, s)
        end
    end
    # if all particles are terminal, issue an error
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    # return a particle collection that includes all dead particles and n-length(dead_particles) resampled ones
    target_particles = up.n_init - length(dp)
    living = resample(WeightedParticleBelief(pm, wm, sum(wm), nothing), target_particles, up.rng)
    return ParticleCollection([living; dp])
end

# low variance resampler = page 110 of Probabilistic Robotics by Thrun Burgard and Fox. O(n) runtime
function resample(b::AbstractParticleBelief{S}, n::Int, rng::AbstractRNG) where {S}
    ps = Array{S}(undef, n)
    r = rand(rng)*weight_sum(b)/n # (0,1/n*weightedsum)
    c = weight(b, 1)
    i = 1
    U = r
    for m in 1:n
        while U > c && i < n_particles(b)
            i += 1
            c += weight(b, i)
        end
        U += weight_sum(b)/n
        ps[m] = particles(b)[i]
    end
    return ps
end

# initialize belief state
ParticleFilters.initialize_belief(up::RoombaSearchParticleFilter, d) = ParticleCollection([rand(up.rng, d) for i in 1:up.n_init])