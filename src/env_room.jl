# Code to define the environment room and rectangles used to define it
# maintained by {jmorton2,kmenda}@stanford.edu

# Define constants  -- all units in m
RW = 5. # room width
mutable struct ROBOT_W_struct
    val::Float64 # robot width
end
DEFAULT_ROBOT_W = 1.0
ROBOT_W = ROBOT_W_struct(DEFAULT_ROBOT_W)
MARGIN = 1e-12

# Define rectangle type for constructing hallway
# corners: 4x2 np array specifying
#		   bottom-left, top-left,
#		   top-right, bottom-right corner
# walls: length 4 list of bools specifying
#		 if left, top, right, bottom sides are
#		 open (False) or walls (True)
mutable struct Rectangle
    corners::Array{Float64, 2}
    walls::Array{Bool, 1}
    segments::Array{LineSegment, 1}
    width::Float64
    height::Float64
    midpoint::Array{Float64, 1}
    area::Float64
    xl::Float64
    xu::Float64
    yl::Float64
    yu::Float64

    function Rectangle(
        corners::Array{Float64, 2},
        walls::Array{Bool, 1};
        goal_idx::Int=0,
        stair_idx::Int=0
        )

        retval = new()

        retval.corners = corners
        retval.walls = walls

        retval.width = corners[3, 1] - corners[2, 1]
        retval.height = corners[2, 2] - corners[1, 2]
        mean_vals = mean(corners, dims=1)
        retval.midpoint = SVector(mean_vals[1, 1], mean_vals[1, 2])
        
        # compute area in which robot could be initialized
        retval.xl = corners[2, 1]
        retval.xu = corners[3, 1]
        retval.yl = corners[1, 2]
        retval.yu = corners[2, 2]
        if walls[1]
            retval.width -= ROBOT_W.val/2
            retval.xl += ROBOT_W.val/2
        end
        if walls[2]
            retval.height -= ROBOT_W.val/2
            retval.yu -= ROBOT_W.val/2
        end
        if walls[3]
            retval.width -= ROBOT_W.val/2
            retval.xu -= ROBOT_W.val/2
        end
        if walls[4]
            retval.height -= ROBOT_W.val/2
            retval.yl += ROBOT_W.val/2
        end
        @assert retval.width > 0.0 && retval.height > 0.0 "Negative width or height"
        retval.area = retval.width * retval.height
        
        retval.segments = [LineSegment(corners[i, :], corners[i+1, :], (goal_idx == i), (stair_idx == i)) for i =1:3 if walls[i]]
        if walls[4]
            push!(retval.segments, LineSegment(corners[1, :], corners[4, :], (goal_idx == 4), (stair_idx == 4)))
        end

        retval
    end
end

# Randomly initializes the robot in this rectangle
function init_pos(rect::Rectangle, rng)
    w = rect.xu - rect.xl
    h = rect.yu - rect.yl
    init_pos = SVector(rand(rng)*w + rect.xl, rand(rng)*h + rect.yl)
    
    init_pos
end

# Determines if pos (center of robot) is within the rectangle
function in_rectangle(rect::Rectangle, pos::AbstractVector{Float64})
    corners = rect.corners
    xlims = SVector(rect.xl - MARGIN, rect.xu + MARGIN)
    ylims = SVector(rect.yl - MARGIN, rect.yu + MARGIN)
    if xlims[1] < pos[1] < xlims[2]
        if ylims[1] < pos[2] < ylims[2]
            return true
        end
    end
    return false
end

# determines if pos (center of robot) is intersecting with a wall
# returns: -2, -Inf if center of robot not in room
#          -1, -Inf if not in wall contact
#          0~3, violation mag, indicating which wall has contact
#          if multiple, returns largest violation
function wall_contact(rect::Rectangle, pos::AbstractVector{Float64})
    if !(in_rectangle(rect, pos))
        return -2, -Inf
    end
    corners = rect.corners
    xlims = SVector(corners[2, 1], corners[3, 1])
    ylims = SVector(corners[1, 2], corners[2, 2])

    contacts = []
    contact_mags = []
    if pos[1] - ROBOT_W.val/2 <= xlims[1] + MARGIN && rect.walls[1]
        # in contact with left wall
        push!(contacts, 1)
        push!(contact_mags, abs(pos[1] - ROBOT_W.val/2 - xlims[1]))
    end
    if pos[2] + ROBOT_W.val/2 + MARGIN >= ylims[2] && rect.walls[2]
        # in contact with top wall
        push!(contacts, 2)
        push!(contact_mags, abs(pos[2] + ROBOT_W.val/2 - ylims[2]))
    end
    if pos[1] + ROBOT_W.val/2 + MARGIN >= xlims[2] && rect.walls[3]
        # in contact with right wall
        push!(contacts, 3)
        push!(contact_mags, abs(pos[1] + ROBOT_W.val/2 - xlims[2]))
    end
    if pos[2] - ROBOT_W.val/2 <= ylims[1] + MARGIN && rect.walls[4]
        # in contact with bottom wall
        push!(contacts, 4)
        push!(contact_mags, abs(pos[2] - ROBOT_W.val/2 - ylims[1]))
    end

    if length(contacts) == 0
        return -1, -Inf
    else
        return contacts[argmax(contact_mags)], maximum(contact_mags)
    end
end

# Find closest distance to any wall
function furthest_step(rect::Rectangle, pos::AbstractVector{Float64}, heading::AbstractVector{Float64})
    return minimum(furthest_step(seg, pos, heading, ROBOT_W.val/2) for seg in rect.segments)
end

# computes the length of a ray from robot center to closest segment 
# from p0 pointing in direction heading
function ray_length(rect::Rectangle, pos::AbstractVector{Float64}, heading::AbstractVector{Float64})
    return minimum(ray_length(seg, pos, heading) for seg in rect.segments)
end

# Render rectangle based on segments
function render(rect::Rectangle, ctx::CairoContext)
    for seg in rect.segments
        render(seg, ctx)
    end
end

# round corners to discretized coordinates if necessary
function round_corners(sspace, corners)

    if sspace isa DiscreteRoombaStateSpace
        for i in 1:4
            xi = floor(Int, (corners[i,1] - sspace.XLIMS[1]) / sspace.x_step + 0.5) + 1
            yi = floor(Int, (corners[i,2] - sspace.YLIMS[1]) / sspace.y_step + 0.5) + 1
            corners[i,1] = sspace.XLIMS[1] + (xi-1) * sspace.x_step
            corners[i,2] = sspace.YLIMS[1] + (yi-1) * sspace.y_step
        end
    end
    return corners



end

# generate consecutive rectangles that make up the room
# all rectangles share a full "wall" with an adjacent rectangle
# shared walls are not solid - just used to specify geometry
mutable struct Room
    rectangles::Array{Rectangle, 1}
    areas::Array{Float64, 1}
    goal_rect::Int  # Index of rectangle with goal state
    goal_wall::Int  # Index of wall that leads to goal
    stair_rect::Int # Index of rectangle with stairs
    stair_wall::Int # Index of wall that leads to stairs

    function Room(sspace; configuration=1)

        retval = new()

        # Define different configurations for stair and goal locations
        goal_idxs = [0, 0, 0, 0]
        stair_idxs = [0, 0, 0, 0]
        if configuration == 2
            retval.goal_rect = 1
            retval.goal_wall = 4
            retval.stair_rect = 2
            retval.stair_wall = 1
        elseif configuration == 3
            retval.goal_rect = 4
            retval.goal_wall = 3
            retval.stair_rect = 2
            retval.stair_wall = 1
        else
            retval.goal_rect = 4
            retval.goal_wall = 3
            retval.stair_rect = 4
            retval.stair_wall = 4
        end
        goal_idxs[retval.goal_rect] = retval.goal_wall
        stair_idxs[retval.stair_rect] = retval.stair_wall

        # Initialize array of rectangles
        rectangles = []

        # Rectangle 1
        corners = round_corners(sspace,[[-20-RW -20]; [-20-RW 0-RW]; [-20+RW 0-RW]; [-20+RW -20]])
        walls = [true, false, true, true] # top wall shared
        push!(rectangles, Rectangle(corners, walls, goal_idx=goal_idxs[1], stair_idx=stair_idxs[1]))

        # Rectangle 2
        corners = round_corners(sspace,[[-20-RW 0-RW]; [-20-RW 0+RW]; [-20+RW 0+RW]; [-20+RW 0-RW]])
        walls = [true, true, false, false] # bottom, right wall shared
        push!(rectangles, Rectangle(corners, walls, goal_idx=goal_idxs[2], stair_idx=stair_idxs[2]))

        # Rectangle 3
        corners = round_corners(sspace,[[-20+RW 0-RW]; [-20+RW 0+RW]; [10 0+RW]; [10 0-RW]])
        walls = [false, true, false, true] # left wall shared
        push!(rectangles, Rectangle(corners, walls, goal_idx=goal_idxs[3], stair_idx=stair_idxs[3]))

        # Rectangle 4
        corners = round_corners(sspace,[[10 0-RW]; [10 0+RW]; [10+RW 0+RW]; [10+RW 0-RW]])
        walls = [false, true, true, true] # left wall shared
        push!(rectangles, Rectangle(corners, walls, goal_idx=goal_idxs[4], stair_idx=stair_idxs[4]))

        retval.rectangles = rectangles
        retval.areas = [r.area for r in rectangles]
        
        retval
    end
end

# Sample from multinomial distribution
function multinomial_sample(p::AbstractVector{Float64})
    rand_num = rand()
    for i = 1:length(p)
        if rand_num < sum(p[1:i])
            return i
        end
    end
end

# Initialize the robot randomly in the room
# Randomly select a rectangle weighted by initializable area
function init_pos(r::Room, rng::AbstractRNG)
    norm_areas = r.areas/sum(r.areas)
    rect = multinomial_sample(norm_areas)
    return init_pos(r.rectangles[rect], rng)
end

# Determines if pos is in contact with a wall
# returns bool indicating contact
function wall_contact(r::Room, pos::AbstractVector{Float64})
    for (i, rect) in enumerate(r.rectangles)
        wc, _ = wall_contact(rect, pos)
        if wc >= 0
            return true
        end
    end
    return false
end

# Determines if pos is in contact with a specific wall
# returns true if true
function contact_wall(r::Rectangle, wall::Int, pos::Array{Float64, 1})
    wc,_ = wall_contact(r, pos)
    return wc == wall
end    

# Determines if pos (center of robot) is within the room
function in_room(r::Room, pos::AbstractVector{Float64})
    return any([in_rectangle(rect, pos) for rect in r.rectangles])
end 

# Attempts to translate from pos0 in direction heading for des_step without violating boundaries
function legal_translate(r::Room, pos0::AbstractVector{Float64}, heading::AbstractVector{Float64}, des_step::Float64)
    fs = minimum(furthest_step(rect, pos0, heading) for rect in r.rectangles)
    fs = min(des_step, fs)
    pos1 = pos0 + fs*heading
    if !in_room(r, pos1)
        return pos0
    else
        return pos1
    end
end

# computes the length of a ray from robot center to closest segment
# from p0 pointing in direction heading
# inputs: p0: array specifying initial point
#         heading: array specifying heading unit vector
#         R: robot radius [m]
# outputs: ray_length [m]
function ray_length(r::Room, pos0::AbstractVector{Float64}, heading::AbstractVector{Float64})
    return minimum(ray_length(rect, pos0, heading) for rect in r.rectangles)
end

# Render room based on individual rectangles
function render(r::Room, ctx::CairoContext)
    for rect in r.rectangles
        render(rect, ctx)
    end
end
