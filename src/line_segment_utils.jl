# functions for determining whether the Roomba's path interects
# with a line segment and struct defining line segments
# maintained by {jmorton2,kmenda}@stanford.edu

const SVec2 = SVector{2, Float64}
@inline dot(x::SVec2, y::SVec2) = x[1]*y[1] + x[2]*y[2]

const MARGIN = 1e-8
"""
finds the real points of intersection between a line and a circle
inputs: 
- `p0::SVec2` anchor point 
- `uvec::SVec2` unit vector specifying heading
- `p1::SVec2` centroid (x,y) of a circle
- `R::Float64` radius of a circle
returns: 
- `R1,R2::Float64` where R1,R2 are lengths of vec to get from p0 to the intersecting
         points. If intersecting points are imaginary, returns `NaN` in their place
"""
function real_intersect_line_circle(p0::SVec2, 
                                    uvec::SVec2, 
                                    p1::SVec2, 
                                    R::Float64)

    diff = p1 - p0
    det = uvec[2]*diff[1] - uvec[1]*diff[2]
    radicand =  R - det * det
    if radicand < 0 # intersecting points are imaginary
        return NaN
    else
        return dot(uvec, diff) - sqrt(radicand)
    end
end

"""
finds the intersection between a line and a line segment
inputs: 
- `p0::SVec2` anchor point 
- `uvec::SVec2` unit vector specifying heading
- `p1, p2::SVec2` x,y of the endpoints of the segment
returns: 
- `sol::SVec2` x,y of intersection or `NaN` if doesn't intersect
"""
function intersect_line_linesegment(p0::SVec2, uvec::SVec2, p1::SVec2, p2::SVec2)
    dx, dy = uvec
    n = SVec2(-dy, dx)
    p10 = p1 - p0
    if sign(dot(n, p10)) != sign(dot(n, p2-p0))
        # there's an intersection
        x21 = p2[1] - p1[1]
        y21 = p2[2] - p1[2]

        R = (x21*p10[2] - y21*p10[1])/(x21*dy - y21*dx)
        if R >= 0
            return R
        end
    end
    return NaN
end

# Define LineSegment
mutable struct LineSegment
    p1::SVec2 # anchor point of line-segment
    p2::SVec2 # anchor point of line-segment
    goal::Bool # used for rendering purposes
    stairs::Bool # used for rendering purposes
    n::SVec2 # normal vector
end

function LineSegment(p1::AbstractVector{Float64}, p2::AbstractVector{Float64}, goal::Bool,stairs::Bool)
    dp12 = p2 - p1
    dp12_sum = sum(dp12)
    np12 = SVec2(-dp12[2]/dp12_sum, dp12[1]/dp12_sum)
    return LineSegment(p1, p2, goal, stairs, np12)
end

"""
computes the length of a ray from robot center to segment from p0 pointing in direction heading
inputs: 
- `ls::LineSegment` line segment under test
- `p0::SVec2` initial point being travelled from
- `heading::SVec2` heading unit vector
returns: 
- `::Float64` that is the length of the ray
"""
function ray_length(ls::LineSegment, p0::SVec2, heading::SVec2)
    intr = intersect_line_linesegment(p0, heading, ls.p1, ls.p2)
    if isnan(intr)
        return Inf
    else
        return intr
    end
end

"""
computes the furthest step a robot of radius R can take
inputs: 
- `ls::LineSegment` line segment under test
- `p0::SVec2` initial point being travelled from
- `heading::SVec2` heading unit vector
- `des_step::Float64` desired step
- `R::Float64` radius of robot
returns: 
- `furthest_step::Float64` furthest step the robot can take
--
The way this is computed is by seeing if a ray originating from
p0 in direction heading intersects the following object. Consider
the shape made by moving the robot along the length of the segment.
We can construct this shape by placing a circle with radius of
the robot radius R at each end, and connecting their sides by shifting
segment line out to its left and right by R.
If the line from p0 intersects this object, then choosing the closest 
intersection gives the point at which the robot would stop if traveling
along this line.
"""
function furthest_step(ls::LineSegment, p0::SVec2, heading::SVec2, R::Float64)
    furthest_step = Inf
    f(s) = isnan(s) ? furthest_step : min(max(-MARGIN, s), furthest_step)

    # intesection with p1
    furthest_step = f(real_intersect_line_circle(p0, heading, ls.p1, R))

    # intesection with p2
    furthest_step = f(real_intersect_line_circle(p0, heading, ls.p2, R))

    # make sure the normal vector goes opposite with heading
    if dot(ls.n, heading) > 0
        ls.n *= -1.0
    end

    # project sides out a robot radius
    p1 = ls.p1 + R * ls.n
    p2 = ls.p2 + R * ls.n

    # intersection with the segment
    furthest_step = f(intersect_line_linesegment(p0, heading, p1, p2))

    return max(furthest_step, 0.0)
end

# Transform coordinates in world frame to coordinates used for rendering
function transform_coords(pos::SVec2)
    x, y = pos

    # Specify dimensions of window
    h = 600
    w = 600

    # Perform conversion
    x_trans = (x + 30.0)/50.0*h
    y_trans = -(y - 20.0)/50.0*w

    x_trans, y_trans
end

# Draw line in gtk window based on start and end coordinates
function render(ls::LineSegment, ctx::CairoContext)
    start_x, start_y = transform_coords(ls.p1)
    if ls.goal
        set_source_rgb(ctx, 0, 1, 0)
    elseif ls.stairs
        set_source_rgb(ctx, 1, 0, 0)
    else
        set_source_rgb(ctx, 0, 0, 0)
    end
    move_to(ctx, start_x, start_y)
    end_x, end_y = transform_coords(ls.p2)
    line_to(ctx, end_x, end_y)
    stroke(ctx)
end
