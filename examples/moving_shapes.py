"""Three shapes moving across the screen in sequence: blue pixel, green triangle, and yellow star."""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Main function demonstrating three shapes moving in sequence.
    """
    tv = Tolvera(n=3, species=1, **kwargs)
    
    # Shape positions and states
    pixel_pos = ti.Vector.field(2, ti.f32, shape=())
    triangle_pos = ti.Vector.field(2, ti.f32, shape=())
    star_pos = ti.Vector.field(2, ti.f32, shape=())
    
    # Movement states
    pixel_finished = ti.field(ti.i32, shape=())
    triangle_finished = ti.field(ti.i32, shape=())
    star_finished = ti.field(ti.i32, shape=())
    
    # Movement speeds
    pixel_speed = ti.field(ti.f32, shape=())
    triangle_speed = ti.field(ti.f32, shape=())
    star_speed = ti.field(ti.f32, shape=())
    
    @ti.kernel
    def init_shapes():
        """Initialize all shapes at their starting positions."""
        # Blue pixel: starts at left, moves right
        pixel_pos[None] = ti.Vector([50.0, tv.y / 2])
        pixel_speed[None] = 10.0  # Increased speed
        pixel_finished[None] = 0
        
        # Green triangle: starts at top, moves down  
        triangle_pos[None] = ti.Vector([tv.x / 2, tv.y - 50])  # Start at top (tv.y)
        triangle_speed[None] = 10.0  # Speed for downward movement
        triangle_finished[None] = 0
        
        # Yellow star: starts at top-right, moves to bottom-left
        star_pos[None] = ti.Vector([tv.x - 30.0, tv.y - 50])  # Start at top-right
        star_speed[None] = 10.0  # Speed for diagonal movement
        star_finished[None] = 0

    @ti.kernel
    def move_shapes():
        """Move shapes based on their current state."""
        frame = tv.ctx.i[None]
        
        # Blue pixel: starts immediately, moves left to right
        if pixel_finished[None] == 0:
            pixel_pos[None][0] += pixel_speed[None]
            if pixel_pos[None][0] >= tv.x - 50:
                pixel_finished[None] = 1
                pixel_pos[None][0] = tv.x - 50  # Clamp position
        
        # Green triangle: starts after pixel finishes, moves top to bottom
        if pixel_finished[None] == 1 and triangle_finished[None] == 0:
            triangle_pos[None][1] -= triangle_speed[None]  # Move DOWN (decrease y from tv.y toward 0)
            if triangle_pos[None][1] <= 50:
                triangle_finished[None] = 1
                triangle_pos[None][1] = 50.0  # Clamp position near bottom
        
        # Yellow star: starts after triangle finishes, moves diagonally
        if triangle_finished[None] == 1 and star_finished[None] == 0:
            # Move diagonally from top-right to bottom-left
            star_pos[None][0] -= star_speed[None] * 0.7  # Move LEFT (decrease x)
            star_pos[None][1] -= star_speed[None] * 0.7  # Move DOWN (decrease y from tv.y toward 0)
            if star_pos[None][0] <= 50 or star_pos[None][1] <= 50:
                star_finished[None] = 1
                # Clamp to final position
                star_pos[None][0] = ti.max(50.0, star_pos[None][0])
                star_pos[None][1] = ti.max(50.0, star_pos[None][1])

    @ti.func
    def draw_star(x: ti.i32, y: ti.i32, size: ti.i32, color: ti.math.vec4):
        """Draw a filled 5-pointed star using polygon approach."""
        outer_radius = ti.cast(size, ti.f32)
        inner_radius = outer_radius * 0.38  # Good ratio for star shape
        
        # Create arrays for the 10 star points (5 outer, 5 inner, alternating)
        star_x = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        star_y = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Calculate all 10 points - need to do this without variable assignment in loop
        for i in ti.static(range(10)):
            angle = i * ti.math.pi / 5.0 - ti.math.pi / 2.0  # Start from top
            
            # Calculate radius directly in the assignment
            if i % 2 == 0:
                # Outer points
                star_x[i] = ti.cast(x, ti.f32) + outer_radius * ti.cos(angle)
                star_y[i] = ti.cast(y, ti.f32) + outer_radius * ti.sin(angle)
            else:
                # Inner points
                star_x[i] = ti.cast(x, ti.f32) + inner_radius * ti.cos(angle)
                star_y[i] = ti.cast(y, ti.f32) + inner_radius * ti.sin(angle)
        
        # Draw the star as a filled polygon
        tv.px.polygon(star_x, star_y, color, fill=1)

    @ti.kernel
    def draw_shapes():
        """Draw all shapes in their current positions."""
        # Dark background
        tv.px.background(0.02, 0.02, 0.05)
        
        # Colors
        blue_color = ti.Vector([0.2, 0.4, 1.0, 1.0])
        green_color = ti.Vector([0.2, 0.8, 0.3, 1.0])
        yellow_color = ti.Vector([1.0, 0.9, 0.2, 1.0])
        
        # Draw blue pixel (circle)
        if pixel_finished[None] == 0 or (pixel_finished[None] == 1):
            px = ti.cast(pixel_pos[None][0], ti.i32)
            py = ti.cast(pixel_pos[None][1], ti.i32)
            tv.px.circle(px, py, 20, blue_color, fill=1)
        
        # Draw green triangle
        if (pixel_finished[None] == 1 and triangle_finished[None] == 0) or triangle_finished[None] == 1:
            tx = ti.cast(triangle_pos[None][0], ti.i32)
            ty = ti.cast(triangle_pos[None][1], ti.i32)
            
            # Triangle points (flipped for inverted y-axis - point DOWN)
            size = 30
            p1 = ti.Vector([tx, ty + size])          # Bottom point (tip pointing down)
            p2 = ti.Vector([tx - size, ty - size])   # Top left
            p3 = ti.Vector([tx + size, ty - size])   # Top right
            
            tv.px.triangle(p1, p2, p3, green_color, fill=1)
        
        # Draw yellow star
        if triangle_finished[None] == 1:
            sx = ti.cast(star_pos[None][0], ti.i32)
            sy = ti.cast(star_pos[None][1], ti.i32)
            draw_star(sx, sy, 25, yellow_color)

    # Initialize shapes
    init_shapes()
    
    @tv.render
    def _():
        move_shapes()  # Update positions
        draw_shapes()  # Draw all shapes
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\nExiting.")