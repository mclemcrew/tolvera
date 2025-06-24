"""Three blue pixels moving left to right at different times."""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Main function to demonstrate three blue pixels moving at different speeds.
    """
    tv = Tolvera(n=3, species=1, **kwargs)  # Just 3 particles for the pixels
    
    # Timing and movement variables
    start_times = ti.field(ti.f32, shape=3)  # When each pixel starts moving
    move_speeds = ti.field(ti.f32, shape=3)  # Speed of each pixel
    finished = ti.field(ti.i32, shape=3)     # Whether each pixel finished
    
    @ti.kernel
    def init_pixels():
        """Initialize the three pixels with different start times and speeds."""
        # Pixel positions (start on left side)
        tv.p.field[0].pos = ti.Vector([50.0, tv.y/2 - 50])   # Top pixel
        tv.p.field[1].pos = ti.Vector([50.0, tv.y/2])        # Middle pixel  
        tv.p.field[2].pos = ti.Vector([50.0, tv.y/2 + 50])   # Bottom pixel
        
        # Reset velocities
        for i in range(3):
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].active = 1.0
            tv.p.field[i].size = 15.0
            tv.p.field[i].species = 0
        
        # Different start times (in frames)
        start_times[0] = 60.0   # First pixel starts after 1 second (at 60fps)
        start_times[1] = 120.0  # Second pixel starts after 2 seconds
        start_times[2] = 180.0  # Third pixel starts after 3 seconds
        
        # Different movement speeds
        move_speeds[0] = 3.0    # Fastest
        move_speeds[1] = 2.0    # Medium
        move_speeds[2] = 1.5    # Slowest
        
        # Reset finished flags
        for i in range(3):
            finished[i] = 0

    @ti.kernel
    def move_pixels():
        """Move pixels based on their timing and speed."""
        frame = tv.ctx.i[None]
        
        for i in range(3):
            # Check if it's time for this pixel to start moving
            if frame >= start_times[i] and finished[i] == 0:
                # Move the pixel to the right
                tv.p.field[i].pos[0] += move_speeds[i]
                
                # Stop when reaching the right side
                if tv.p.field[i].pos[0] >= tv.x - 100:
                    finished[i] = 1
                    tv.p.field[i].pos[0] = tv.x - 100  # Clamp position

    @ti.kernel
    def draw_pixels():
        """Draw the blue pixels and clear background."""
        # Dark background
        tv.px.background(0.05, 0.05, 0.1)
        
        # Draw each pixel as a blue circle
        blue_color = ti.Vector([0.2, 0.4, 1.0, 1.0])
        
        for i in range(3):
            x = ti.cast(tv.p.field[i].pos[0], ti.i32)
            y = ti.cast(tv.p.field[i].pos[1], ti.i32)
            size = ti.cast(tv.p.field[i].size, ti.i32)
            
            # Only draw if pixel is active
            if tv.p.field[i].active > 0:
                tv.px.circle(x, y, size, blue_color, fill=1)

    # Initialize the pixels
    init_pixels()
    
    @tv.render
    def _():
        move_pixels()     # Update pixel positions
        draw_pixels()     # Draw the pixels
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\nExiting.")