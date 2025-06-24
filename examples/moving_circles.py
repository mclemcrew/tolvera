"""Move individual larger pixels around the screen."""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Main function to demonstrate individual pixel movement with bigger pixels.
    """
    tv = Tolvera(n=10, species=1, **kwargs)  # Just 10 particles for clarity
    
    # Store some pixel positions for manual control
    pixel_pos = ti.Vector.field(2, ti.f32, shape=5)  # 5 individual pixels
    pixel_colors = ti.Vector.field(4, ti.f32, shape=5)  # Colors for each pixel
    pixel_sizes = ti.field(ti.i32, shape=5)  # Size for each pixel
    
    @ti.kernel
    def init_pixels():
        """Initialize our individual pixels."""
        for i in range(5):
            # Spread pixels across screen
            pixel_pos[i] = ti.Vector([
                (i + 1) * tv.x / 6,  # x position
                tv.y / 2             # y position (center)
            ])
            # Different colors for each pixel
            pixel_colors[i] = ti.Vector([
                i / 4.0,      # red component
                1.0 - i / 4.0, # green component  
                0.5,          # blue component
                1.0           # alpha
            ])
            # Different sizes for each pixel
            pixel_sizes[i] = 10 + i * 5  # Sizes from 10 to 30 pixels
    
    @ti.kernel
    def move_pixels():
        """Move individual pixels."""
        for i in range(5):
            # Move each pixel in a different pattern
            time = tv.ctx.i[None] * 0.02  # Use frame counter as time
            
            if i == 0:
                # Horizontal oscillation
                pixel_pos[i][0] += ti.sin(time) * 2.0
            elif i == 1:
                # Vertical oscillation  
                pixel_pos[i][1] += ti.cos(time) * 2.0
            elif i == 2:
                # Circular motion
                radius = 80.0
                pixel_pos[i][0] = tv.x/2 + radius * ti.cos(time)
                pixel_pos[i][1] = tv.y/2 + radius * ti.sin(time)
            elif i == 3:
                # Diagonal drift
                pixel_pos[i][0] += 1.0
                pixel_pos[i][1] += 0.5
                # Wrap around screen (accounting for size)
                size = pixel_sizes[i]
                if pixel_pos[i][0] > tv.x + size:
                    pixel_pos[i][0] = -size
                if pixel_pos[i][1] > tv.y + size:
                    pixel_pos[i][1] = -size
            else:
                # Random walk
                pixel_pos[i][0] += (ti.random() - 0.5) * 4.0
                pixel_pos[i][1] += (ti.random() - 0.5) * 4.0
                
                # Keep in bounds (accounting for size)
                size = pixel_sizes[i]
                pixel_pos[i][0] = ti.max(size, ti.min(tv.x-size, pixel_pos[i][0]))
                pixel_pos[i][1] = ti.max(size, ti.min(tv.y-size, pixel_pos[i][1]))
    
    @ti.kernel
    def draw_pixels():
        """Draw our individual bigger pixels."""
        # Clear with dark background
        tv.px.background(0.1, 0.1, 0.1)
        
        # Draw each individual pixel as different shapes
        for i in range(5):
            x = ti.cast(pixel_pos[i][0], ti.i32)
            y = ti.cast(pixel_pos[i][1], ti.i32)
            size = pixel_sizes[i]
            color = pixel_colors[i]
            
            if i == 0:
                # Draw as a filled circle
                tv.px.circle(x, y, size, color, fill=1)
            elif i == 1:
                # Draw as a filled rectangle
                tv.px.rect(x - size//2, y - size//2, size, size, color, fill=1)
            elif i == 2:
                # Draw as an unfilled circle (ring)
                tv.px.circle(x, y, size, color, fill=0)
                # With a smaller filled circle inside
                tv.px.circle(x, y, size//3, color, fill=1)
            elif i == 3:
                # Draw as an unfilled rectangle (border only)
                tv.px.rect(x - size//2, y - size//2, size, size, color, fill=0)
            else:
                # Draw as a triangle (using three points to approximate)
                half_size = size // 2
                # Create triangle points
                p1 = ti.Vector([x, y - half_size])      # top
                p2 = ti.Vector([x - half_size, y + half_size])  # bottom left
                p3 = ti.Vector([x + half_size, y + half_size])  # bottom right
                tv.px.triangle(p1, p2, p3, color, fill=1)
    
    # Initialize pixels once
    init_pixels()
    
    @tv.render
    def _():
        move_pixels()    # Update pixel positions
        draw_pixels()    # Draw the bigger pixels
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\nExiting.")