"""Field of particles with wave motion radiating outward while rotating clockwise."""

import taichi as ti
from tolvera import Tolvera, run

def main(**kwargs):
    """
    Main function demonstrating a field of particles with wave motion.
    """
    # Create a grid of particles
    grid_size = 20  # 20x20 = 400 particles
    total_particles = grid_size * grid_size
    
    tv = Tolvera(n=total_particles, species=1, **kwargs)
    
    # Wave and rotation parameters
    wave_time = ti.field(ti.f32, shape=())
    rotation_time = ti.field(ti.f32, shape=())
    wave_speed = ti.field(ti.f32, shape=())
    rotation_speed = ti.field(ti.f32, shape=())
    wave_amplitude = ti.field(ti.f32, shape=())
    
    # Grid layout parameters
    center_x = tv.x / 2
    center_y = tv.y / 2
    grid_spacing = ti.field(ti.f32, shape=())
    
    # Store original grid positions for each particle
    base_positions = ti.Vector.field(2, ti.f32, shape=total_particles)
    
    @ti.kernel
    def init_particles():
        """Initialize particles in a grid formation."""
        wave_time[None] = 0.0
        rotation_time[None] = 0.0
        wave_speed[None] = 0.05      # Speed of wave propagation
        rotation_speed[None] = 0.01   # Speed of clockwise rotation
        wave_amplitude[None] = 25.0   # Height of the wave effect
        grid_spacing[None] = 30.0     # Distance between particles
        
        # Calculate grid offset to center the grid
        total_width = (grid_size - 1) * grid_spacing[None]
        start_x = center_x - total_width / 2
        start_y = center_y - total_width / 2
        
        # Place particles in grid and store base positions
        for i in range(total_particles):
            grid_x = i % grid_size
            grid_y = i // grid_size
            
            # Calculate base position in grid
            base_x = start_x + grid_x * grid_spacing[None]
            base_y = start_y + grid_y * grid_spacing[None]
            
            base_positions[i] = ti.Vector([base_x, base_y])
            
            # Initialize particle properties
            tv.p.field[i].pos = ti.Vector([base_x, base_y])
            tv.p.field[i].vel = ti.Vector([0.0, 0.0])
            tv.p.field[i].active = 1.0
            tv.p.field[i].size = 3.0
            tv.p.field[i].species = 0
    
    @ti.kernel
    def update_wave_motion():
        """Update particle positions with wave and rotation effects."""
        # Increment time
        wave_time[None] += wave_speed[None]
        rotation_time[None] += rotation_speed[None]
        
        # Rotation matrix components
        cos_rot = ti.cos(rotation_time[None])
        sin_rot = ti.sin(rotation_time[None])
        
        for i in range(total_particles):
            # Get base position
            base_pos = base_positions[i]
            
            # Apply rotation around center
            rel_x = base_pos[0] - center_x
            rel_y = base_pos[1] - center_y
            
            rotated_x = rel_x * cos_rot - rel_y * sin_rot
            rotated_y = rel_x * sin_rot + rel_y * cos_rot
            
            final_x = rotated_x + center_x
            final_y = rotated_y + center_y
            
            # Calculate distance from center for wave effect
            dist_from_center = ti.sqrt(rotated_x * rotated_x + rotated_y * rotated_y)
            
            # Create wave that radiates outward from center
            wave_phase = wave_time[None] + dist_from_center * 0.02
            wave_height = ti.sin(wave_phase) * wave_amplitude[None]
            
            # Apply wave as brightness/size variation (simulating z-position)
            wave_factor = (ti.sin(wave_phase) + 1.0) / 2.0  # 0 to 1
            
            # Update particle position
            tv.p.field[i].pos = ti.Vector([final_x, final_y])
            
            # Store wave factor in particle size for visual effect
            tv.p.field[i].size = 2.0 + wave_factor * 6.0  # Size varies from 2 to 8
            
            # Store wave height in mass for color calculation
            tv.p.field[i].mass = wave_factor
    
    @ti.kernel
    def draw_particles():
        """Draw particles with wave-based brightness and size."""
        # Clear background
        tv.px.background(0.02, 0.02, 0.05)
        
        for i in range(total_particles):
            if tv.p.field[i].active <= 0:
                continue
                
            x = ti.cast(tv.p.field[i].pos[0], ti.i32)
            y = ti.cast(tv.p.field[i].pos[1], ti.i32)
            size = ti.cast(tv.p.field[i].size, ti.i32)
            
            # Use wave factor (stored in mass) for brightness
            wave_factor = tv.p.field[i].mass
            
            # Create color based on wave height and distance from center
            dist_from_center = ti.sqrt((x - center_x)**2 + (y - center_y)**2)
            hue = (dist_from_center * 0.003 + wave_time[None] * 0.5) % 1.0
            
            # Convert hue to RGB with wave-based brightness
            brightness = 0.3 + wave_factor * 0.7  # Brightness varies with wave
            
            # Simple HSV to RGB conversion
            h = hue * 6.0
            sector = ti.cast(h, ti.i32) % 6
            f = h - ti.cast(h, ti.i32)
            p = brightness * (1.0 - 1.0)  # saturation = 1
            q = brightness * (1.0 - f)
            t = brightness * (1.0 - (1.0 - f))
            
            r, g, b = 0.0, 0.0, 0.0
            if sector == 0:
                r, g, b = brightness, t, p
            elif sector == 1:
                r, g, b = q, brightness, p
            elif sector == 2:
                r, g, b = p, brightness, t
            elif sector == 3:
                r, g, b = p, q, brightness
            elif sector == 4:
                r, g, b = t, p, brightness
            else:  # sector == 5
                r, g, b = brightness, p, q
            
            # Add white highlight for peaks
            white_factor = wave_factor * 0.5
            r = ti.min(r + white_factor, 1.0)
            g = ti.min(g + white_factor, 1.0)
            b = ti.min(b + white_factor, 1.0)
            
            color = ti.Vector([r, g, b, 1.0])
            
            # Draw particle with size based on wave height
            if size > 0:
                tv.px.circle(x, y, size, color, fill=1)
                
                # Add glow effect for brighter particles
                if wave_factor > 0.7:
                    glow_color = ti.Vector([r * 0.3, g * 0.3, b * 0.3, 0.3])
                    tv.px.circle(x, y, size + 3, glow_color, fill=0)
    
    # Initialize the animation
    init_particles()
    
    @tv.render
    def _():
        update_wave_motion()  # Update wave and rotation
        draw_particles()      # Draw with wave effects
        return tv.px

if __name__ == '__main__':
    try:
        run(main)
    except KeyboardInterrupt:
        print("\nExiting.")