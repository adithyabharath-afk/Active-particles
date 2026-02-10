import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import molsim
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from Analysis import phic
#reduced parameters(sigma=1,epsilon=1,m=1)
target_U = 1
target_Pe =23
sigma = 1.0  # Particle diameter
gamma = 1.0  # Friction coefficient
gamma_r=3*gamma
B_scale=10**(-5)
nu_scale=1/B_scale
T = 1.0 / target_U                      # Temperature
D = T / gamma                           # Translational diffusion coefficient
#tau = (sigma**2) / D                    # Brownian time
v0 =target_Pe        # Self-propulsion velocity
Dr = T/gamma_r             # Rotational diffusion coefficient
t = 0.00002                      # Time step
rho = 0.23/(np.pi/4)  #papers density 0.4    
reps =5000  # Number of time steps
N = 225# Number of particles# Cutoff radius for the Lennard-Jones potential
frame_rate=200
nu=3
# Magnetic field amplitudes (will vary in time)
# B_x0: x-component amplitude (sin), B_y0: y-component amplitude (cos)
B_x0 = 20
B_y0 = 20
mag_period=1000
omega = 2*np.pi/mag_period
del_c=1.28
'''print(f"NEW TARGET: Attraction-dominated gel")
print(f"Simulating for U = {1/T:.2f} and Pe = {v0*tau/sigma:.2f}")
print("---------------------------------")
print(f"T = {T}")
print(f"v0 = {v0}")
print(f"Dr = {Dr}")
print(f"rho = {rho}")'''
#initial positions and velocities
def initialisation(N=484,rho=0.84):
    # Box length from density
    L = np.sqrt(N / rho)
    #print(L )
    # Find number of particles along one side
    n = int(np.round((np.sqrt(N))))
    c=n*n
    #print(N,c)
    if c!= N:
        raise ValueError("N must be a perfect square for square lattice.")
    a = L / n  # lattice spacing
    coords = []
    for i in range(n):
        for j in range(n):
            coords.append([i * a, j * a])
    coordinates=np.array(coords)
    velocities = np.zeros_like(coordinates)
    return coordinates,velocities,L
coordinates,velocities,L=initialisation(N,rho)
#plt.plot(coordinates[:,0],coordinates[:,1],'o')
T_series = []
r_c=L/2-sigma
U_series=[]
test_coordinates=0
test_velocities=0
plot_cords=[]
phi_c_frames=[]
phi_p_frames=[]
plot_thetas=[]
def integrate(coordinates, velocities, time=10000):
    """
    Simple 1st order integrator for the 2D particle system.
    """
    print("Starting integration (1st order)...")
    x = coordinates.copy()
    
    # Initialize orientation as angle array
    theta = np.random.rand(N) * 2.0 * np.pi
    
    # Pre-calculate noise strengths for efficiency
    sqrt_2DT_dt = np.sqrt(2 * D * t)
    sqrt_2DR_dt = np.sqrt(2 * Dr * t)

    print("Creating simulation object...")
    mol = molsim.Simulation(L, r_c, N, nu, B_x0, B_y0)
    print("Simulation object created successfully")
    
    for i in range(time):
        # Get orientation vector
        orientation = np.column_stack((np.cos(theta), np.sin(theta)))
        
        # Update magnetic field sinusoidally in time
        current_time = i * t
        B_x_t = B_x0 * np.sin(omega * current_time)
        B_y_t = B_y0 * np.cos(omega * current_time)
        mol.set_field(B_x_t, B_y_t)

        # Calculate forces and torques
        coordinates1 = x.flatten()
        orientation1 = orientation.flatten()
        f, torque = mol.force2dhp(coordinates1, orientation1)
        
        # Reshape forces and torques
        f = np.array(f).reshape(N, 2)
        torque = np.array(torque)
        
        # Generate random noise
        G_T = np.random.normal(0, 1, (N, 2))
        G_R = np.random.normal(0, 1, N)
        
        # Calculate noise terms
        W_T = sqrt_2DT_dt * G_T
        W_R = sqrt_2DR_dt * G_R
        
        # Simple 1st order update
        x = x + (t / gamma) * f + v0 * orientation * t + W_T
        theta = theta + (t / gamma_r) * torque + W_R
        
        # Apply periodic boundary conditions
        x = x % L
        
        # Calculate and save phi values at intervals
        if i % frame_rate == 0 and i >= 500:
            phi_p = mol.phip(coordinates1, orientation1, del_c)
            phi_c = phic(np.array(coordinates1).reshape(N, 2), L, N, del_c)
            print(f"Simulation step {i} / {time}, phi_p={phi_p:.4f}, phi_c={phi_c:.4f} (1st order)")
            plot_cords.append(x.copy())
            plot_thetas.append(theta.copy())
            phi_p_frames.append(phi_p)
            phi_c_frames.append(phi_c)

integrate(coordinates, velocities, reps)

def animate(plot_cords, interval):
    fig=plt.figure()#creating a blank canvas
    fig.set_facecolor('black')  # Set figure background to black
    ax=fig.add_subplot()#creates a specific plotting area in the canvas
    ax.set_facecolor('black')  # Set axes background to black
    
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    sm.set_array([])
    
    def update(frame):
        cords=plot_cords[frame]       
        ax.clear()#erases everything on the ax plotting area,the points,the tilte , the dimentions everything       
        ax.set_facecolor('black')  # Reset background to black after clear
        
        # color particles by orientation if available
        if frame < len(plot_thetas):
            thetas = plot_thetas[frame]
            # Normalize theta from [-pi, pi] to [0, 1] for viridis colormap
            thetas_normalized = (thetas + np.pi) / (2 * np.pi)
            sc = ax.scatter(cords[:,0], cords[:,1], c=thetas_normalized, cmap='viridis', vmin=0, vmax=1, s=10)
        else:
            sc = ax.scatter(cords[:,0],cords[:,1],s=10, c='white')
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])
        # Show time and phi values (if available) at the top of the plot
        phi_p_val = None
        phi_c_val = None
        if frame < len(phi_p_frames):
            phi_p_val = phi_p_frames[frame]
        if frame < len(phi_c_frames):
            phi_c_val = phi_c_frames[frame]

        title_parts = [f"time {frame*frame_rate:.2f}"]
        if phi_p_val is not None:
            title_parts.append(f"phi_p={phi_p_val:.3f}")
        if phi_c_val is not None:
            title_parts.append(f"phi_c={phi_c_val:.3f}")
        ax.set_title(" | ".join(title_parts), color='white')  # White title for visibility
        ax.grid(True, color='gray', alpha=0.3)  # Gray grid for visibility on black background
        ax.tick_params(colors='white')  # White tick labels
        
        # Draw rotating magnetic field indicator in bottom-right corner
        # Calculate current time for this frame
        current_time = frame * frame_rate * t
        B_x_t = B_x0 * np.sin(omega * current_time)
        B_y_t = B_y0 * np.cos(omega * current_time)
        
        # Calculate phase angle of magnetic field
        B_phase = np.arctan2(B_y_t, B_x_t)
        
        # Draw circle in corner (bottom-right)
        corner_x = L * 0.85
        corner_y = L * 0.15
        circle_radius = L * 0.08
        
        # Draw circular outline
        circle = plt.Circle((corner_x, corner_y), circle_radius, color='cyan', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Draw rotating arrow
        arrow_length = circle_radius * 0.8
        arrow_dx = arrow_length * np.cos(B_phase)
        arrow_dy = arrow_length * np.sin(B_phase)
        ax.arrow(corner_x, corner_y, arrow_dx, arrow_dy, 
                head_width=circle_radius*0.3, head_length=circle_radius*0.25, 
                fc='yellow', ec='yellow', linewidth=2)
        
        # Add text label
        ax.text(corner_x, corner_y - circle_radius - L*0.05, 'B field', 
                color='cyan', ha='center', fontsize=10, weight='bold')
        
    ani=FuncAnimation(fig,update,frames=len(plot_cords),interval=100)
    
    # Add colorbar after creating animation
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Orientation (radians)', color='white')
    cbar.ax.tick_params(colors='white')
    
    return ani      # <--- ADD THIS LINE
print("Animation finished. Displaying plot...")
animation_object = animate(plot_cords,100) 

# 2. NOW show the plot at the very end.
plt.show()
'''def animate(plot_cords,interval):
    fig=plt.figure()#creating a blank canvas
    ax=fig.add_subplot()#creates a specific plotting area in the canvas
    def update(frame):
        cords=plot_cords[frame]       
        ax.clear()#erases everything on the ax plotting area,the points,the tilte , the dimentions everything       
        ax.scatter(cords[:,0],cords[:,1],s=2)
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])
        ax.set_title(f"time{frame*frame_rate:.2f}")
        ax.grid(True)  
    ani=FuncAnimation(fig,update,frames=len(plot_cords),interval=100)
    plt.show()
animate(plot_cords,100)'''
