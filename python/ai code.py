import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# --- Target Parameters from the Paper (Fig. 1, U=4, Pe=100) ---
target_U = 4.0
target_Pe = 100.0

# --- Set Fundamental Simulation Parameters ---
sigma = 1.0  # Particle diameter (reduced unit of length)
gamma = 1.0  # Friction coefficient
epsilon = 1.0 # LJ energy scale (reduced unit of energy)
kb = 1.0     # Boltzmann constant (reduced unit)

# --- Calculated Parameters for the Simulation ---
T = epsilon / (kb * target_U)           # Temperature
D = (kb * T) / gamma                    # Translational diffusion coefficient
tau_B = (sigma**2) / D                  # Brownian time
v0 = (target_Pe * sigma) / tau_B        # Self-propulsion velocity
Dr = (3.0 * D) / (sigma**2)             # Rotational diffusion coefficient

# --- Other Simulation Settings ---
N = 2500                                # Increased number of particles
phi = 0.4                               # Area fraction from the paper
rho = phi / (np.pi * (sigma/2)**2)      # Number density
L = np.sqrt(N / rho)                    # Box side length
r_c = 2.5 * sigma                       # Cutoff radius for LJ potential

# --- Time Settings ---
# Let's use a slightly smaller, safer timestep
t = 0.001
# We want to run for 1000 Brownian times (1000 * tau_B)
total_time = 1000 * tau_B
reps = int(total_time / t)
frame_interval = 2000 # How often to save a frame for the animation

print("--- Simulating Active Colloids ---")
print(f"Targeting: U = {target_U:.2f}, Pe = {target_Pe:.2f}")
print(f"Box Side L = {L:.2f}, N = {N}, Density rho = {rho:.2f}")
print(f"Propulsion v0 = {v0:.2f}, Temperature T = {T:.3f}")
print(f"Total time = {total_time:.2f} ({reps} steps)")
print("-----------------------------------")

@njit
def get_forces_and_potential_2d(coords, L, r_c, N):
    """
    Calculates LJ forces and potential for all particles using Numba for speed.
    """
    forces = np.zeros_like(coords)
    potential_energy = 0.0
    r_c2 = r_c * r_c
    
    # Pre-calculate shifted potential at cutoff
    sr_c6 = 1.0 / (r_c**6)
    U_shift = 4.0 * (sr_c6*sr_c6 - sr_c6)

    for i in range(N):
        for j in range(i + 1, N):
            rij = coords[i, :] - coords[j, :]
            
            # Minimum image convention
            rij = rij - L * np.round(rij / L)
            
            r2 = np.sum(rij**2)
            
            if r2 < r_c2:
                # Use inverse r2 for efficiency
                ir2 = 1.0 / r2
                ir6 = ir2 * ir2 * ir2
                
                # Force calculation
                f_magnitude = 48.0 * ir2 * ir6 * (ir6 - 0.5)
                forces[i, :] += f_magnitude * rij
                forces[j, :] -= f_magnitude * rij
                
                # Potential energy calculation (with shift)
                potential_energy += 4.0 * ir6 * (ir6 - 1.0) - U_shift

    return forces, potential_energy

def initialisation_random(N, L):
    """Initializes random positions and orientations."""
    coordinates = L * np.random.rand(N, 2)
    angles = 2 * np.pi * np.random.rand(N)
    orientations = np.column_stack([np.cos(angles), np.sin(angles)])
    return coordinates, orientations

# --- Main Simulation Loop ---
x, e = initialisation_random(N, L)
plot_cords = []
U_series = []

for i in range(reps):
    # Calculate forces using our self-contained function
    f, U = get_forces_and_potential_2d(x, L, r_c, N)
    
    # Thermal noise
    eta_T = np.random.normal(scale=1.0, size=(N, 2))
    # Rotational noise
    eta_R = np.random.normal(scale=1.0, size=N)
    
    # Update positions (Euler-Maruyama)
    x += (t / gamma) * f + v0 * e * t + np.sqrt(2 * D * t) * eta_T
    
    # Update orientations
    d_theta = np.sqrt(2 * Dr * t) * eta_R
    e_new_x = e[:, 0] * np.cos(d_theta) - e[:, 1] * np.sin(d_theta)
    e_new_y = e[:, 0] * np.sin(d_theta) + e[:, 1] * np.cos(d_theta)
    e = np.column_stack([e_new_x, e_new_y])
    
    # Apply periodic boundary conditions
    x = x % L 
    
    # --- Data Logging ---
    if i % frame_interval == 0:
        print(f"Step: {i}/{reps}, Potential Energy/N: {U/N:.3f}")
        plot_cords.append(x.copy()) # Use .copy()!
    U_series.append(U/N)

print("Simulation finished!")

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 8))

def animate(frame):
    ax.clear()
    coords = plot_cords[frame]
    
    # Plot particles as points for speed with many particles
    ax.plot(coords[:,0], coords[:,1], 'o', markersize=2, color='steelblue')
            
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0, L])
    ax.set_ylim([0, L])
    time_stamp = frame * frame_interval * t
    ax.set_title(f"Active Colloid Simulation (U={target_U}, Pe={target_Pe})\nTime = {time_stamp:.1f} (or {time_stamp/tau_B:.1f} tau)")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

ani = FuncAnimation(fig, animate, frames=len(plot_cords), interval=100)
plt.show()

# You can also save the animation
# ani.save('active_colloids_U4_Pe100.mp4', writer='ffmpeg', fps=10)