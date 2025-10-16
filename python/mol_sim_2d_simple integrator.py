import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import molsim
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
#reduced parameters(sigma=1,epsilon=1,m=1)
'''T=0.25  # Temperature in reduced units
U=1/T
t=0.0001# Time step in reduced units
rho=0.65#0.840
reps=10000
N=225
r_c=2.5  # Cutoff radius for the Lennard-Jones potential
gamma=1 # Friction coefficient for Langevin thermostat
D=T/gamma #translational diffusion coeff
Dr=3*D
v0=25#self propulsion velocity
mag=1.5
sigma=np.sqrt(2*gamma*T)  # Noise strength for Langevin thermostat
tau=3/Dr#persistence time
pe=v0*tau
print("Peclet number:",pe)
print("ATTRACTION",U)'''
target_U = 4 # Increased attraction
target_Pe =140   # Decreased activity

# --- Set Fundamental Simulation Parameters ---
sigma = 1.0  # Particle diameter
gamma = 1.0  # Friction coefficient (a convenient choice)

# --- Calculated Parameters for the Simulation ---
T = 1.0 / target_U                      # Temperature
D = T / gamma                           # Translational diffusion coefficient
tau = (sigma**2) / D                    # Brownian time
v0 = (target_Pe * sigma) / tau          # Self-propulsion velocity
Dr = (3.0 * D) / (sigma**2)             # Rotational diffusion coefficient

# --- Other Simulation Settings ---
t = 0.01                                # Time step
# Return to the paper's density for phi=0.4
rho = 0.5 / (np.pi/4)      
reps = 10000                    # Keep the long run time!
N =5000                         # Number of particles
r_c = 2.5 
frame_rate=100
print(f"NEW TARGET: Attraction-dominated gel")
print(f"Simulating for U = {1/T:.2f} and Pe = {v0*tau/sigma:.2f}")
print("---------------------------------")
print(f"T = {T}")
print(f"v0 = {v0}")
print(f"Dr = {Dr}")
print(f"rho = {rho}")
#initial positions and velocities
'''def initialisation(N=484,rho=0.84):
    # Box length from density
    L = np.sqrt(N / rho)
    print(L )
    # Find number of particles along one side
    n = int(np.round((np.sqrt(N))))
    c=n*n
    print(N,c)
    if c!= N:
        raise ValueError("N must be a perfect square for square lattice.")
    a = L / n  # lattice spacing
    coords = []
    for i in range(n):
        for j in range(n):
            coords.append([i * a, j * a])
    coordinates=np.array(coords)
    print(coordinates.shape)
    velocities=np.random.rand(*coordinates.shape)-0.5 
    avg_v=np.mean(velocities, axis=0)
    velocities=velocities-avg_v
    avg_v2=np.mean(np.linalg.norm(velocities, axis=1)**2)
    scaling_factor=np.sqrt(2*T/(avg_v2))
    velocities=(velocities)*scaling_factor#dont actually needed for langevin dynamics
    e=np.random.rand(*coordinates.shape)-0.5
    norm=np.linalg.norm(e, axis=1,keepdims=True)
    e=e/norm
    return coordinates,velocities,L,avg_v,avg_v2,scaling_factor,e'''
def initialisation_random(N=225, rho=0.509):
    L = np.sqrt(N / rho)
    print(f"Box side length: {L}")
    # Generate random coordinates within the box
    coordinates = L * np.random.rand(N, 2)
    
    # Active particle orientations (unit vectors)
    angles = 2 * np.pi * np.random.rand(N)
    e = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # No need to initialize velocities for overdamped dynamics
    velocities = np.zeros_like(coordinates) 
    
    return coordinates, velocities, L, e
coordinates, velocities, L, e = initialisation_random(N, rho)
plt.plot(coordinates[:,0],coordinates[:,1],'o')
#coordinates,velocities,L,avg_v,avg_v2,scaling_factor,e=initialisation(N,rho)
coordinate_last=coordinates-velocities*t
def test_force(coordinates):
    return -1*coordinates
T_series = []
U_series=[]
test_coordinates=0
test_velocities=0
plot_cords=[]
def integrate(coordinates,velocities,e,time=10000):
    x=coordinates
    v=velocities
    for i in range(time):
        h=t
        eta=np.random.normal(0,1,(N,2))
        d_theta = np.random.normal(scale=np.sqrt(2 * Dr * h), size=N) 
        # 2. Compute the rotation update for each particle
        de_rot = np.column_stack((-d_theta * e[:, 1], d_theta * e[:, 0])) 
        # 3. Apply the vector update:
        e = e + de_rot
        norm=np.linalg.norm(e, axis=1,keepdims=True)
        e=e/norm
        coordinates1=x.flatten()
        #f,U=molsim.force2d(coordinates1,L,N,r_c)
        mol=molsim.Simulation(L,r_c,N)
        mol.makegrid(coordinates1)
        f,U=mol.force2dhp(coordinates1)
        f=np.array(f).reshape(N,2)
        h=t
        #simple integrator
        x_n=x+(h/gamma)*f+v0*e*h+np.sqrt(2*D*h)*eta 
        #x_n=x+(h/gamma)*f
        #white noise
        x=x_n 
        x= x % L 
        if i%frame_rate==0:
            plot_cords.append(x)
        #v=v_n
        #U+=N*rho*2*np.pi*(0.5*(1/r_c)**10-(1/r_c)**4)  #replace with 2d tail correction
        #total_energy=U+0.5*np.sum(v_n**2)
        U_series.append(U/N)
        print("Step:", i, "Potential Energy:", (U/N))
integrate(coordinates,velocities,e,reps)
def animate(plot_cords,interval=5):
    fig=plt.figure()#creating a blank canvas
    ax=fig.add_subplot()#creates a specific plotting area in the canvas
    def update(frame):
        cords=plot_cords[frame]       
        ax.clear()#erases everything on the ax plotting area,the points,the tilte , the dimentions everything       
        ax.scatter(cords[:,0],cords[:,1],s=1)
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])
        ax.set_title(f"time{frame*frame_rate:.2f}")
        ax.grid(True)
    
    
    '''def update(frame):
        ax.clear()
        cords = plot_cords[frame]
        
        # Loop through each particle's coordinates and draw a Circle for it
        for i in range(len(cords)):
            x, y = cords[i]
            # Create a circle patch with center (x,y) and the correct radius
            circle = Circle((x, y), radius=0.125, color='steelblue')
            ax.add_patch(circle)
            
        # This is crucial to ensure circles are not drawn as ellipses
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlim([0, L])
        ax.set_ylim([0, L])
        # The time value in your title might need adjusting based on how you save frames
        ax.set_title(f"time step {frame * frame_rate}") # Example: if you save every 2000 steps
        ax.grid(True)'''
    ani=FuncAnimation(fig,update,frames=len(plot_cords),interval=50)
    plt.show()
animate(plot_cords,50)