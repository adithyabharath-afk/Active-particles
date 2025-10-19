import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import molsim
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
#reduced parameters(sigma=1,epsilon=1,m=1)
target_U = 6
target_Pe =140


sigma = 1.0  # Particle diameter
gamma = 1.0  # Friction coefficient

# --- Calculated Parameters for the Simulation ---
T = 1.0 / target_U                      # Temperature
D = T / gamma                           # Translational diffusion coefficient
tau = (sigma**2) / D                    # Brownian time
v0 = (target_Pe * sigma) / tau          # Self-propulsion velocity
Dr = (3.0 * D) / (sigma**2)             # Rotational diffusion coefficient
t = 0.00025                              # Time step
rho = 0.4 / (np.pi/4)  #papers density 0.4    
reps = 1500000        
N =10000   # Number of particles
r_c = 2.5  # Cutoff radius for the Lennard-Jones potential
frame_rate=100
print(f"NEW TARGET: Attraction-dominated gel")
print(f"Simulating for U = {1/T:.2f} and Pe = {v0*tau/sigma:.2f}")
print("---------------------------------")
print(f"T = {T}")
print(f"v0 = {v0}")
print(f"Dr = {Dr}")
print(f"rho = {rho}")
#initial positions and velocities
def initialisation(N=484,rho=0.84):
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
    velocities = np.zeros_like(coordinates)
    return coordinates,velocities,L
'''def initialisation_random(N=225, rho=0.509):
    L = np.sqrt(N / rho)
    print(f"Box side length: {L}")
    coordinates = L * np.random.rand(N, 2)
    angles = 2 * np.pi * np.random.rand(N)
    e = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # No need to initialize velocities for overdamped dynamics
    velocities = np.zeros_like(coordinates) 
    
    return coordinates, velocities, L, e'''
#coordinates, velocities, L, e = initialisation_random(N, rho)
coordinates,velocities,L=initialisation(N,rho)
plt.plot(coordinates[:,0],coordinates[:,1],'o')
T_series = []
U_series=[]
test_coordinates=0
test_velocities=0
plot_cords=[]
def integrate(coordinates,velocities,time=10000):
    x=coordinates
    v=velocities
    #v_norm=np.linalg.norm(v, axis=1,keepdims=True)
    orientation=np.random.rand(N, 2)
    norm=np.linalg.norm(orientation, axis=1,keepdims=True)
    orientation=orientation/norm
    for i in range(time):
        h=t
        eta=np.random.normal(0,1,(N,2))
        #old eeta
        theta=np.arctan2(orientation[:,1],orientation[:,0])
        theta_new=theta+np.sqrt(2*Dr*h)*np.random.normal(0,1,N)
        orientation=np.column_stack((np.cos(theta_new),np.sin(theta_new)))
        coordinates1=x.flatten()
        mol=molsim.Simulation(L,r_c,N)
        mol.makegrid(coordinates1)
        f,U=mol.force2dhp(coordinates1)
        f=np.array(f).reshape(N,2)
        h=t
        #simple integrator
        x_n=x+(h/gamma)*f+v0*orientation*h+np.sqrt(2*D*h)*eta 
        x=x_n 
        x= x % L 
        if i%frame_rate==0 and i>=500:
            plot_cords.append(x)
        U_series.append(U/N)
        print("Step:", i, "Potential Energy:", (U/N))
integrate(coordinates,velocities,reps)
def animate(plot_cords,interval):
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
    ani=FuncAnimation(fig,update,frames=len(plot_cords),interval=100)
    plt.show()
animate(plot_cords,100)