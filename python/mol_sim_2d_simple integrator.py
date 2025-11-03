import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import molsim
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
#reduced parameters(sigma=1,epsilon=1,m=1)
target_U = 1
target_Pe =20
sigma = 1.0  # Particle diameter
gamma = 1.0  # Friction coefficient
gamma_r=3*gamma
B_scale=10**(-5)
nu_scale=1/B_scale
T = 1.0 / target_U                      # Temperature
D = T / gamma                           # Translational diffusion coefficient
tau = (sigma**2) / D                    # Brownian time
v0 = (target_Pe * sigma) / tau          # Self-propulsion velocity
Dr = T/gamma_r             # Rotational diffusion coefficient
t = 0.0001                           # Time step
rho = 0.23 / (np.pi/4)  #papers density 0.4    
reps = 50000    
N =900 # Number of particles
r_c = 2.5  # Cutoff radius for the Lennard-Jones potential
frame_rate=100
nu=2
B_x=14
B_y=0
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
U_series=[]
test_coordinates=0
test_velocities=0
plot_cords=[]
def integrate(coordinates,velocities,time=10000):
    print("Starting integration...")
    x=coordinates
    v=velocities
    #v_norm=np.linalg.norm(v, axis=1,keepdims=True)
    orientation=np.random.rand(N, 2)
    norm=np.linalg.norm(orientation, axis=1,keepdims=True)
    orientation=orientation/norm
    print("Creating simulation object...")
    mol=molsim.Simulation(L,r_c,N,nu,B_x,B_y)
    print("Simulation object created successfully")
    for i in range(time):
        coordinates1=x.flatten() 
        orientation1=orientation.flatten()     
        mol.makegrid(coordinates1)
        result = mol.force2dhp(coordinates1,orientation1)  # Capture the return value in a single variable
        f, torque = result  # Unpack inside the same expression
        f=np.array(f).reshape(N,2)
        torque=np.array(torque)
        h=t
        eta=np.random.normal(0,1,(N,2))
        #old eeta
        theta=np.arctan2(orientation[:,1],orientation[:,0])
        theta_new=theta+torque*h/(gamma_r)+np.sqrt(2*Dr*h)*np.random.normal(0,1,N)
        orientation=np.column_stack((np.cos(theta_new),np.sin(theta_new)))      
        #sqrt_h = np.sqrt(h)
        #simple integrator
        x_n=x+(h/gamma)*f+v0*orientation*h+np.sqrt(2*D*h)*eta
        x=x_n   
        x= x % L
        if i%frame_rate==0 and i>=500:
            print(f"Simulation step {i} / {time}")
            plot_cords.append(x)
        #print(i)
        #U_series.append(U/N)
        #print("Step:", i, "Potential Energy:", (U/N))
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
        
    ani=FuncAnimation(fig,update,frames=len(plot_cords),interval=100)
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
