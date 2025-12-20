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
reps =50000  # Number of time steps
N = 225# Number of particles# Cutoff radius for the Lennard-Jones potential
frame_rate=200
nu=1
# Magnetic field amplitudes (will vary in time)
# B_x0: x-component amplitude (sin), B_y0: y-component amplitude (cos)
B_x0 = 5
B_y0 = 5
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
def integrate(coordinates,velocities,time=10000):
    '''print("Starting integration(ist order)...")
    x=coordinates
    v=velocities
    #v_norm=np.linalg.norm(v, axis=1,keepdims=True)
    orientation=np.random.rand(N, 2)-0.5
    norm=np.linalg.norm(orientation, axis=1,keepdims=True)
    orientation=orientation/norm
    print("Creating simulation object...")
    mol=molsim.Simulation(L,r_c,N,nu,B_x,B_y)
    print("Simulation object created successfully")
    for i in range(time):
        coordinates1=x.flatten() 
        orientation1=orientation.flatten()     
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
        x= x % L'''
    print("Starting integration (2D Heun Method)...")
    x = coordinates
    
    # Initialize orientation as an ANGLE array (more efficient)
    theta = np.random.rand(N) * 2.0 * np.pi
    
    # Pre-calculate noise strengths for efficiency
    sqrt_2DT_dt = np.sqrt(2 * D * t)
    sqrt_2DR_dt = np.sqrt(2 * Dr* t)

    print("Creating simulation object... (will update B_x each step)")
    # create initial simulation object; we'll recreate it each step with the current B_x
    mol = molsim.Simulation(L, r_c, N, nu, B_x0, B_y0)
    print("Simulation object created successfully")
    
    for i in range(time):
        
        # --- 1. CALCULATE INITIAL DRIFTS & NOISE ---
        
        # Get orientation vector (only when needed for force calculation)
        orientation_t = np.column_stack((np.cos(theta), np.sin(theta)))
        
        # Calculate forces/torques at time (t)
        # Update magnetic field B_x sinusoidally in time
        current_time = i * t
        B_x_t = B_x0 * np.sin(omega * current_time)
        B_y_t = B_y0 * np.cos(omega * current_time)
        mol.set_field(B_x_t, B_y_t)

        coordinates1 = x.flatten()
        orientation1 = orientation_t.flatten()
        f_t, torque_t = mol.force2dhp(coordinates1, orientation1)
        
        # Reshape
        f_t = np.array(f_t).reshape(N, 2)
        torque_t = np.array(torque_t)
        
        # Store deterministic drifts at time (t)
        fx_t = v0 * orientation_t + (1.0 / gamma) * f_t
        ftheta_t = (1.0 / gamma_r) * torque_t
        
        # Generate random numbers ONCE per step
        G_T = np.random.normal(0, 1, (N, 2))
        G_R = np.random.normal(0, 1, N)
        
        # Calculate total noise terms (W)
        W_T = sqrt_2DT_dt * G_T
        W_R = sqrt_2DR_dt * G_R
        
        # --- 2. PREDICTOR STEP ---
        
        # Predict state at (t + dt)
        x_pred = x + fx_t * t + W_T
        theta_pred = theta + ftheta_t * t + W_R
        
        # Apply periodic boundaries to predicted position
        x_pred = x_pred % L
        
        # --- 3. CALCULATE PREDICTED DRIFTS ---
        
        # Get predicted orientation vector
        orientation_pred = np.column_stack((np.cos(theta_pred), np.sin(theta_pred)))
        
        # Calculate forces/torques at the PREDICTED state
        coordinates1_pred = x_pred.flatten()
        orientation1_pred = orientation_pred.flatten()
        f_pred, torque_pred = mol.force2dhp(coordinates1_pred, orientation1_pred)
        
        # Reshape
        f_pred = np.array(f_pred).reshape(N, 2)
        torque_pred = np.array(torque_pred)
        
        # Store deterministic drifts at predicted state
        fx_pred = v0 * orientation_pred + (1.0 / gamma) * f_pred
        ftheta_pred = (1.0 / gamma_r) * torque_pred
        
        # --- 4. CORRECTOR STEP (FINAL UPDATE) ---
        
        # Update state from (t) using the AVERAGE of the two drifts
        x = x + 0.5 * (fx_t + fx_pred) * t + W_T
        theta = theta + 0.5 * (ftheta_t + ftheta_pred) * t + W_R
        
        # Apply periodic boundary conditions to the final position
        x = x % L
        
        # (Optional) Wrap theta to [-pi, pi] for numerical stability
        # theta = np.arctan2(np.sin(theta), np.cos(theta))
        if i%frame_rate==0 and i>=500:
            phi_p =mol.phip(coordinates1,orientation1,del_c)
            phi_c=phic(np.array(coordinates1).reshape(N,2),L,N,del_c)
            print(f"Simulation step {i} / {time},phi_p={phi_p},phi_c={phi_c} (Heun)")
            plot_cords.append(x)
            # store particle orientations (angles) for coloring in the animation
            plot_thetas.append(theta.copy())
            phi_p_frames.append(phi_p)
            phi_c_frames.append(phi_c)
        #print(i)
        #U_series.append(U/N)
        #print("Step:", i, "Potential Energy:", (U/N))
def integrate_rk3(coordinates, velocities, time=10000):
    """
    Integrates the particle system using the 3rd-Order Runge-Kutta (SRK3) method.
    WARNING: This requires 3 force calculations per step and is 3x as slow as Euler.
    """
    print("Starting integration (2D RK3 Method)...")
    x = coordinates.copy() # Use a copy to avoid modifying the original array
    
    # Initialize orientation as an ANGLE array
    theta = np.random.rand(N) * 2.0 * np.pi
    
    # Pre-calculate noise strengths and time steps
    sqrt_2DT_dt = np.sqrt(2 * D * t)
    sqrt_2DR_dt = np.sqrt(2 * Dr * t)
    t_2 = t / 2.0 # half time step
    
    print("Creating simulation object...")
    mol = molsim.Simulation(L, r_c, N, nu, B_x0, B_y)
    print("Simulation object created successfully")
    
    for i in range(time):
        
        # --- 0. GENERATE NOISE ONCE ---
        # Generate N(0,1) random numbers ONCE per step
        G_T = np.random.normal(0, 1, (N, 2))
        G_R = np.random.normal(0, 1, N)
        
        # Calculate total noise terms (W) for the full step t
        W_T = sqrt_2DT_dt * G_T
        W_R = sqrt_2DR_dt * G_R

        # --- 1. STAGE 1 (k1) ---
        # Get drifts at the starting position (x, theta)
        # compute current time for B_x(t)
        current_time = i * t
        # set B_x appropriate for this stage and update mol
        B_x_t = B_x0 * np.sin(omega * current_time)
        B_y_t = B_y0 * np.cos(omega * current_time)
        mol.set_field(B_x_t, B_y_t)
        orientation_1 = np.column_stack((np.cos(theta), np.sin(theta)))
        coordinates1_1 = x.flatten()
        orientation1_1 = orientation_1.flatten()
        
        f_1, torque_1 = mol.force2dhp(coordinates1_1, orientation1_1)
        f_1 = np.array(f_1).reshape(N, 2)
        torque_1 = np.array(torque_1)
        
        # Drifts at state 1
        fx_1 = v0 * orientation_1 + (1.0 / gamma) * f_1
        ftheta_1 = (1.0 / gamma_r) * torque_1
        
        # --- 2. STAGE 2 (k2) ---
        # Predict state at the mid-point (t + t/2)
        # We use noise scaled by sqrt(0.5) for the t/2 step
        x_pred_2 = x + fx_1 * t_2 + W_T * np.sqrt(0.5) 
        theta_pred_2 = theta + ftheta_1 * t_2 + W_R * np.sqrt(0.5)
        x_pred_2 = x_pred_2 % L
        
        # Get drifts at the predicted mid-point
        orientation_2 = np.column_stack((np.cos(theta_pred_2), np.sin(theta_pred_2)))
        coordinates1_2 = x_pred_2.flatten()
        orientation1_2 = orientation_2.flatten()
        
        # For the mid-point (t + t/2), use B_x at current_time + t/2 and update mol
        B_x_t_mid = B_x0 * np.sin(omega * (current_time + t_2))
        B_y_t_mid = B_y0 * np.cos(omega * (current_time + t_2))
        mol.set_field(B_x_t_mid, B_y_t_mid)
        f_2, torque_2 = mol.force2dhp(coordinates1_2, orientation1_2)
        f_2 = np.array(f_2).reshape(N, 2)
        torque_2 = np.array(torque_2)
        
        # Drifts at state 2
        fx_2 = v0 * orientation_2 + (1.0 / gamma) * f_2
        ftheta_2 = (1.0 / gamma_r) * torque_2
        
        # --- 3. STAGE 3 (k3) ---
        # Predict state at the end-point (t + t) using k1 and k2
        x_pred_3 = x - fx_1 * t + 2.0 * fx_2 * t
        theta_pred_3 = theta - ftheta_1 * t + 2.0 * ftheta_2 * t
        x_pred_3 = x_pred_3 % L
        
        # Get drifts at the predicted end-point
        orientation_3 = np.column_stack((np.cos(theta_pred_3), np.sin(theta_pred_3)))
        coordinates1_3 = x_pred_3.flatten()
        orientation1_3 = orientation_3.flatten()
        
        # For the end-point (t + t), use B_x at current_time + t and update mol
        B_x_t_end = B_x0 * np.sin(omega * (current_time + t))
        B_y_t_end = B_y0 * np.cos(omega * (current_time + t))
        mol.set_field(B_x_t_end, B_y_t_end)
        f_3, torque_3 = mol.force2dhp(coordinates1_3, orientation1_3)
        f_3 = np.array(f_3).reshape(N, 2)
        torque_3 = np.array(torque_3)
        
        # Drifts at state 3
        fx_3 = v0 * orientation_3 + (1.0 / gamma) * f_3
        ftheta_3 = (1.0 / gamma_r) * torque_3

        # --- 4. FINAL STEP ---
        # Combine using the 1/6, 4/6, 1/6 weighted average
        fx_avg = (1./6.)*fx_1 + (4./6.)*fx_2 + (1./6.)*fx_3
        ftheta_avg = (1./6.)*ftheta_1 + (4./6.)*ftheta_2 + (1./6.)*ftheta_3
        
        # Update state from (t) using the AVERAGE drift and the ORIGINAL noise
        x = x + fx_avg * t + W_T
        theta = theta + ftheta_avg * t + W_R
        
        # Apply periodic boundary conditions
        x = x % L
        
        # --- 5. SAVING ---
        if i % frame_rate == 0 and i >= 500:
            # Calculate phi_p/c at the START of the step (state 1)
            phi_p = mol.phip(coordinates1_1, orientation1_1, del_c)
            phi_c = phic(np.array(coordinates1_1).reshape(N, 2), L, N, del_c)
            
            print(f"Simulation step {i} / {time} (RK3)")
            
            # Save the NEW state
            plot_cords.append(x.copy()) 
            plot_thetas.append(theta.copy())
            phi_p_frames.append(phi_p)
            phi_c_frames.append(phi_c)
integrate(coordinates,velocities,reps)
def animate(plot_cords,interval):
    fig=plt.figure()#creating a blank canvas
    ax=fig.add_subplot()#creates a specific plotting area in the canvas
    # create a colorbar mapping for orientation (angles in radians)
    #sm = plt.cm.ScalarMappable(cmap='hsv', norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
    #sm.set_array([])
    #cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    #cbar.set_label('orientation (rad)')
    
    def update(frame):
        cords=plot_cords[frame]       
        ax.clear()#erases everything on the ax plotting area,the points,the tilte , the dimentions everything       
        # color particles by orientation if available
        if frame < len(plot_thetas):
            thetas = plot_thetas[frame]
            sc = ax.scatter(cords[:,0], cords[:,1], c=thetas, cmap='hsv', vmin=-np.pi, vmax=np.pi, s=10)
        else:
            sc = ax.scatter(cords[:,0],cords[:,1],s=10)
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
        ax.set_title(" | ".join(title_parts))
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
