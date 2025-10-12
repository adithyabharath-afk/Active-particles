import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import molsim
#reduced parameters(sigma=1,epsilon=1,m=1)
T=0.85  # Temperature in reduced units
t=0.005  # Time step in reduced units
rho=0.840
N=500
r_c=3  # Cutoff radius for the Lennard-Jones potential
gamma=100  # Friction coefficient for Langevin thermostat
sigma=np.sqrt(2*gamma*T)  # Noise strength for Langevin thermostat
#initial positions and velocities
def initialisation(N=500,rho=0.84):#N=4n^3 ,N-no of particles,n-number of particles in one direction, rho-density
    n=np.round((N/4)**(1/3))
    if 4*n**3 != N:
        raise ValueError("N must be a multiple of 4*n^2 for a simple cubic lattice.")
    #The length of the box to accomodate the denisty and N
    L = (N / rho) ** (1/3)
    a=L/n
    #generate coordinates for fcc lattice
    basis=[[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
    coordinates = []
    for i in range(int(n)):
        for j in range(int(n)):
            for k in range(int(n)):
                origin=a*np.array([i,j,k])
                for b in basis:
                    coordinates.append(origin + a * np.array(b))
    coordinates=np.array(coordinates)
    print(coordinates.shape)
    velocities=np.random.rand(*coordinates.shape)-0.5 
    avg_v=np.mean(velocities, axis=0)
    velocities=velocities-avg_v
    avg_v2=np.mean(np.linalg.norm(velocities, axis=1)**2)
    scaling_factor=np.sqrt(3*T/(avg_v2))
    velocities=(velocities)*scaling_factor
    e=np.random.rand(*coordinates.shape)-0.5
    norm=np.linalg.norm(e, axis=1,keepdims=True)
    e=e/norm
    return coordinates,velocities,L,avg_v,avg_v2,scaling_factor,e
coordinates,velocities,L,avg_v,avg_v2,scaling_factor,e=initialisation(N,rho)
coordinate_last=coordinates-velocities*t
@njit
def force(coordinates,box_length,N):
    forces= np.zeros_like(coordinates)
    U=0
    for i in range(N):
        for j in range(i+1, N):
            r_ij = coordinates[i] - coordinates[j]
            r_ij=r_ij-box_length*np.round(r_ij/box_length)#periodic boundary conditions
            r = np.linalg.norm(r_ij)
            if r>r_c:
               continue    
            lenard_force= 48.0*r_ij/ r**14 - 24.0*r_ij/ r**8 
            lenard_potential = 4 * (r**-12 - r**-6)
            U+=lenard_potential#-4*(r_c**-12 - r_c**-6)  #shifted potential
            forces[i]+=lenard_force
            forces[j]-=lenard_force
    return forces,U
def test_force(coordinates):
    return -1*coordinates
T_series = []
U_series=[]
test_coordinates=0
test_velocities=0
run_molsim=True
run_test=False
run_NVE=False
run_langevin=True
def integrate(coordinates,velocities,time=10000):
    x=coordinates
    v=velocities
    for i in range(time):
        if run_NVE==True and run_molsim==True:#velocity_varlet algorithm
            coordinates1=coordinates.flatten()
            f=molsim.forces(coordinates1,L,N,r_c)[0]
            f=np.array(f).reshape(N,3)
            v_half=velocities+0.5*f*t
            x_n=coordinates+v_half*t
            x_n_1=x_n.flatten()
            F,U=molsim.forces(x_n_1,L,N,r_c)
            F=np.array(F).reshape(N,3)
            v_n=v_half+0.5*F*t
        if run_NVE==True and run_molsim==False:
            f=force(coordinates,L,N)[0]
            v_half=velocities+0.5*f*t
            x_n=coordinates+v_half*t
            F,U=force(x_n,L,N)
            v_n=v_half+0.5*F*t
        if run_molsim==True and run_langevin==True and run_test==False:
            eta=np.random.normal(0,1,(N,3))
            xi=np.random.normal(0,1,(N,3))
            coordinates1=coordinates.flatten()
            f=molsim.forces(coordinates1,L,N,r_c)[0]
            f=np.array(f).reshape(N,3)
            h=t
            sqrt_h=np.sqrt(h)
            #white noise
            v_half = (
            v
            + 0.5 * h * f                    # 1/2 h f(x^n)
            - 0.5 * h * gamma * v               # -1/2 h γ v^n
            + 0.5 * sqrt_h * sigma * xi         # +1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (f - gamma * v)   # -1/8 h^2 γ (f(x^n) - γ v^n)
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta))
            x_n= (
            x
            + h * v_half                        # + h v^{n+1/2}
            + h**1.5 * sigma * (1/(2*np.sqrt(3))) * eta)
            x_n_1=x_n.flatten()
            F,U=molsim.forces(x_n_1,L,N,r_c)
            #print(U)
            #print(F)
            F=np.array(F).reshape(N,3)
            #print(F) 
            v_n = (
            v_half
            + 0.5 * h * F                # + 1/2 h f(x^{n+1})
            - 0.5 * h * gamma * v_half          # - 1/2 h γ v^{n+1/2}
            + 0.5 * sqrt_h * sigma * xi         # + 1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (F - gamma * v_half)  # -1/8 h^2 γ (f(x^{n+1}) - γ v^{n+1/2})
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta))              
        #use the below code if you want to use python force function instead of the c++ one(its going to be like 10 times slower)
        if run_molsim==False and run_test==False and run_langevin==True:
            eta=np.random.normal(0,1,(N,3))
            xi=np.random.normal(0,1,(N,3))
            f=force(coordinates,L,N)[0]
            h=t 
            sqrt_h=np.sqrt(h)
            v_half = (
            v
            + 0.5 * h * f                    # 1/2 h f(x^n)
            - 0.5 * h * gamma * v               # -1/2 h γ v^n
            + 0.5 * sqrt_h * sigma * xi         # +1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (f - gamma * v)   # -1/8 h^2 γ (f(x^n) - γ v^n)
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta))
            x_n= (
            x
            + h * v_half                        # + h v^{n+1/2}
            + h**1.5 * sigma * (1/(2*np.sqrt(3))) * eta)
            
            F,U=force(x_n,L,N)
            v_n = (
            v_half
            + 0.5 * h * F                # + 1/2 h f(x^{n+1})
            - 0.5 * h * gamma * v_half          # - 1/2 h γ v^{n+1/2}
            + 0.5 * sqrt_h * sigma * xi         # + 1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma * (F - gamma * v_half)  # -1/8 h^2 γ (f(x^{n+1}) - γ v^{n+1/2})
            - 0.25 * h**1.5 * gamma * sigma * (0.5 * xi + (1/np.sqrt(3)) * eta))
            #print(F)
            #print(U)
        if run_test==True:
            gamma1 = 1.0             # friction coefficient   
            f = lambda x: -x
            sigma1=np.sqrt(2)
            h=t
            sqrt_h=np.sqrt(h)
            eta=np.random.normal(0,1)
            xi=np.random.normal(0,1)            
            v_half = (
            v
            + 0.5 * h * f(x)                    # 1/2 h f(x^n)
            - 0.5 * h * gamma1 * v               # -1/2 h γ v^n
            + 0.5 * sqrt_h * sigma1 * xi         # +1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma1 * (f(x) - gamma1 * v)   # -1/8 h^2 γ (f(x^n) - γ v^n)
            - 0.25 * h**1.5 * gamma1 * sigma1 * (0.5 * xi + (1/np.sqrt(3)) * eta))
            x_n= (
            x
            + h * v_half                        # + h v^{n+1/2}
            + h**1.5 * sigma1 * (1/(2*np.sqrt(3))) * eta)
            v_n = (
            v_half
            + 0.5 * h * f(x_n)                # + 1/2 h f(x^{n+1})
            - 0.5 * h * gamma1 * v_half          # - 1/2 h γ v^{n+1/2}
            + 0.5 * sqrt_h * sigma1 * xi         # + 1/2 sqrt(h) σ ξ^n
            - (1/8) * h**2 * gamma1 * (f(x_n) - gamma1 * v_half)  # -1/8 h^2 γ (f(x^{n+1}) - γ v^{n+1/2})
            - 0.25 * h**1.5 * gamma1* sigma1 * (0.5 * xi + (1/np.sqrt(3)) * eta))
            x=x_n
            v=v_n           
        #the below if part is not needed if you want to use langevin thermostat
        #use it only for simple NVE molecular dynamics
        if i<5000 and run_NVE==True:
            avg_v2=np.mean(np.linalg.norm(v_n, axis=1)**2)
            scaling_factor=np.sqrt(3*T/avg_v2)
            v_n=v_n*scaling_factor
        if i>6000 and run_test==False:
            T_m=np.mean(np.linalg.norm(v_n, axis=1)**2)/3
            T_series.append(T_m)
        coordinates=x_n
        velocities=v_n
        if run_test==False:
            U+=N*(8 * np.pi * rho / 3.0) * (1.0/(3.0 * r_c**9) - 1.0/(r_c**3))  # tail correction
            total_energy=U+0.5*np.sum(v_n**2)
            U_series.append(U/N)
            print("Step:", i, "Total Energy:", total_energy, "Temperature:",np.mean(np.linalg.norm(v_n, axis=1)**2)/3, "Potential Energy:", (U/N))
    if run_test==True:
        return coordinates, velocities
integrate(coordinates,velocities)
if run_test==True:
    num=100000
    E=0.0
    for i in range(num):
        print(i)
        test_coordinates,test_velocities=0,0
        x,v=integrate(test_coordinates,test_velocities,time=200)
        E+=x**2+v**2
    print("Average energy after {} runs is {}".format(num,E/(num)))
if run_test==False:
    print("Average potential energy per particle:",np.mean(U_series))
    print("Average temperature:",np.mean(T_series))
    plt.plot(T_series)
    plt.xlabel('Time Steps')    
    plt.ylabel('Temperature')
    plt.title('Temperature vs Time Steps')
    plt.show()


           
 


