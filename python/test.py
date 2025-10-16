import numpy as np
import molsim # This is your C++ module!

# --- Simulation Parameters ---
N = 3000#mber of particles
L = 10.0     # Box length
r_c = 2.5    # Cutoff radius

# --- 1. Generate some random particle coordinates ---
# Using NumPy for convenience
np.random.seed(42) # for reproducible results
coordinates = L * np.random.rand(N, 2)
# Flatten the array to a simple list of doubles, which our C++ function expects
coordinates_list = coordinates.flatten().tolist()

print(f"Testing with {N} particles in a {L}x{L} box.")
print("-" * 30)


# --- 2. Test the simple 'force2d' function ---
print("Testing the simple force2d function...")
try:
    forces, potential = molsim.force2d(coordinates_list, L, N, r_c)
    print(f"  -> Success!")
    print(f"  -> Calculated Potential Energy: {potential:.4f}")
    # print(f"  -> First few force components: {forces[:6]}") # Uncomment to see forces
except Exception as e:
    print(f"  -> An error occurred: {e}")

print("-" * 30)


# --- 3. Test the high-performance 'Simulation' class ---
print("Testing the optimized Simulation class...")
try:
    # Create an instance of our C++ class from Python!
    sim = molsim.Simulation(length=L, cutoff=r_c, num_particles=N)
    print("  -> C++ Simulation object created successfully.")

    # Use the class methods
    sim.makegrid(coordinates_list)
    print("  -> makegrid() called successfully.")

    hp_forces, hp_potential = sim.force2dhp(coordinates_list)
    print("  -> force2dhp() called successfully.")
    print(f"  -> Calculated Potential Energy (HP): {hp_potential:.4f}")
    
    # Let's check if the results are close (they should be!)
    if abs(potential - hp_potential) < 1e-9:
        print("\n[SUCCESS] Both methods produced nearly identical potential energies!")
    else:
        print("\n[WARNING] Potential energies from the two methods are different.")

except Exception as e:
    print(f"  -> An error occurred with the Simulation class: {e}")