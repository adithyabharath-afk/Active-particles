import freud
import numpy as np

def phic(coordinates, L, N, delta_c):
    """
    Calculates the largest cluster fraction (phi_c)
    using freud to handle periodic boundary conditions.
    """
    
    # 1. Create the freud simulation box
    box = freud.box.Box(Lx=L, Ly=L, is2D=True)

    # 2. Pad coordinates to (N, 3)
    coords_3d = np.zeros((N, 3))
    coords_3d[:, :2] = coordinates  # This is your (N, 2) array

    # 3. Create the clustering object
    cl = freud.cluster.Cluster()

    # 4. Compute the clusters (This part is correct)
    cl.compute(system=(box, coords_3d), 
               neighbors={'r_max': delta_c, 'exclude_ii': True})
    
    # --- THIS IS THE FIX ---
    # 5. Get cluster sizes
    #    'cl.cluster_idx' is the per-particle array of cluster IDs
    if cl.num_clusters > 0:
        # 'key_counts' will be an array of the sizes of each cluster
        unique_keys, key_counts = np.unique(cl.cluster_idx, return_counts=True)
        
        # 6. Find the largest cluster
        n_c_star = np.max(key_counts)
        phi_c = n_c_star / N
    else:
        # This case happens if N=0 or no particles are passed
        phi_c = 0.0 
    # --- END FIX ---

    return phi_c