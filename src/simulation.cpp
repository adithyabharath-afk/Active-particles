#include "simulation.h" // Include the header to ensure consistency
#include <cmath>
#include <vector>
#include <iostream> // Good for debugging if needed

// --- Important ---
// The class definition is GONE from this file. We are only IMPLEMENTING
// the functions that were DECLARED in simulation.h.

// Use the scope resolution operator '::' to tell the compiler these
// functions belong to the 'simulation' class.

// Constructor implementation. Note: num_particles is now an 'int' to match the header.
simulation::simulation(double Length, double cutoff, int num_particles) {
    L = Length;
    r_c = cutoff;
    N = num_particles;
    // It's good practice to initialize all member variables
    nx = 0;
    ny = 0;
}

// makegrid implementation. Note: 'coordinates' is now 'const' to match the header.
void simulation::makegrid(const std::vector<double>& coordinates) {
    nx = static_cast<int>(floor(L / r_c));
    ny = nx;
    grid.assign(nx, std::vector<std::vector<int>>(ny));

    for (int i = 0; i < N; ++i) {
        double x = coordinates[2 * i];
        double y = coordinates[2 * i + 1];
        int cellx = static_cast<int>(floor(x / r_c));
        int celly = static_cast<int>(floor(y / r_c));
        if (cellx >= 0 && cellx < nx && celly >= 0 && celly < ny) {
            grid[cellx][celly].push_back(i);
        }
    }
}

// force2dhp implementation. Note: 'coordinates' is now 'const' to match the header.
std::pair<std::vector<double>, double> simulation::force2dhp(const std::vector<double>& coordinates) {
    double U = 0.0, r2, r6, r12, r_ij_x, r_ij_y, r, force_magnitude, lenard_force_x, lenard_force_y, lenard_potential;
    std::vector<double> forces_vec(2 * N, 0.0); // Renamed to avoid shadowing
    double r_c2 = r_c * r_c;
    double r_c6_inv = 1.0 / (r_c2 * r_c2 * r_c2);
    double U_shift = 4.0 * (r_c6_inv * r_c6_inv - r_c6_inv);

    for (int cx = 0; cx < nx; ++cx) {
        for (int cy = 0; cy < ny; ++cy) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    int cnx = (cx + dx + nx) % nx;
                    int cny = (cy + dy + ny) % ny;
                    for (int i : grid[cx][cy]) {
                        for (int j : grid[cnx][cny]) {
                            if (i >= j) {
                                continue;
                            }
                            r_ij_x = coordinates[2 * i] - coordinates[2 * j];
                            r_ij_y = coordinates[2 * i + 1] - coordinates[2 * j + 1];
                            r_ij_x -= L * round(r_ij_x / L);
                            r_ij_y -= L * round(r_ij_y / L);
                            double r_sq = r_ij_x * r_ij_x + r_ij_y * r_ij_y;
                            if (r_sq > r_c2) continue;
                            
                            r = sqrt(r_sq);
                            r2 = 1.0 / r_sq;
                            r6 = r2 * r2 * r2;
                            r12 = r6 * r6;
                            force_magnitude = (48.0 * r12 - 24.0 * r6) * r2;
                            lenard_force_x = r_ij_x * force_magnitude;
                            lenard_force_y = r_ij_y * force_magnitude;
                            forces_vec[2 * i] += lenard_force_x;
                            forces_vec[2 * i + 1] += lenard_force_y;
                            forces_vec[2 * j] -= lenard_force_x;
                            forces_vec[2 * j + 1] -= lenard_force_y;
                            lenard_potential = 4.0 * (r12 - r6) - U_shift;
                            U += lenard_potential;
                        }
                    }
                }
            }
        }
    }
    return std::make_pair(forces_vec, U);
}
