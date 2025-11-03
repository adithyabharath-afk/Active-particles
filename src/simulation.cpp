#include "simulation.h" 
#include "lj.h"
#include "magforce.h"
#include "magtorque.h"
#include <cmath>
#include <vector>
#include <iostream> 

simulation::simulation(double Length, double cutoff, int num_particles,double mag_moment,double mag_field_x,double mag_field_y) {
    L = Length;
    r_c = cutoff;
    N = num_particles;
    nx = 0;
    ny = 0;
    nu=mag_moment;
    B_x=mag_field_x;
    B_y=mag_field_y;
}
void simulation::makegrid(const std::vector<double>& coordinates) {
    nx = static_cast<int>(floor(L / r_c));
    ny = nx;
    grid.assign(nx, std::vector<std::vector<int>>(ny));

    for (int i = 0; i < N; ++i) {
        double x = coordinates[2 * i];
        double y = coordinates[2 * i + 1];
        int cellx = static_cast<int>(x / r_c);
        int celly = static_cast<int>(y / r_c);
        if (cellx >= nx) cellx = nx - 1;
        if (celly >= ny) celly = ny - 1;
        if (cellx < 0) cellx = 0; 
        if (celly < 0) celly = 0; 
        grid[cellx][celly].push_back(i);
    }
}
std::pair<std::vector<double>,std::vector<double>> simulation::force2dhp(const std::vector<double>& coordinates,const std::vector<double>& orientation) {
    double U = 0.0, r2, r6, r12, r_ij_x, r_ij_y, r, force_magnitude, lenard_force_x, lenard_force_y, lenard_potential,e_ix,e_iy,e_jx,e_jy,mag_force_x,mag_force_y;
    std::vector<double> forces_vec(2 * N, 0.0);
    std::vector<double> torques(N,0.0);
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
                        e_ix=orientation[2*i];
                        e_iy=orientation[2*i+1];
                        for (int j : grid[cnx][cny]) {    
                            if (i >= j) {
                                continue;
                            }       
                            e_jx=orientation[2*j];
                            e_jy=orientation[2*j+1];
                            r_ij_x = coordinates[2 * i] - coordinates[2 * j];
                            r_ij_y = coordinates[2 * i + 1] - coordinates[2 * j + 1];
                            r_ij_x -= L * round(r_ij_x / L);
                            r_ij_y -= L * round(r_ij_y / L);
                            double r_sq = r_ij_x * r_ij_x + r_ij_y * r_ij_y;                  
                            if (r_sq > r_c2) continue;
                            std::pair<double,double> lj_results=lj(r_sq,r_c2,U_shift); 
                            std::pair<double,double> mag_forces=mag_force(r_ij_x,r_ij_y,r_sq,e_ix,e_iy,e_jx,e_jy,nu);
                            double magtorquei=torque(nu,r_ij_x,r_ij_y,e_ix,e_iy,e_jx,e_jy,r_sq);
                            double magtorquej=torque(nu,r_ij_x,r_ij_y,e_jx,e_jy,e_ix,e_iy,r_sq);
                            torques[i]+=magtorquei;
                            torques[j]+=magtorquej; 
                            mag_force_x=mag_forces.first;
                            mag_force_y=mag_forces.second;                         
                            force_magnitude=lj_results.first;
                            lenard_potential=lj_results.second;
                            lenard_force_x = r_ij_x * force_magnitude;
                            lenard_force_y = r_ij_y * force_magnitude;
                            forces_vec[2 * i] += lenard_force_x+mag_force_x;
                            forces_vec[2 * i + 1] += lenard_force_y+mag_force_y;
                            forces_vec[2 * j] -= lenard_force_x+mag_force_x;
                            forces_vec[2 * j + 1] -= lenard_force_y+mag_force_y;
                            U += lenard_potential;
                        }
                        torques[i]+=nu*(e_ix*B_y-e_iy*B_x);
                    }
                }
            }
        }
    }
    return std::make_pair(forces_vec, torques);
}
