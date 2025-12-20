#include "simulation.h" 
#include "lj.h"
#include "magforce.h"
#include "magtorque.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <set> 

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
void simulation::set_field(double Bx, double By) {
    B_x = Bx;
    B_y = By;
}
std::pair<std::vector<double>,std::vector<double>> simulation::force2dhp(const std::vector<double>& coordinates,const std::vector<double>& orientation) {
    double U = 0.0, r2, r6, r12, r_ij_x, r_ij_y, r, force_magnitude, lenard_force_x, lenard_force_y, lenard_potential,e_ix,e_iy,e_jx,e_jy,mag_force_x,mag_force_y;
    std::vector<double> forces_vec(2 * N, 0.0);
    std::vector<double> torques(N,0.0);
    double r_c2 = r_c * r_c;
    double r_c6_inv = 1.0 / (r_c2 * r_c2 * r_c2);
    double U_shift = 4.0 * (r_c6_inv * r_c6_inv - r_c6_inv);
    for (int i = 0; i < N; ++i) {
        double e_ix = orientation[2*i];
        double e_iy = orientation[2*i+1];
        for (int j = i + 1; j < N; ++j) { // Start j from i+1 to avoid double counting
            double e_jx = orientation[2*j];
            double e_jy = orientation[2*j+1];
            r_ij_x = coordinates[2 * i] - coordinates[2 * j];
            r_ij_y = coordinates[2 * i + 1] - coordinates[2 * j + 1];
            r_ij_x -= L * round(r_ij_x / L);
            r_ij_y -= L * round(r_ij_y / L);
            double r_sq = r_ij_x * r_ij_x + r_ij_y * r_ij_y;
            mag_force_x=0;
            mag_force_y=0;
            if (r_sq < r_c2){
                std::pair<double,double> mag_forces = mag_force(r_ij_x, r_ij_y, r_sq, e_ix, e_iy, e_jx, e_jy, nu);
                double magtorquei = torque(nu, r_ij_x, r_ij_y, e_ix, e_iy, e_jx, e_jy, r_sq);
                double magtorquej = torque(nu, -r_ij_x, -r_ij_y, e_jx, e_jy, e_ix, e_iy, r_sq);
                // Defensive: ensure computed values are finite
                if (!std::isfinite(magtorquei)) magtorquei = 0.0;
                if (!std::isfinite(magtorquej)) magtorquej = 0.0;
                if (!std::isfinite(mag_forces.first)) mag_forces.first = 0.0;
                if (!std::isfinite(mag_forces.second)) mag_forces.second = 0.0;
                torques[i] += magtorquei;
                #pragma omp atomic
                torques[j] += magtorquej; 
                mag_force_x = mag_forces.first;
                mag_force_y = mag_forces.second; 
            }
            double lenard_force_x = 0.0;
            double lenard_force_y = 0.0;
            if (r_sq < 1.25992104989){ // The WCA (repulsive-only) part
                std::pair<double,double> result=lj(r_sq,r_c2,U_shift);
                force_magnitude=result.first;   
                /*double r6_inv = 1.0 / (r_sq * r_sq * r_sq);
                double r12_inv = r6_inv * r6_inv;
                force_magnitude = (48.0 / r_sq) * (r12_inv - 0.5 * r6_inv);*/
                lenard_force_x = r_ij_x * force_magnitude;
                lenard_force_y = r_ij_y * force_magnitude;
            }
            forces_vec[2 * i]     += lenard_force_x + mag_force_x;
            forces_vec[2 * i + 1] += lenard_force_y + mag_force_y;
            #pragma omp atomic
            forces_vec[2 * j]     -= lenard_force_x + mag_force_x;
            #pragma omp atomic
            forces_vec[2 * j + 1] -= lenard_force_y + mag_force_y;
        } 
        torques[i] += nu * (e_ix * B_y - e_iy * B_x);
    }
    return std::make_pair(forces_vec, torques);
}
double simulation::phip(const std::vector<double>& coordinates,const std::vector<double>& orientation,double del_c){
    double del_c_sq=del_c*del_c;
    std::set<int> bonded_particles;
    for (int i=0;i<N;++i){
        double e_ix=orientation[2*i];
        double e_iy=orientation[2*i+1];
        for (int j=i+1;j<N;++j){    
            double e_jx=orientation[2*j];
            double e_jy=orientation[2*j+1];
            double r_ij_x = coordinates[2 * i] - coordinates[2 * j];
            double r_ij_y = coordinates[2 * i + 1] - coordinates[2 * j + 1];
            r_ij_x -= L * round(r_ij_x / L);
            r_ij_y -= L * round(r_ij_y / L);
            double r_sq = r_ij_x * r_ij_x + r_ij_y * r_ij_y; 
            if (r_sq>del_c_sq || r_sq < 1e-12){//first condition
                continue;
            }
            if (e_ix*e_jx+e_iy*e_jy>0){//second condition
                double r_ix=r_ij_x/pow(r_sq,0.5);
                double r_iy=r_ij_y/pow(r_sq,0.5);
                //if ((e_ix*r_iy-e_iy*r_ix)*(e_jx*r_iy-e_jy*r_ix)>0){
                //    bonded_particles.insert(i);
                //    bonded_particles.insert(j);
                double dot_i = e_ix * r_ij_x + e_iy * r_ij_y;
                double dot_j = e_jx * r_ij_x + e_jy * r_ij_y;

                if (dot_i * dot_j > 0) {
                    // All 3 criteria passed
                    bonded_particles.insert(i);
                    bonded_particles.insert(j);
                }
            }
        }
    }
    return static_cast<double>(bonded_particles.size()) / N;
}
