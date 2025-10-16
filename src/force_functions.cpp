#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <cfenv>
#include "force_functions.h"
namespace py = pybind11;
std::pair<std::vector<double>,double> force2d(const std::vector<double> &coordinates,double L,int N,double r_c){
    double U=0.0,r2,r6,r12,r_ij_x,r_ij_y,r_ij_z,r,lenard_force_x,force_magnitude,lenard_force_y,lenard_force_z,lenard_potential;
    std::vector<double> forces(2*N,0.0);
    double r_c2 = r_c * r_c;
    double r_c6_inv = 1.0 / (r_c2 * r_c2 * r_c2);
    double U_shift = 4.0 * (r_c6_inv * r_c6_inv - r_c6_inv);
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            r_ij_x=coordinates[2*i]-coordinates[2*j];
            r_ij_y=coordinates[2*i+1]-coordinates[2*j+1];
            //periodic boundary conditions
            r_ij_x-=L*round(r_ij_x/L);
            r_ij_y-=L*round(r_ij_y/L);  
            r=sqrt(r_ij_x*r_ij_x+r_ij_y*r_ij_y);
            if (r>r_c) continue;
            r12=1/(r*r*r*r*r*r*r*r*r*r*r*r);
            r2=1/(r*r);
            r6=1/(r*r*r*r*r*r);
            force_magnitude=48.0*r12*r2-24.0*r6*r2;
            lenard_force_x=r_ij_x*force_magnitude;
            lenard_force_y=r_ij_y*force_magnitude;
            forces[2*i]+=lenard_force_x;
            forces[2*i+1]+=lenard_force_y;
            forces[2*j]-=lenard_force_x;
            forces[2*j+1]-=lenard_force_y;
            lenard_potential=4.0*(r12-r6)-U_shift;
            U=U+lenard_potential;
        }
    }
    return std::make_pair(forces,U); 
}
std::pair<std::vector<double>,double> forces(const std::vector<double>& coordinates,double L,int N,double r_c){
    double r6,r2,r12,U=0.0,r_ij_x,r_ij_y,r_ij_z,r,lenard_force_x,force_magnitude,lenard_force_y,lenard_force_z,lenard_potential;
    std::vector<double> forces(3*N,0.0);
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            r_ij_x=coordinates[3*i]-coordinates[3*j];
            r_ij_y=coordinates[3*i+1]-coordinates[3*j+1];
            r_ij_z=coordinates[3*i+2]-coordinates[3*j+2];
            //periodic boundary conditions
            r_ij_x-=L*round(r_ij_x/L);
            r_ij_y-=L*round(r_ij_y/L);  
            r_ij_z-=L*round(r_ij_z/L);
            r=sqrt(r_ij_x*r_ij_x+r_ij_y*r_ij_y+r_ij_z*r_ij_z);
            if (r>r_c) continue;
            r12=1/(r*r*r*r*r*r*r*r*r*r*r*r);
            r2=1/(r*r);
            r6=1/(r*r*r*r*r*r);
            force_magnitude=48.0*r12*r2-24.0*r6*r2;
            lenard_force_x=r_ij_x*force_magnitude;
            lenard_force_y=r_ij_y*force_magnitude;
            lenard_force_z=r_ij_z*force_magnitude;
            forces[3*i]+=lenard_force_x;
            forces[3*i+1]+=lenard_force_y;
            forces[3*i+2]+=lenard_force_z;
            forces[3*j]-=lenard_force_x;
            forces[3*j+1]-=lenard_force_y;
            forces[3*j+2]-=lenard_force_z;
            lenard_potential=4.0*(r12-r6);
            U=U+lenard_potential;
        }
    }
    return std::make_pair(forces,U); 
}






