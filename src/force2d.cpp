#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <cfenv>

namespace py = pybind11;
std::pair<std::vector<double>,double> force2d(std::vector<double> coordinates,double L,int N,double r_c){
    double U=0.0,r_ij_x,r_ij_y,r_ij_z,r,lenard_force_x,force_magnitude,lenard_force_y,lenard_force_z,lenard_potential;
    std::vector<double> forces(2*N,0.0);
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            r_ij_x=coordinates[2*i]-coordinates[2*j];
            r_ij_y=coordinates[2*i+1]-coordinates[2*j+1];
            //periodic boundary conditions
            r_ij_x-=L*round(r_ij_x/L);
            r_ij_y-=L*round(r_ij_y/L);  
            r=sqrt(pow(r_ij_x,2)+pow(r_ij_y,2));
            if (r>r_c) continue;
            force_magnitude=48.0*pow(r,-14)-24.0*pow(r,-8);
            lenard_force_x=r_ij_x*force_magnitude;
            lenard_force_y=r_ij_y*force_magnitude;
            forces[2*i]+=lenard_force_x;
            forces[2*i+1]+=lenard_force_y;
            forces[2*j]-=lenard_force_x;
            forces[2*j+1]-=lenard_force_y;
            lenard_potential=4.0*(1/(r*r*r*r*r*r*r*r*r*r*r*r)-1/(r*r*r*r*r*r));
            U=U+lenard_potential;
        }
    }
    return std::make_pair(forces,U); 
}

