#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <cfenv>

// Forward declaration of force2d (implemented in force2d.cpp)
std::pair<std::vector<double>, double> force2d(
    std::vector<double> coordinates, double L, int N, double r_c);

namespace py = pybind11;
std::pair<std::vector<double>,double> forces(std::vector<double> coordinates,double L,int N,double r_c){
    double U=0.0,r_ij_x,r_ij_y,r_ij_z,r,lenard_force_x,force_magnitude,lenard_force_y,lenard_force_z,lenard_potential;
    std::vector<double> forces(3*N,0.0);
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            r_ij_x=coordinates[3*i]-coordinates[3*j];
            r_ij_y=coordinates[3*i+1]-coordinates[3*j+1];
            r_ij_z=coordinates[3*i+2]-coordinates[3*j+2];
            //periodic boundary conditions
            r_ij_x-=L*round(r_ij_x/L);
            r_ij_y-=L*round(r_ij_y/L);  
            r_ij_z-=L*round(r_ij_z/L);
            r=sqrt(pow(r_ij_x,2)+pow(r_ij_y,2)+pow(r_ij_z,2));
            if (r>r_c) continue;
            force_magnitude=48.0*pow(r,-14)-24.0*pow(r,-8);
            lenard_force_x=r_ij_x*force_magnitude;
            lenard_force_y=r_ij_y*force_magnitude;
            lenard_force_z=r_ij_z*force_magnitude;
            forces[3*i]+=lenard_force_x;
            forces[3*i+1]+=lenard_force_y;
            forces[3*i+2]+=lenard_force_z;
            forces[3*j]-=lenard_force_x;
            forces[3*j+1]-=lenard_force_y;
            forces[3*j+2]-=lenard_force_z;
            lenard_potential=4.0*(1/(r*r*r*r*r*r*r*r*r*r*r*r)-1/(r*r*r*r*r*r));
            U=U+lenard_potential;
        }
    }
    return std::make_pair(forces,U); 
}
PYBIND11_MODULE(molsim, m) {
    m.doc() = "Module to compute Lennard-Jones forces and potential energy"; // Optional module docstring
    m.def("forces", &forces, "Compute Lennard-Jones forces and potential energy",
          py::arg("coordinates"), py::arg("L"), py::arg("N"), py::arg("r_c"));
    m.def("force2d", &force2d,"Compute 2D Lennard-Jones forces and potential energy",
          py::arg("coordinates"), py::arg("L"), py::arg("N"), py::arg("r_c"));
}
