#ifndef SIMULATION_H 
/*everytime a new .cpp file is compiled and if it has called 
simulation.h two times in its compiling c++ will give an
 error.This line acts as a bouncer . Thids bouncer have a checklist
 if the .h is called for the first time it lets the compiler
 enter otherwise it blocks the entire code till he endif.*/
#define SIMULATION_H
/*This line basically tells the bouncer to tick the name simulation.h*/
#include <vector>
#include <utility> // Required for std::pair

// Declaration of the simulation class. This is the "menu".
class simulation {
public:
    // Member variables
    double L, r_c,nu,B_x,B_y;
    int nx, ny, N;
    std::vector<std::vector<std::vector<int>>> grid;

    // Constructor declaration
    simulation(double Length, double cutoff, int num_particles,double mag_moment,double mag_field_x,double mag_field_y);

    // Method declaration
    std::pair<std::vector<double>,std::vector<double>> force2dhp(const std::vector<double>& coordinates,const std::vector<double>& orientation);
    double phip(const std::vector<double>& coordinates,const std::vector<double>& orientation,double del_c);
    // set external magnetic field components
    void set_field(double Bx, double By);
};

#endif // SIMULATION_H