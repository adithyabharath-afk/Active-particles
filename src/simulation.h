#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <utility> // Required for std::pair

// Declaration of the simulation class. This is the "menu".
class simulation {
public:
    // Member variables
    double L, r_c;
    int nx, ny, N;
    std::vector<std::vector<std::vector<int>>> grid;

    // Constructor declaration
    simulation(double Length, double cutoff, int num_particles);

    // Method declarations
    void makegrid(const std::vector<double>& coordinates);
    std::pair<std::vector<double>, double> force2dhp(const std::vector<double>& coordinates);
};

#endif // SIMULATION_H