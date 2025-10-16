#ifndef FORCE_FUNCTIONS_H
#define FORCE_FUNCTIONS_H

#include <vector>
#include <utility> // Required for std::pair

// Declaration for the 3D forces function
std::pair<std::vector<double>, double> forces(
    const std::vector<double>& coordinates, double L, int N, double r_c);

// Declaration for the 2D forces function
std::pair<std::vector<double>, double> force2d(
    const std::vector<double>& coordinates, double L, int N, double r_c);

#endif // FORCE_FUNCTIONS_H