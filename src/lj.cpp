#include "lj.h"
#include <cmath>
#include <vector>
#include <iostream> 
std::pair<double,double> lj(double r_sq,double r_c2,double U_shift){
    if (r_sq > r_c2){
        return std::make_pair(0.0,0.0);
    }
    double r2 = 1.0 / r_sq;
    double r6 = r2 * r2 * r2;
    double r12 = r6 * r6;
    double force_magnitude = (48.0 * r12 - 24.0 * r6) * r2;
    double lenard_potential=4.0 * (r12 - r6) - U_shift;
    return std::make_pair(force_magnitude,lenard_potential);
}