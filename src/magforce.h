#ifndef MAGFORCE_H
#define MAGFORCE_H
#include <vector>
#include <utility> 
std::pair<double,double> mag_force(double r_x,double r_y,double r_sq,double e_ix,double e_iy,double e_jx,double e_jy,double nu);
#endif