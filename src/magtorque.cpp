#include <cmath>
#include <vector>
#include <iostream> 
#include "magtorque.h"
double torque(double nu,double r_x,double r_y,double e_ix,double e_iy,double e_jx,double e_jy,double r_sq){
    double r3,t_z,r;
    r=std::sqrt(r_sq);
    r_x=r_x/(r);
    r_y=r_y/(r);
    r3=1/pow(r_sq,3/2);
    t_z=(nu*nu)*r3*(3*(e_ix*r_x+e_iy*r_y)*(e_jx*r_y-e_jy*r_x)+(e_ix*e_jy-e_iy*e_jx));
    return t_z;
}