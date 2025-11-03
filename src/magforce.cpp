#include <cmath>
#include <vector>
#include <iostream> 
#include "magforce.h"
std::pair<double,double> mag_force(double r_x,double r_y,double r_sq,double e_ix,double e_iy,double e_jx,double e_jy,double nu){
    double r4,f_x,f_y,i1,i2,i3,r,nu_sq;
    nu_sq=nu*nu;
    r4=1/(r_sq*r_sq);
    r=std::sqrt(r_sq);
    r_x=r_x/(r);
    r_y=r_y/(r);
    i1=e_ix*e_jx+e_iy*e_jy;
    i2=r_x*e_jx+r_y*e_jy;
    i3=r_x*e_ix+r_y*e_iy;
    f_x=3*r4*(nu_sq)*(r_x*(i1)+e_ix*(i2)+e_jx*(i3)-5*r_x*(i3)*(i2));
    f_y=3*r4*(nu_sq)*(r_y*(i1)+e_iy*(i2)+e_jy*(i3)-5*r_y*(i3)*(i2));  
    return std::make_pair(f_x,f_y);
}