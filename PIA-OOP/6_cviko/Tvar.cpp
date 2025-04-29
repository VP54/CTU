#include "Tvar.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>


double Tvar::obsah() const {

	const double x_mi=x_min();
	const double x_ma=x_max();
	const double y_mi=y_min();
	const double y_ma=y_max();
	int n=0;

    double rand_x, rand_y, obsah;
    int in, out, range;


    range = int(x_ma) * 2 * 100;
    in = 0;
    out = 0;


    for (int i = 0; i <1000000; i++){
        rand_x = (rand() % range + (-x_ma*100));
        rand_x = rand_x / 100;
        rand_y = (rand() % range + (-x_ma*100));
        rand_y = rand_y / 100;
       if(leziUvnitr(rand_x, rand_y)){
        in +=1;
       }
       else{
        out += 1;
       }
    }
    obsah = 2 * x_ma * 2 * x_ma * in / (in + out);

	return obsah;

}