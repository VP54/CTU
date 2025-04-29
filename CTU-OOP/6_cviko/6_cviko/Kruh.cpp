#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include "Kruh.h"
#include "Tvar.h"
using namespace std;


//double _x_min, double _x_max, double _y_min, double _y_max, 
//x_min, x_max, y_min, y_max

Kruh::Kruh(double r, double x, double y) : Tvar() {
    r = r;
    x = x;
    y = y;
}

double Kruh::obsah(){
    // Pomoci Monte Carla
    double rand_x, rand_y, obsah;
    int in, out, range;

    range = int(r) * 2 * 100;
    in = 0;
    out = 0;


    for (int i = 0; i <1000000; i++){
        rand_x = (rand() % range + (-r*100));
        rand_x = rand_x / 100;
        rand_y = (rand() % range + (-r*100));
        rand_y = rand_y / 100;
       if(rand_x * rand_x + rand_y * rand_y <= r*r){
        in +=1;
       }
       else{
        out += 1;
       }
    }
    obsah = 4.0 * in / (in + out);

    std::cout << "obsah: " << obsah << "\t\t\t" << in + out << "\n";

    return obsah;
}