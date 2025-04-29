#include <iostream>
#include <vector>
#include <cmath>
#include "Kruh.h"

Kruh::Kruh(double rr, double xx, double yy){
    r = rr;
    x = xx;
    y = yy;
}

void Kruh::vypisInfo(){
    std::cout << "Polomer: \t" << r << "\n" << "Souradnice x: \t" << x << "\n" << "Souradnice y: \t" << y << "\n";
}

double Kruh::obsah(){
    return 3.14 * pow(r, 2);
}

double Kruh::obvod(){
    return 2 * 3.14 * r;
}

bool Kruh::leziUvnitr(double x2, double y2){
    double d;

    d = sqrt( pow(x2 - x, 2) + pow( y2 - y, 2) );
    if (r > d){
        return true;
    }
    else{
        return false;
    }
}


/*
Kruh::Bod(double xx, double yy) {
x = xx;
y = yy;
}
void Kruh::posun(double xx, double yy) {
x += xx;
y += yy;
}
double Kruh::vzdalenost() const {
return std::sqrt(x*x + y*y);
}*/