#include <cmath>
#include "Kruh.h"
#include <iostream>
using namespace std;

Kruh::Kruh(double rr, double xx, double yy){
	r=rr;
	x=xx;
	y=yy;
}

double Kruh::obsah() const {

	return 3.14 * r * r;
}

bool Kruh::leziUvnitr(double rand_x, double rand_y)const {
	return (rand_x * rand_x + rand_y * rand_y <= r*r);
}