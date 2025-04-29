
/*
#include <iostream>
#include <cmath>
#include <cmath>
using namespace std;

#ifndef KRUH_H
#define KRUH_H

class Kruh
{
    public:
        double r, x, y;

        void vypisInfo(double r , double x, double y){
            std::cout << "Polomer: \t" << r << "\n" << "Souradnice x: \t" << x << "\n" << "Souradnice y: \t" << y << "\n";
        }

        double obsah(double r){
            return 3.14 * r * r;
        }

        double obvod(double r){
            return 2 * 3.14 * r;
        }

        string odmocnina(double r1, double r2, double x1, double x2, double y1, double y2){
            double d;

            d = sqrt( pow(x2 - x1, 2) + pow( y2 - y1, 2) );
            if (r1 > (d + r2)){
                return "True";
            }
            else{
                return "False";
            }
        }
};



#endif KRUH_H
*/


#include <string>
#include <fstream>
using namespace std;

class Kruh
{

public:
    double r;
    double x;
    double y;
    
    Kruh(double rr, double xx, double yy);


    void vypisInfo();

    double obvod();

    double obsah();

    bool leziUvnitr(double x2, double y2);
};

