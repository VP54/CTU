#include <string>
#include <fstream>
#include "Tvar.h"
using namespace std;

class Kruh : public Tvar
{
//, double x_min, double x_max, double y_min, double y_max
public:
    double r;
    double x;
    double y;
    
    Kruh(double r, double x, double y);

    double obsah();
    
    double Tvar::x_min();
    double Tvar::x_max();
    double Tvar::y_min();
    double Tvar::y_max();

};