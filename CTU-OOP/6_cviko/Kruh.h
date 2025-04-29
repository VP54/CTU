#include <string>
#include <fstream>
#include "Tvar.h"
using namespace std;

class Kruh : public Tvar
{

public:
    double r;
    double x;
    double y;
    
    Kruh(double rr, double xx, double yy);

    virtual double obsah();

};

