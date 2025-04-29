#ifndef KRUH_H
#define KRUH_H
#include "Tvar.h"
class Kruh : public Tvar
{
    public:
        double r; double x; double y;
        Kruh(double rr, double xx, double yy);

        double obsah() const;
        virtual bool leziUvnitr(double xb, double yb) const;

        virtual double x_min() const{
            return -r;
            };

        virtual double x_max() const{
            return r;
            };

        virtual double y_min() const{
            return -r;
            };

        virtual double y_max() const{
            return r;
            };
        };
#endif