#ifndef TVAR_H
#define TVAR_H

// Toto je tzv. abstraktni trida, ktera popisuje obecny tvar, jehoz obsah umime vypocitat pomoci
// numericke integrace. Abstraktni znamena, ze pouze predepisuje funkcnost pro odvozene tridy,
// ale samotny objekt tridy Tvar nelze vytvorit.
class Tvar {
    public:

	// Trida Tvar bude mit v souboru Tvar.cpp definovanou metodu pro vypocet obsahu obecneho
        // tvaru pomoci numericke integrace (napr. pomoci site nebo Monte Carlo).
        // Protoze je metoda virtualni, nebrani to odvozenym tridam, ktere si MOHOU tuto metodu predefinovat. 
        // Napr. trida Kruh muze vypocitat obsah pomoci vzorecku misto integrace a usetrime tim spoustu CPU casu.

        virtual double obsah() const;

        // Toto je abstraktni metoda, v teto tride ji neimplementujeme, bude zde pouze tato hlavicka.
        // Rikame tim ze kazda odvozena trida MUSI tuto metodu implementovat. Pokud totiz budeme 
        // chtit vypocitat obsah integraci, budeme tuto funkci vzdy potrebovat, at uz je odvozena trida jakakoli.

        //virtual bool leziUvnitr(double xx, double yy) const = 0; 

    protected:

        // Dale budeme po odvozene tride chtit, aby umela urcila hranice obdelnikove oblasti, ve
        // ktere tvar lezi (abychom vedeli kde pocitat integral). K tomu budou slouzit nasledujici metody, ktere
        // zde take definujeme jako abstraktni.
        // Tyto metody budou pouze pro interni potreby tridy Tvar a odvozenych trid a proto je skryjeme na
        // uroven protected (tzn. dostupne pouze uvnitr tridy Tvar a potomku).

        
        virtual double x_min() const = 0;
        virtual double x_max() const = 0;
        virtual double y_min() const = 0;
        virtual double y_max() const = 0;

};

#endif // TVAR_H 

//https://stackoverflow.com/questions/19814779/c-does-virtual-function-always-got-to-be-a-const