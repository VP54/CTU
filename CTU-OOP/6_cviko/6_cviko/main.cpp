#include <iostream>
#include <memory>

#include "Tvar.h"
#include "Kruh.h"
//include "Srdce.h"

int main()
{
    // Toto je smart pointer
    // Funguje podobne jako ukazatel (Tvar*), ale nemusime se starat o dealokaci
    // Diky polymorfismu muze ukazovat take na jakykoli objekt, jehoz trida je odvozena od tridy Tvar
    std::shared_ptr<Tvar> utvar;

    int utvar_id;
    std::cout << "Vyberte druh utvaru (1 - kruh, 2 - kyticka, 3 - srdce): " << std::endl;
    std::cin >> utvar_id;

    // Podle uzivatelskeho zadani vytvorime vybrany objekt a nasmerujeme na nej ukazatel
    // K tomu slouzi funkce make_shared, jeji parametry budou stejne jako u konstruktoru naseho objektu
    if (utvar_id == 1) {
        utvar = std::make_shared<Tvar>(1.0,0.0,0.0);
    }
    //else if (utvar_id == 2) {
    //    utvar = std::make_shared<Kyticka>(4.0,0.0,0.0);
    //}
    //else if (utvar_id == 3) {
    //    utvar = std::make_shared<Srdce>(0.0,0.0);
    //}
    else {
	std::cout << "Neznamy tvar.\n";
	return 1;
    }

    // Nyni vypocitame obsah utvaru
    // Podle toho, ktere tridy je dany objekt se bud pouzije vypocet vzoreckem (trida Kruh)
    // anebo vypocet pomoci numericke integrace (Kyticka, Srdce)
    double obsah = utvar->obsah(); // Pri predavani objektu musime ukazatel dereferencovat operatorem *
                               // Jinak by se prekladac snazil predat funkci pouze
                               // ciselnou hodnotu adresy v pameti

    std::cout << "Obsah utvaru je " << obsah << std::endl;

    return 0;
}

//https://cplusplus.com/forum/general/107753/

//