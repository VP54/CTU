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
/*

#include<random>
//include Kruh.h

int main(){
    double rand_x, rand_y;
    double r;
    int in, out, range;

    r = 1;
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
       if (i % 10000 == 0){
        std::cout << "Temp obsah: \t\t" << 4.0 * in / (in + out) << "\n";
       }
    }
    double obsah = 4.0 * in / (in + out);

    std::cout << "obsah: " << obsah << "\t\t\t" << in + out << "\n";

}
*/