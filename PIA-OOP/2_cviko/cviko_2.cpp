#include <iostream>
#include <cmath>
#include <typeinfo>
#include <fstream>

/*
Program načte z uživatelského vstupu následující údaje do odpovídajících proměnných:
  jméno,                            OK
  příjmení,                         OK 
  výšku,                            
  věk,                              
  vzdělání (ZŠ/SŠ/VŠ apod.).        
Program zkontroluje jestli uživatel nezadal nesmyslné číselné údaje (věk např. v intervalu 18-120, výška 100-230). U výšky jsou přípustná i desetinná čísla.
Pro zadání vzdělání dostane uživatel nabídku přípustných hodnot (např. jako seznam čísel nebo výrazů představující jednotlivé možnosti). Program zkontroluje jestli uživatel vybral platnou volbu.
Program vypíše údaje na obrazovku ve strukturované formě a/nebo je uloží je do souboru.


*/


int main() {
    int vek, id_vzdelani;
    std::string jmeno, prijmeni,vzdelani;

    
    // JMENO a PRIJMENI
    std::cout << "Enter name: ";
    std::cin >> jmeno;

    std::cout << "Enter last name: ";
    std::cin >> prijmeni;


    // VZDELANI
    std::cout << "Enter your jmeno in form (ZS == 1, SS == 2, VS == 3): \n";
    std::cin >> id_vzdelani;
    
    while((id_vzdelani != 1) && (id_vzdelani != 2) && (id_vzdelani != 3))
    {
        std::cout << "Worngfully set, enter Your name (ZS == 1, SS == 2, VS == 3): \n";
        std::cin >> id_vzdelani;
    }
    
    
    if(id_vzdelani == 1) { //podminka c.1
    vzdelani = "ZS";
    }
    else if(id_vzdelani == 2) { //podminka c.2
    vzdelani = "SS";
    }
    else {
    vzdelani = "VS";
    }    

    // VYSKA
    double vyska;
    std::cout << "Zadej vysku v rozmezi 0-250 [cm]: " << std::endl;
    std::cin >> vyska;

    while((vyska < 0.0) || vyska > 250.0){
        std::cout << "Zadej vysku v rozmezi 0-250[cm]: " << std::endl;
        std::cin >> vyska;
    }

    // VEK
    std::cout << "Zadej vek:  (Ve tvaru celeho cisla od 0-120)" << std::endl;
    std::cin >> vek;
    
    while((vek < 0) || (vek > 120) ){ 
    // Pokud je cislo float veme jen jeho celou cast (int.floor), to je OK
    std::cout << "Zadej vek:  (Ve tvaru celeho cisla od 0-120)" << std::endl;
    std::cin >> vek;
    }
    

    std::cout << "Jmeno: " << jmeno << '\n';
    std::cout << "Prijmeni: " << prijmeni << '\n';
    std::cout << "Final vzdelani: " << vzdelani << '\n';
    std::cout << "Vek je: " << vek << '\n';
    std::cout << "Vyska je: " << vyska << "\n";

    std::ofstream zapis("newfile.txt");
    zapis << jmeno;
    zapis << "\n";
    zapis << prijmeni;
    zapis << "\n";
    zapis << vzdelani;
    zapis << "\n";
    zapis << vek;
    zapis << "\n";
    zapis << vyska;


}

