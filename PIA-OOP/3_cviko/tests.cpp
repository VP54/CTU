#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

std::vector<int> nacti_soubor(){
    std::vector<int> vstupni_vektor;
    int vstupni_data;
    
    //Nacti data
    std::ifstream cteni_soubor("./vstupni_data.txt");

    while (cteni_soubor >> vstupni_data){
        vstupni_vektor.push_back(vstupni_data);
    }

    return vstupni_vektor;
}

void print_result(std::vector<int> vstupni_vektor, std::vector<int> merici_vektor, double suma){
    // Funkce nevraci nic protoze void ale printi vlastne to co chci (zde mozna zapis do souboru)

    std::cout << "Velikost celku: " << vstupni_vektor.size() << "\n";
    std::cout << "velikost_souboru: " << merici_vektor.size() << "\n";
    std::cout << "Prumer cisel: " << suma / merici_vektor.size() << '\n';
}

int main(){
    // Zadefinuju si promenne
    std::vector<int> merici_vektor;
    std::vector<int> vstupni_vektor;
    double suma;
    suma = 0.0;

    //Inicializace
    vstupni_vektor = nacti_soubor();
    
    // Worker asi
    for (int i = 0; i < vstupni_vektor.size(); i++ ){
        if ((vstupni_vektor[i] > 4735) && (vstupni_vektor[i] < 5230)){
            merici_vektor.push_back(vstupni_vektor[i]);
        }
    }

    for (int i = 0; i < merici_vektor.size(); i++){
        suma += merici_vektor[i];
    }

    // Vytiskni mi muj vystup
    print_result(vstupni_vektor, merici_vektor, suma);

    return 0;
}