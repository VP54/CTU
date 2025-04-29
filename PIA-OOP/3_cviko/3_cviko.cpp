#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>


int main(){
    std::vector<int> merici_vektor;
    std::vector<int> vstupni_vektor;
    int vstupni_data;
    double suma;
    suma = 0.0;

    //Nacti data
    std::ifstream cteni_soubor("./vstupni_data.txt");

    while (cteni_soubor >> vstupni_data){
        vstupni_vektor.push_back(vstupni_data);
    }
    
    for (int i = 0; i < vstupni_vektor.size(); i++ ){
        if ((vstupni_vektor[i] > 4735) && (vstupni_vektor[i] < 5230)){
            merici_vektor.push_back(vstupni_vektor[i]);
        }
    }

    for (int i = 0; i < merici_vektor.size(); i++){
        suma += merici_vektor[i];
    }

    std::cout << "Velikost celku: " << vstupni_vektor.size() << "\n";
    std::cout << "velikost_souboru: " << merici_vektor.size() << "\n";
    std::cout << "Prumer cisel: " << suma / merici_vektor.size() << '\n';

    

    return 0;
}