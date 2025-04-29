#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include "Clovek.h"
#include <typeinfo>
#include <algorithm>
#include <fstream>


//using namespace std;

Clovek::Clovek(){
}
bool Clovek::zkontroluj_vysku(double vyska){

    if((vyska > 0.0) && (vyska < 250.0)){
        std::cout << "Vyska sedi \n";
        std::cout << "\n";
        return true;
    }
    else {
        std::cout << "Vyska nesedi zadej znovu \n";
        std::cout << "\n";
        return false;
    }
}

double Clovek::zadej_vysku(){
    double vyska;

    std::cout << "Zadej Vysku: \t \n";
    std::cin >> vyska;
    std::cout << "\n";
    while(zkontroluj_vysku(vyska) != true){
        std:cout << "Zadej prosim vek ve tvaru celeho cisla bez desetinne carky: \t\t \n"; 
        std::cin.clear();
        std::cin.ignore();
        std::cin >> vyska;
        std::cout << "\n";
    }
    return vyska;
}


bool Clovek::check_vek(int age){
    if((age > 0) && (age < 250)){
        std::cout << "Vek sedi \n";
        std::cout << "\n";
        return true;
    }
    else {
        std::cout << "Vek nesedi zadej znovu \n";
        std::cout << "\n";
        return false;
    }
}

int Clovek::zadej_vek(){
    int temp_vek;
    bool spravny_input;

    std::cout << "Zadej vek: \t \n";
    std::cin >> temp_vek;
    std::cout << "\n";

    while(check_vek(temp_vek) != true){
        std::cout << "\n";
        std:cout << "Zadej prosim vek ve tvaru celeho cisla bez desetinne carky: \t\t \n"; 
        std::cin.clear();
        std::cin.ignore(numeric_limits<streamsize>::max());
        std::cin >> temp_vek;
        //std::cout << "\n";
    }
    return temp_vek;
}


std::string Clovek::zadej_jmeno(){
    std::string name;

    std::cout << "zadej jmeno \n";
    std::cin >> name;
    std::cout << "\n";

    return name;
    
}


std::string Clovek::string_to_lower(std::string to_transform){

    for (int i = 0; i< to_transform.length(); i++){
        to_transform[i] = tolower(to_transform[i]);
    }
    return to_transform;
}

std::string Clovek::zadej_prijmeni(){
    std::string prijmeni;

    std::cout << "zadej prijmeni \n";
    std::cin >> prijmeni;
    std::cout << "\n";
    prijmeni = string_to_lower(prijmeni);

    return prijmeni;
    
}

std::string Clovek::map_vzdelani(int id_vzdelani){
    std::string vzdelani;

    if(id_vzdelani == 1){
        return vzdelani = "ZS";
    }
    else if(id_vzdelani == 2){ 
        return vzdelani = "SS";
    }
    else if(id_vzdelani == 3){
        return vzdelani = "VS";
    }
    else{
        std::cout << "Worngfully set, enter Your name (ZS == 1, SS == 2, VS == 3): \n";
        return "false";
    }
}

std::string Clovek::zadej_vzdelani(){
    int id_vzdelani;
    std::string vzdelani;

        bool is_vzdelani_valid = false;

    while(is_vzdelani_valid == false){
        std::cout << "zadej skolu (1 == SS, 2 == SS, 3 == VS) \n";
        std::cin >> id_vzdelani;
        std::cout << "\n";
        vzdelani = map_vzdelani(id_vzdelani);
        std::cout << vzdelani << "\n";

        if (vzdelani == "VS"){ 
            std::cout << "Vs je svet";
            std::cout << "\n";
            is_vzdelani_valid = true;
        }
        else if (vzdelani == "SS"){
            std::cout << "SS je svet";
            std::cout << "\n";
            is_vzdelani_valid = true;
        }
        else if (vzdelani == "ZS"){
            std::cout << "ZS je svet";
            std::cout << "\n";
            is_vzdelani_valid = true;
        }
        else {
            std::cout << "fakt nic";
            std::cout << "\n";
            is_vzdelani_valid = false;
        }

    }

    return vzdelani;
    
}

void Clovek::vypis_info(std::vector<tuple <std::string, std::string, int, double, std::string>> person){
    std::cout << "zacnu iterovat" << person.size() << "\n";
    for(int i = 0; i <= person.size(); i++){
        std::cout << "Jmeno: \t\t" << get<1>(person[i]) << "\t";
        std::cout << "Prijmeni: \t" << get<0>(person[i]) << "\t";
        std::cout << "Vek: \t\t" << get<2>(person[i]) << "\t";
        std::cout << "Vyska: " << get<3>(person[i]) << "\t";
        std::cout << "Vzdelani " << get<4>(person[i]) << "\n";
    }
}

void Clovek::zapis_do_souboru(std::vector<tuple <std::string, std::string, int, double, std::string>> person){
    std::ofstream zapis;
    zapis.open("hw_ouput.txt");

    for(int i = 0; i <= person.size(); i++){

        zapis << std::get<1>(person[i]);
        zapis << ";";
        zapis << std::get<0>(person[i]);
        zapis << ";";
        zapis << std::get<4>(person[i]);
        zapis << ";";
        zapis << std::get<2>(person[i]);
        zapis << ";";
        zapis << std::get<3>(person[i]);
        zapis << ";";
    }
    zapis.close();
}


