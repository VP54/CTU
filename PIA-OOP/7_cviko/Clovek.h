
#include <typeinfo>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
using namespace std;
/*  */
class Clovek
{

public:
        Clovek();

        static bool zkontroluj_vysku(double vyska);
        static double zadej_vysku();
        static bool check_vek(int age);
        static int zadej_vek();
        static std::string zadej_jmeno();
        static std::string zadej_prijmeni();
        
        static std::string zadej_vzdelani();
        static std::string map_vzdelani(int id_vzdelani);

        static void vypis_info(std::vector<tuple <std::string, std::string, int, double, std::string>> person);
        static std::string string_to_lower(std::string to_transform);
        static void zapis_do_souboru(std::vector<tuple <std::string, std::string, int, double, std::string>> person);

};


