#include <iostream>
#include <memory>
#include "Tvar.cpp"
#include "Kruh.cpp"

int main()
{
    int utvar_id;
    std::shared_ptr<Tvar> utvar;
    std::cout << "Vyberte druh utvaru (1 - kruh): " << std::endl;
    std::cin >> utvar_id;

    if (utvar_id == 1) {
        utvar = std::make_shared<Kruh>(1.0,0.0,0.0);
    }

    else {
    	std::cout << "Neznamy tvar.\n";
	    return 1;
    }

    double obsah = utvar->obsah(); 
    std::cout << "Obsah utvaru je " << obsah << std::endl;

    return 0;
}