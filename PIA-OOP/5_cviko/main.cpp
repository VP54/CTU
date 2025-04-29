#include <iostream>
#include <vector>
#include <cmath>
#include <cmath>
#include "Kruh.cpp"

/**

Dobry den, ja programuju ve VS Codu na Win10, a tak jsem dostaval errory na 
                    -->     #include "Kruh.h" (zmenil jsem koncovku)
                    -->     infdef KRUH_H -- endif
 
 
*/

int main() {
    // Trida Kruh bude vytvorena pomoci konstruktoru Kruh(r,x,y), kde r je polomer a x,y souradnice stredu kruhu
     
    std::vector<Kruh> utvary = {Kruh(1,1,1), Kruh(0.5,0,0), Kruh(0.2,1,0)};

    for (Kruh k : utvary) {
        std::cout << "Obsah kruhu = " << k.obsah() << ", obvod kruhu = " << k.obvod() << "\n";
        
        if (k.leziUvnitr(0.3,0.3)) {
            std::cout << "Bod se souradnicemi (0.3,0.3) lezi uvnitr tohoto kruhu.\n";
        } 
	else {
            std::cout << "Bod se souradnicemi (0.3,0.3) nelezi uvnitr tohoto kruhu.\n";
        }
        
    } 
    
};

