#include <iostream>
#include <vector>
#include <random>

int f(int x) {
    return (x*13) % 10 - 1;
}

int main(){

    std::vector<int> x;
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<std::mt19937::result_type> kostka(1,6); // distribution in range [1, 6]
    std::vector<int> cisla(10);


    for (int i = 0; i <= kostka.size(); i++){
        x = kostka(gen);
    }

    double h = 0.0;

    for (int i=0; i <= cisla.size(); i++){
        h += 1 / cisla[i];
    }
    h = 1.0 / h;

    std::cout << "Harmonicky prumer nahodnych cisel je: " << h << "\n";

    return 0;
}