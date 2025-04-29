// Online C++ compiler to run C++ program online
#include <iostream>
#include <tuple>
#include <vector>
using namespace std;

int main() {
    //definice tech blbosti
    int r_zz = 12;
    int g_zz = 13;
    int b_zz = 14;
    
    // Write C++ code here
    vector< tuple < int, int, int > > hra;
    //naplnim hru pushbachovanim jlelikoz je to vektor tuplu
    hra.push_back(make_tuple(12,13,14));
    hra.push_back(make_tuple(1,2,3));
    hra.push_back(make_tuple(20,40,50));
    
    for(int tah; tah< hra.size(); tah++){
        int getnuty_r = get<0>(hra[tah]);
        int getnuty_g = get<1>(hra[tah]);
        int getnuty_b = get<2>(hra[tah]);
        
        if(getnuty_r == r_zz && getnuty_g == g_zz && getnuty_b == b_zz ){
            cout << "Mnewwww";
            return 1;
        }
        else{return 0;}
    }
    

    return 0;
}