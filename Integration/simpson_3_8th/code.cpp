#include<bits/stdc++.h>
#include <functional>
#include<fstream>

using namespace std;
auto f=[](double x){

return x*x*x;

};


double simpsons38(double a, double b, int n) {
    if (n % 3 != 0) {
        cerr << "Error: n must be a multiple of 3.\n";
        return NAN;
    }

    double h = (b - a) / n;
    double sum = f(a) + f(b);


    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 3 == 0) {
            sum += 2 * f(x);
        } else {
            sum += 3 * f(x);
        }
    }

    return (3 * h / 8) * sum;
}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    double a,b;
    int n;
    if(!fin.is_open()){
        cerr<<"Error !could not open the input.txt"<<endl;
        return 1;

    }

    if(!fout.is_open()){
        cerr<<"Error!could not open the output.txt"<<endl;
        return 1;

    }

    if(!(fin>>a>>b>>n)){
        fout<<"Error: Failed to load a,b,n from input.txt"<<endl;

        fin.close();
        fout.close();
        return 1;

    }

    double result = simpsons38(a, b, n);
    if(isnan(result)){
        fout<<"error:number of subintervals must be a multiple of 3 for simpsons 3/8th"<<endl;
    }
    else{
        fout<<fixed<<setprecision(10)<<"The integral result is : "<<result<<endl;
    }
    fin.close();
    fout.close();

    return 0;
}
