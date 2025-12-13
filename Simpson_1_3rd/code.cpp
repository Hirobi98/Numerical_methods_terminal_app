#include<bits/stdc++.h>
#include <fstream>
#include <cmath>
#include <functional>

auto f = [](double x) {

    return x * x * x;

};

double simpsons13(double a, double b, int n, std::function<double(double)> func) {
    if (n % 2 != 0) {

        return NAN;
    }
    double h = (b - a) / n;
    double sum = func(a) + func(b);

    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * func(x);
        } else {
            sum += 4 * func(x);
        }
    }

    return (h / 3) * sum;
}

using namespace std;

int main() {

    ifstream fin("input.txt");

    ofstream fout("output.txt");
    double a = 0.0, b = 0.0;
    int n = 0;

    if (!fin.is_open()) {
        cerr << "Error: Could not open input.txt" << std::endl;
        return 1;
    }
    if (!fout.is_open()) {
        cerr << "Error: Could not open output.txt" << std::endl;
        return 1;
    }

    if (!(fin >> a >> b >> n)) {
        fout << "ERROR: Failed to read input data." << std::endl;
        fin.close();
        fout.close();
        return 1;
    }


    double result = simpsons13(a, b, n, f);


    if (isnan(result)) {

        fout << "ERROR: Number of sub-intervals (n) must be even for Simpson's 1/3 Rule." << std::endl;
    } else {

        fout << result << std::endl;
    }

    fin.close();
    fout.close();

    return 0;
}
