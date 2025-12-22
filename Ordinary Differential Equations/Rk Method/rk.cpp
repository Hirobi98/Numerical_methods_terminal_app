#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

double f(double x, double y) {
    return x * x + y;
}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    double x0, y0, x, h;
    fin >> x0 >> y0 >> x >> h;

    int n = (x - x0) / h;
    double y = y0;
    double k1, k2, k3, k4;

    for (int i = 0; i < n; i++) {
        k1 = h * f(x0, y);
        k2 = h * f(x0 + h/2, y + k1/2);
        k3 = h * f(x0 + h/2, y + k2/2);
        k4 = h * f(x0 + h, y + k3);

        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6;
        x0 = x0 + h;
    }

    fout << fixed << setprecision(6);
    fout << "Value of y at x = " << x << " is " << y << endl;

    fin.close();
    fout.close();
    return 0;
}
