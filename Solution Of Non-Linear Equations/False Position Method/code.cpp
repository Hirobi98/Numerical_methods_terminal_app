#include <bits/stdc++.h>
using namespace std;

double f(double x) {
    return x*x*x - x - 2;  
}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    double a, b, c, e;
    int n, i = 1;

    fin >> a >> b;
    fin >> e;
    fin >> n;

    if (f(a) * f(b) >= 0) {
        fout << "Method not applicable\n";
        return 0;
    }

    fout << fixed << setprecision(6);
    fout << "Iter\t a\t\t b\t\t c\t\t f(c)\n";

    while (i <= n) {
        c = (a * f(b) - b * f(a)) / (f(b) - f(a));

        fout << i << "\t "
             << a << "\t "
             << b << "\t "
             << c << "\t "
             << f(c) << "\n";

        if (fabs(f(c)) < e) {
            fout << "\nRoot = " << c << "\n";
            return 0;
        }

        if (f(a) * f(c) < 0)
            b = c;
        else
            a = c;

        i++;
    }

    fout << "\nApprox Root = " << c << "\n";

    fin.close();
    fout.close();

    return 0;
}
