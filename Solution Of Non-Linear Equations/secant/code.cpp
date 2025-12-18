#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace std;

#define ll double


ll poly(ll x, const vector<ll>& coeff) {
    ll result = 0;
    int n = coeff.size();
    for (int i = 0; i < n; i++) {
        result += coeff[i] * pow(x, n - 1 - i);
    }
    return result;
}

int main() {
    ifstream fin("input.txt"); 
    ofstream fout("output.txt"); 

    if (!fin) {
        cerr << "Error: input.txt not found.\n";
        return 1;
    }

    double step = 0.1;
    double epsilon = 0.0001;

    int n;
    while (fin >> n) {
        vector<ll> coeff(n + 1);
        for (int i = 0; i <= n; i++) fin >> coeff[i];

        fout << "Polynomial degree " << n << "\n";
        fout << "Coefficients: ";
        for (auto c : coeff) fout << c << " ";
        fout << "\n";

        vector<pair<double,double>> intervals;

    
        for (double i = -10; i <= 10; i += step) {
            double f1 = poly(i, coeff);
            double f2 = poly(i + step, coeff);
            if (f1 * f2 < 0) {
                intervals.push_back({i, i + step});
            }
        }

        fout << "Intervals detected: " << intervals.size() << "\n";
        fout << fixed << setprecision(6);

        for (auto interval : intervals) {
            double x0 = interval.first;
            double x1 = interval.second;
            double x2;
            int iter = 0;

            while (true) {
                double fx0 = poly(x0, coeff);
                double fx1 = poly(x1, coeff);

                if (fabs(fx1 - fx0) < 1e-12) break;

                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0);
                iter++;

                if (fabs(x2 - x1) < epsilon) {
                    fout << "Root ≈ " << x2 << "  in " << iter << " iterations\n";
                    break;
                }

                x0 = x1;
                x1 = x2;

                if (iter > 1000) break;
            }
        }

        if (intervals.empty()) {
            fout << "No root intervals found in the given range.\n";
        }

        fout << "\n";
    }

    fin.close();
    fout.close();

    cout << "Processing complete. Results written to output.txt\n";
    return 0;
}