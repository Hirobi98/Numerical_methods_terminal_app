#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n;
    fin >> n;

    vector<double> t(n), T(n);

    for (int i = 0; i < n; i++) fin >> t[i];
    for (int i = 0; i < n; i++) fin >> T[i];

    // Compute f(t_i) = exp(t_i / 4)
    vector<double> f(n);

    double sumf = 0, sumf2 = 0, sumT = 0, sumfT = 0;

    for (int i = 0; i < n; i++) {
        f[i] = exp(t[i] / 4.0);
        sumf += f[i];
        sumf2 += f[i] * f[i];
        sumT += T[i];
        sumfT += f[i] * T[i];
    }

    double denom = n * sumf2 - sumf * sumf;

    if (fabs(denom) < 1e-15) {
        fout << "Error: Singular matrix. Cannot compute regression.\n";
        return 0;
    }

    // Least squares formulas
    double b = (n * sumfT - sumf * sumT) / denom;
    double a = (sumT - b * sumf) / n;

    fout << fixed << setprecision(10);
    fout << "Computed parameters:\n";
    fout << "a = " << a << "\n";
    fout << "b = " << b << "\n";

    // Prediction
    double t_predict;
    fin >> t_predict;

    double f_predict = exp(t_predict / 4.0);
    double T_predict = a + b * f_predict;

    fout << "\nEstimated T(" << t_predict << ") = " << T_predict << "\n";

    fin.close();
    fout.close();

    return 0;
}
