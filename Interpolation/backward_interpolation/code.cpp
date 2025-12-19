#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
using namespace std;

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n;
    fin >> n;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) fin >> x[i];
    for (int i = 0; i < n; i++) fin >> y[i];

    vector<vector<double>> diff(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) diff[i][0] = y[i];
    for (int j = 1; j < n; j++)
        for (int i = j; i < n; i++)
            diff[i][j] = diff[i][j-1] - diff[i-1][j-1];

    double value;
    fin >> value;
    double h = x[1] - x[0];
    double u = (value - x[n-1]) / h;
    double result = y[n-1];
    double term = 1.0;

    for (int i = 1; i < n; i++) {
        term *= (u + (i - 1));
        result += (term * diff[n-1][i]) / tgamma(i + 1);
    }

    fout << fixed << setprecision(6) << result << endl;
    return 0;
}