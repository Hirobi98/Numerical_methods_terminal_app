#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
using namespace std;

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n;
    fin >> n;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) fin >> x[i];
    for (int i = 0; i < n; i++) fin >> y[i];

    vector<vector<double>> d(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) d[i][0] = y[i];
    for (int j = 1; j < n; j++)
        for (int i = 0; i < n - j; i++)
            d[i][j] = (d[i+1][j-1] - d[i][j-1]) / (x[i+j] - x[i]);

    double value;
    fin >> value;
    double result = d[0][0];
    double term = 1.0;

    for (int i = 1; i < n; i++) {
        term *= (value - x[i-1]);
        result += d[0][i] * term;
    }

    fout << fixed << setprecision(6) << result << endl;
    return 0;
}