#include <bits/stdc++.h>
using namespace std;

vector<double> gaussianElimination(vector<vector<double>> A, vector<double> b) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int j = i + 1; j < n; j++)
            if (abs(A[j][i]) > abs(A[pivot][i]))
                pivot = j;

        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);

        double factor = A[i][i];
        for (int j = i; j < n; j++) A[i][j] /= factor;
        b[i] /= factor;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double f = A[j][i];
            for (int k = i; k < n; k++)
                A[j][k] -= f * A[i][k];
            b[j] -= f * b[i];
        }
    }
    return b;
}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n, degree;
    fin >> n >> degree;

    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++)
        fin >> x[i];
    for (int i = 0; i < n; i++)
         fin >> y[i];

    vector<vector<double>> A(degree + 1, vector<double>(degree + 1, 0));
    vector<double> B(degree + 1, 0);

    for (int i = 0; i <= degree; i++) {
        for (int j = 0; j <= degree; j++)
            for (int k = 0; k < n; k++)
                A[i][j] += pow(x[k], i + j);

        for (int k = 0; k < n; k++)
            B[i] += y[k] * pow(x[k], i);
    }

    vector<double> coeff = gaussianElimination(A, B);

    fout << fixed << setprecision(6);
    for (int i = 0; i <= degree; i++)
        fout << "a" << i << " = " << coeff[i] << "\n";

    return 0;
}
