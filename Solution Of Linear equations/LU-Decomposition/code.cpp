#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
using namespace std;

const double EPS = 1e-9;

int rankOfMatrix(vector<vector<double>> mat) {
    int n = mat.size();
    int m = mat[0].size();
    int rank = 0;

    for (int col = 0, row = 0; col < m && row < n; col++) {
        int pivot = row;
        for (int i = row; i < n; i++) {
            if (fabs(mat[i][col]) > fabs(mat[pivot][col]))
                pivot = i;
        }
        if (fabs(mat[pivot][col]) < EPS) continue;

        swap(mat[row], mat[pivot]);

        for (int i = row + 1; i < n; i++) {
            double factor = mat[i][col] / mat[row][col];
            for (int j = col; j < m; j++)
                mat[i][j] -= factor * mat[row][j];
        }
        row++;
        rank++;
    }
    return rank;
}

int main() {
    ifstream fin("input.txt");   
    ofstream fout("output.txt");

    if (!fin.is_open() || !fout.is_open()) {
        cerr << "Error opening input/output file.\n";
        return 1;
    }

    int n;
    
    while (fin >> n) {
        vector<vector<double>> aug(n, vector<double>(n + 1));
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= n; j++)
                fin >> aug[i][j];

        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                A[i][j] = aug[i][j];
            b[i] = aug[i][n];
        }

        int rankA = rankOfMatrix(A);
        int rankAug = rankOfMatrix(aug);

        if (rankA < rankAug) {
            fout << "The system has no solution.\n\n";
        } else if (rankA < n) {
            fout << "The system has infinite solutions.\n\n";
        } else {
            vector<vector<double>> L(n, vector<double>(n, 0));
            vector<vector<double>> U(n, vector<double>(n, 0));
            vector<int> P(n);
            for (int i = 0; i < n; i++) P[i] = i;

            bool singular = false;

            for (int i = 0; i < n; i++) {
                int pivot = i;
                for (int k = i; k < n; k++) {
                    if (fabs(A[k][i]) > fabs(A[pivot][i]))
                        pivot = k;
                }
                if (fabs(A[pivot][i]) < EPS) {
                    singular = true;
                    break;
                }

                swap(A[i], A[pivot]);
                swap(P[i], P[pivot]);

                for (int j = i; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < i; k++)
                        sum += L[i][k] * U[k][j];
                    U[i][j] = A[i][j] - sum;
                }
                for (int j = i; j < n; j++) {
                    if (i == j) L[i][i] = 1;
                    else {
                        double sum = 0;
                        for (int k = 0; k < i; k++)
                            sum += L[j][k] * U[k][i];
                        L[j][i] = (A[j][i] - sum) / U[i][i];
                    }
                }
            }

            if (singular) {
                fout << "Matrix is singular or nearly singular. No unique solution.\n\n";
            } else {
                fout << "Lower Matrix L:\n";
                for (auto& row : L) {
                    for (double val : row)
                        fout << setw(10) << fixed << setprecision(3) << val << " ";
                    fout << "\n";
                }

                fout << "Upper Matrix U:\n";
                for (auto& row : U) {
                    for (double val : row)
                        fout << setw(10) << fixed << setprecision(3) << val << " ";
                    fout << "\n";
                }

                vector<double> y(n);
                for (int i = 0; i < n; i++) {
                    double sum = 0;
                    for (int j = 0; j < i; j++)
                        sum += L[i][j] * y[j];
                    y[i] = b[P[i]] - sum;
                }

                vector<double> x(n);
                bool unique = true;
                for (int i = n - 1; i >= 0; i--) {
                    double sum = 0;
                    for (int j = i + 1; j < n; j++)
                        sum += U[i][j] * x[j];
                    if (fabs(U[i][i]) < EPS) {
                        unique = false;
                        break;
                    }
                    x[i] = (y[i] - sum) / U[i][i];
                }

                if (unique) {
                    fout << "Solution:\n";
                    for (int i = 0; i < n; i++)
                        fout << "x" << i + 1 << " = " << fixed << setprecision(3) << x[i] << "\n";
                    fout << "The system has a unique solution.\n\n";
                } else {
                    fout << "The system has no unique solution.\n\n";
                }
            }
        }
    }

    fin.close();
    fout.close();
    return 0;
}