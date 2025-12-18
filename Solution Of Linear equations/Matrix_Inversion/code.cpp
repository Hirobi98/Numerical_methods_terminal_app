#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;


const double EPS = 1e-9;

void getCofactor(const vector<vector<double>>& A, vector<vector<double>>& temp,
                 int p, int q, int n);
double determinant(const vector<vector<double>>& A, int n);
void adjoint(const vector<vector<double>>& A, vector<vector<double>>& adj);
bool inverse(const vector<vector<double>>& A, vector<vector<double>>& inv, ofstream& fout, int n);

void getCofactor(const vector<vector<double>>& A, vector<vector<double>>& temp,
                 int p, int q, int n) {
    int i = 0, j = 0;

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row != p && col != q) {
                temp[i][j++] = A[row][col];

                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

double determinant(const vector<vector<double>>& A, int n) {
    if (n == 1)
        return A[0][0];

    double det = 0;

    vector<vector<double>> temp(n, vector<double>(n));
    int sign = 1;

    for (int f = 0; f < n; f++) {
        getCofactor(A, temp, 0, f, n);
        det += sign * A[0][f] * determinant(temp, n - 1);
        sign = -sign;
    }
    return det;
}

void adjoint(const vector<vector<double>>& A, vector<vector<double>>& adj) {
    int n = A.size();

    if (n == 1) {
        adj[0][0] = 1;
        return;
    }

    int sign;

    vector<vector<double>> temp(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            getCofactor(A, temp, i, j, n);
            sign = ((i + j) % 2 == 0) ? 1 : -1;
            adj[j][i] = sign * determinant(temp, n - 1);
        }
    }
}

bool inverse(const vector<vector<double>>& A, vector<vector<double>>& inv, ofstream& fout, int n) {
    double det = determinant(A, n);
    fout << "\nDeterminant of A: " << fixed << setprecision(4) << det << endl;

    if (fabs(det) < EPS) {

        fout << "System has infinite solutions or no solution." << endl;
        return false;
    }

    fout << "Inverse exists." << endl;

    vector<vector<double>> adj(n, vector<double>(n));
    adjoint(A, adj);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i][j] = adj[i][j] / det;

    return true;
}

void print_matrix(ofstream& fout, const vector<vector<double>>& M) {
    int n = M.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fout << setw(12) << M[i][j] << " ";
        fout << endl;
    }
}

void transpose(const vector<vector<double>>& A, vector<vector<double>>& AT) {
    int n = A.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT[j][i] = A[i][j];
}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    if (!fin.is_open() || !fout.is_open()) {
        cerr << "Error! Could not open input/output files." << endl;
        return 1;
    }

    int n;
    if (!(fin >> n)) {
        fout << "ERROR: Failed to read the number of equations (n)." << endl;
        fin.close(); fout.close(); return 1;
    }

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Read Matrix A coefficients
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (!(fin >> A[i][j])) {
                fout << "ERROR: Failed to read all coefficients of matrix A." << endl;
                fin.close(); fout.close(); return 1;
            }

    // Read Vector b coefficients
    for (int i = 0; i < n; i++)
        if (!(fin >> b[i])) {
            fout << "ERROR: Failed to read all coefficients of vector b." << endl;
            fin.close(); fout.close(); return 1;
        }


    vector<vector<double>> inv(n, vector<double>(n));

    fout << fixed << setprecision(6);
    fout << "Here N = " << n << endl;

    bool inverse_exists = inverse(A, inv, fout, n);

    // TRANSPOSE
    vector<vector<double>> AT(n, vector<double>(n));
    transpose(A, AT);

    fout << "\n TRANSPOSE MATRIX" << endl;
    fout << fixed << setprecision(6);
    print_matrix(fout, AT);


    if (inverse_exists) {
        // INVERSE MATRIX
        fout << "\nFINAL INVERSE MATRIX" << endl;
        fout << fixed << setprecision(6);
        print_matrix(fout, inv);

        // SOLVE x = A^-1 * b
        vector<double> x(n, 0.0);

        // Matrix multiplication
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                x[i] += inv[i][k] * b[k];
            }
        }


        fout << "The solution for Ax=b is :" << endl;
        fout << fixed << setprecision(4);
        for (int i = 0; i < n; i++) {
            fout << "x[" << i + 1 << "] = " << x[i] << endl;
        }
    } else {

        fout << "\n SOLUTION " << endl;
        fout << "Cannot compute solution x = A^-1 * b due to singular matrix (det=0)." << endl;
        fout << "there are infinite or zero solutions." << endl;
    }

    fin.close();
    fout.close();
    return 0;
}
