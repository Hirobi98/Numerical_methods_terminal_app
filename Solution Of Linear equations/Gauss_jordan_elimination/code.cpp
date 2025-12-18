#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

const double EPS = 1e-9;

void print_augmented_matrix(ofstream& fout, int n, const vector<vector<double>>& A, const vector<double>& b) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fout << setw(10) << A[i][j] << " ";
        }
        fout << " | " << setw(10) << b[i] << endl;
    }

}

int gauss_jordan_eliminate(int n, vector<vector<double>>& A, vector<double>& b) {

    for (int i = 0; i < n; ++i) {

        int pivot = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(A[j][i]) > fabs(A[pivot][i]))
                pivot = j;

        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);


        if (fabs(A[i][i]) < EPS) {
            return -1;
        }


        double pivotVal = A[i][i];
        for (int j = 0; j < n; ++j)
            A[i][j] /= pivotVal;
        b[i] /= pivotVal;


        for (int j = 0; j < n; ++j) {
            if (j != i) {
                double factor = A[j][i];
                for (int k = 0; k < n; ++k)
                    A[j][k] -= factor * A[i][k];
                b[j] -= factor * b[i];
            }
        }
    }
    return 0;
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
        fin.close();
        fout.close();
        return 1;
    }

    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Read coefficients of A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(fin >> A[i][j])) {
                fout << "ERROR: Failed to read all coefficients of matrix A." << endl;
                fin.close();
                fout.close();
                return 1;
            }
        }
    }

    // Read constants vector b
    for (int i = 0; i < n; ++i) {
        if (!(fin >> b[i])) {
            fout << "ERROR: Failed to read all coefficients of vector b." << endl;
            fin.close();
            fout.close();
            return 1;
        }
    }



    int status = gauss_jordan_eliminate(n, A, b);

    if (status == -1) {
        fout << "ERROR: No unique solution exists (Singular matrix encountered).\n";
        fin.close();
        fout.close();
        return 1;
    }


    fout << fixed << setprecision(3);
    fout << "Reduced Row Echelon Form of the Augmented Matrix:" << endl;
    print_augmented_matrix(fout, n, A, b);

    fout << endl;
    fout << "Solution for each variable rounded to 3 decimal places:" << endl;

    for (int i = 0; i < n; ++i) {
        fout << "x[" << i + 1 << "] = " << b[i] << endl;
    }

    fin.close();
    fout.close();

    return 0;
}
