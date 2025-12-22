### Numerical methods

## Table of Contents

- [Solution of Linear Equations](#solution-of-linear-equations)
  - [Gauss Elimination Method](#gauss-elimination-method)
    - [Theory](#gauss-elimination-theory)
      - [Intro](#gauss-elimination-introduction)
      - [Formula](#gauss-elimination-formula)
      - [Algorithm](#gauss-elimination-algorithm-steps)
      - [Application](#gauss-elimination-application)
    - [Code](#gauss-elimination-code)
    - [Input](#gauss-elimination-input)
    - [Output](#gauss-elimination-output)
  - [Gauss Jordan Elimination Method](#gauss-jordan-elimination-method)
    - [Theory](#gauss-jordan-theory)
      - [Intro](#gauss-jordan-introduction)
      - [Formula](#gauss-jordan-formula)
      - [Algorithm](#gauss-jordan-algorithm-steps)
      - [Application](#gauss-jordan-application)
    - [Code](#gauss-jordan-code)
    - [Input](#gauss-jordan-input)
    - [Output](#gauss-jordan-output)
  - [LU Decomposition Method](#lu-decomposition-method)
    - [Theory](#lu-decomposition-theory)
      - [Intro](#lu-decomposition-introduction)
      - [Formula](#lu-decomposition-formula)
      - [Algorithm](#lu-decomposition-algorithm-steps)
      - [Application](#lu-decomposition-application)
    - [Code](#lu-decomposition-code)
    - [Input](#lu-decomposition-input)
    - [Output](#lu-decomposition-output)
  - [Matrix Inversion](#matrix-inversion)
    - [Theory](#matrix-inversion-theory)
      - [Intro](#matrix-inversion-introduction)
      - [Formula](#matrix-inversion-formula)
      - [Algorithm](#matrix-inversion-algorithm-steps)
      - [Application](#matrix-inversion-application)
    - [Code](#matrix-inversion-code)
    - [Input](#matrix-inversion-input)
    - [Output](#matrix-inversion-output)

- [Solution of Non-Linear Equations](#solution-of-non-linear-equations)
  - [Bisection Method](#bisection-method)
    - [Theory](#bisection-theory)
      - [Intro](#bisection-introduction)
      - [Formula](#bisection-formula)
      - [Algorithm](#bisection-algorithm-steps)
      - [Application](#bisection-application)
    - [Code](#bisection-code)
    - [Input](#bisection-input)
    - [Output](#bisection-output)
  - [False Position Method](#false-position-method)
    - [Theory](#false-position-theory)
      - [Intro](#false-position-introduction)
      - [Formula](#false-position-formula)
      - [Algorithm](#false-position-algorithm-steps)
      - [Application](#false-position-application)
    - [Code](#false-position-code)
    - [Input](#false-position-input)
    - [Output](#false-position-output)
  - [Secant Method](#secant-method)
    - [Theory](#secant-theory)
      - [Intro](#secant-introduction)
      - [Formula](#secant-formula)
      - [Algorithm](#secant-algorithm-steps)
      - [Application](#secant-application)
    - [Code](#secant-code)
    - [Input](#secant-input)
    - [Output](#secant-output)
  - [Newton Raphson Method](#newton-raphson-method)
    - [Theory](#newton-raphson-theory)
      - [Intro](#newton-raphson-introduction)
      - [Formula](#newton-raphson-formula)
      - [Algorithm](#newton-raphson-algorithm-steps)
      - [Application](#newton-raphson-applications)
    - [Code](#newton-raphson-code)
    - [Input](#newton-raphson-input)
    - [Output](#newton-raphson-output)


- [Solution of Interpolation](#solution-of-interpolation)
  - [Newton's Forward Interpolation Method](#newtons-forward-interpolation-method)
    - [Theory](#newtons-forward-interpolation-theory)
      - [Introduction](#newtons-forward-interpolation-introduction)
      - [Formula](#newtons-forward-interpolation-formula)
      - [Algorithm Steps](#newtons-forward-interpolation-algorithm-steps)
      - [Application](#newtons-forward-interpolation-application)
    - [Code](#newtons-forward-interpolation-code)
    - [Input](#newtons-forward-interpolation-input)
    - [Output](#newtons-forward-interpolation-output)
  - [Newton's Backward Interpolation Method](#newtons-backward-interpolation-method)
    - [Theory](#newtons-backward-interpolation-theory)
      - [Introduction](#newtons-backward-interpolation-introduction)
      - [Formula](#newtons-backward-interpolation-formula)
      - [Algorithm Steps](#newtons-backward-interpolation-algorithm-steps)
      - [Application](#newtons-backward-interpolation-application)
    - [Code](#newtons-backward-interpolation-code)
    - [Input](#newtons-backward-interpolation-input)
    - [Output](#newtons-backward-interpolation-output)
  - [Divided Difference Method](#divided-difference-method)
    - [Theory](#divided-difference-theory)
      - [Introduction](#divided-difference-introduction)
      - [Formula](#divided-difference-formula)
      - [Algorithm Steps](#divided-difference-steps)
      - [Application](#divided-difference-application)
    - [Code](#divided-difference-code)
    - [Input](#divided-difference-input)
    - [Output](#divided-difference-output)

- [Solution of Curve Fitting Model](#solution-of-curve-fitting-model)
  - [Least Square Regression Method For Linear Equations](#least-square-regression-method-for-linear-equations)
    - [Theory](#least-square-regression-method-for-linear-equations-theory)
      - [Introduction](#least-square-regression-method-for-linear-equations-introduction)
      - [Formula](#least-square-regression-method-for-linear-equations-formula)
      - [Algorithm Steps](#least-square-regression-method-for-linear-equations-steps)
      - [Application](#least-square-regression-method-for-linear-equations-application)
    - [Code](#least-square-regression-method-for-linear-equations-code)
    - [Input](#least-square-regression-method-for-linear-equations-input)
    - [Output](#least-square-regression-method-for-linear-equations-output)
  - [Least Square Regression Method For Transcendental Equations](#least-square-regression-method-for-transcendental-equations)
    - [Theory](#least-square-regression-method-for-transcendental-equations-theory)
      - [Introduction](#least-square-regression-method-for-transcendental-equations-introduction)
      - [Formula](#least-square-regression-method-for-transcendental-equations-formula)
      - [Algorithm Steps](#least-square-regression-method-for-transcendental-equations-steps)
      - [Application](#least-square-regression-method-for-transcendental-equations-application)
    - [Code](#least-square-regression-method-for-transcendental-equations-code)
    - [Input](#least-square-regression-method-for-transcendental-equations-input)
    - [Output](#least-square-regression-method-for-transcendental-equations-output)
  - [Least Square Regression Method For Polynomial Equations](#least-square-regression-method-for-polynomial-equations)
    - [Theory](#least-square-regression-method-for-polynomial-equations-theory)
      - [Introduction](#least-square-regression-method-for-polynomial-equations-introduction)
      - [Formula](#least-square-regression-method-for-polynomial-equations-formula)
      - [Algorithm Steps](#least-square-regression-method-for-polynomial-equations-steps)
      - [Application](#least-square-regression-method-for-polynomial-equations-application)
    - [Code](#least-square-regression-method-for-polynomial-equations-code)
    - [Input](#least-square-regression-method-for-polynomial-equations-input)
    - [Output](#least-square-regression-method-for-polynomial-equations-output)



- [Solution of Differential Equations](#solution-of-differential-equations)
  - [Equal Interval Interpolation Method](#equal-interval-interpolation-method)
    - [Theory](#equal-interval-interpolation-theory)
    - [Code](#equal-interval-interpolation-code)
    - [Input](#equal-interval-interpolation-input)
    - [Output](#equal-interval-interpolation-output)
  - [Second Order Derivative Method](#second-order-derivative-method)
    - [Theory](#second-order-derivative-theory)
    - [Code](#second-order-derivative-code)
    - [Input](#second-order-derivative-input)
    - [Output](#second-order-derivative-output)
  - [Runge Kutta Method](#runge-kutta-method)
    - [Theory](#runge-kutta-theory)
    - [Code](#runge-kutta-code)
    - [Input](#runge-kutta-input)
    - [Output](#runge-kutta-output)
  - [Numerical Differentiation Method](#numerical-differentiation-method)
    - [Theory](#numerical-differentiation-theory)
    - [Code](#numerical-differentiation-code)
    - [Input](#numerical-differentiation-input)
    - [Output](#numerical-differentiation-output)

- [Solution of Numerical Integrations](#solution-of-numerical-integrations)
  - [Simpson's One-Third Rule](#simpsons-one-third-rule)
    - [Theory](#simpsons-one-third-rule-theory)
    - [Code](#simpsons-one-third-rule-code)
    - [Input](#simpsons-one-third-rule-input)
    - [Output](#simpsons-one-third-rule-output)
  - [Simpson's Three-Eighths Rule](#simpsons-three-eighths-rule)
    - [Theory](#simpsons-three-eighths-rule-theory)
    - [Code](#simpsons-three-eighths-rule-code)
    - [Input](#simpsons-three-eighths-rule-input)
    - [Output](#simpsons-three-eighths-rule-output)

- [Numerical Differentiation](#numerical-differentiation)
  - [Numerical Differentiation by Forward Interpolation Method](#numerical-differentiation-by-forward-interpolation-method)
    - [Theory](#numerical-differentiation-by-forward-interpolation-theory)
      - [Intro](#numerical-differentiation-by-forward-interpolation-introduction)
      - [Formula](#numerical-differentiation-by-forward-interpolation-formula)
      - [Algorithm](#numerical-differentiation-by-forward-interpolation-algorithm-steps)
      - [Application](#numerical-differentiation-by-forward-interpolation-application)
    - [Code](#numerical-differentiation-by-forward-interpolation-code)
    - [Input](#numerical-differentiation-by-forward-interpolation-input)
    - [Output](#numerical-differentiation-by-forward-interpolation-output)
  - [Numerical Differentiation by Backward Interpolation Method](#numerical-differentiation-by-backward-interpolation-method)
    - [Theory](#numerical-differentiation-by-backward-interpolation-theory)
      - [Intro](#numerical-differentiation-by-backward-interpolation-introduction)
      - [Formula](#numerical-differentiation-by-backward-interpolation-formula)
      - [Algorithm](#numerical-differentiation-by-backward-interpolation-algorithm-steps)
      - [Application](#numerical-differentiation-by-backward-interpolation-application)
    - [Code](#numerical-differentiation-by-backward-interpolation-code)
    - [Input](#numerical-differentiation-by-backward-interpolation-input)
    - [Output](#numerical-differentiation-by-backward-interpolation-output)    
  


---

### Solution of Linear Equations

### Gauss Elimination Method

#### Gauss Elimination Theory
##### Gauss Elimination Introduction
The Gauss Elimination Method is a direct numerical method used to solve a system of linear equations:
Ax=b

where:
A = coefficient matrix
x = vector of unknowns
b = constant vector
The main idea is to transform the coefficient matrix into an upper triangular form using elementary row operations. Once in triangular form, the unknown variables are calculated using back substitution.
##### Gauss Elimination Formula
```python
For a system of 3 equations:

a₁₁x₁ + a₁₂x₂ + a₁₃x₃ = b₁  
a₂₁x₁ + a₂₂x₂ + a₂₃x₃ = b₂  
a₃₁x₁ + a₃₂x₂ + a₃₃x₃ = b₃  

The augmented matrix is formed as:

[A | b] =
⎡ a₁₁  a₁₂  a₁₃ | b₁ ⎤  
⎢ a₂₁  a₂₂  a₂₃ | b₂ ⎥  
⎣ a₃₁  a₃₂  a₃₃ | b₃ ⎦  

Elementary row operations are applied to eliminate the entries below the main diagonal, resulting in an upper triangular matrix.

After elimination, the unknowns are obtained by back substitution:

x₃ = b₃ / a₃₃  
x₂ = (b₂ − a₂₃x₃) / a₂₂  
x₁ = (b₁ − a₁₂x₂ − a₁₃x₃) / a₁₁
```
##### Gauss Elimination Algorithm Steps
Step 1: Form the augmented matrix [A∣b] from the system of equations.
Step 2: Apply elementary row operations to eliminate variables below the main diagonal, converting A into an upper triangular matrix.
Step 3: Once the matrix is in upper triangular form, start back substitution from the last equation to find the unknowns:
x_n,x_(n-1),…,x_1

Step 4: Continue until all variables are determined.
Step 5: Verify the solution by substituting the values back into the original equations.
##### Gauss Elimination Application
##### Used for:
Solving systems of linear algebraic equations
Ax=b

Large systems where direct inversion is inefficient


#### Gauss Elimination Code
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n;
    fin >> n;
    vector<vector<double>> a(n, vector<double>(n + 1));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n + 1; j++)
            fin >> a[i][j];

    for (int i = 0; i < n; i++) {
        if (fabs(a[i][i]) < 1e-6) {
            bool swapped = false;
            for (int j = i + 1; j < n; j++) {
                if (fabs(a[j][i]) > 1e-6) {
                    swap(a[i], a[j]);
                    swapped = true;
                    break;
                }
            }
            if (!swapped) {
                fout << "Matrix is singular" << endl;
                return 0;
            }
        }

        for (int j = i + 1; j < n; j++) {
            double f = a[j][i] / a[i][i];
            for (int k = i; k < n + 1; k++)
                a[j][k] -= a[i][k] * f;
        }
    }

    fout << "Upper Triangular Matrix:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n + 1; j++) {
            if (fabs(a[i][j]) < 1e-6)
                fout << "0 ";
            else
                fout << fixed << setprecision(3) << a[i][j] << " ";
        }
        fout << endl;
    }

    bool allzero;
    for (int i = 0; i < n; i++) {
        allzero = true;
        for (int j = 0; j < n; j++) {
            if (fabs(a[i][j]) > 1e-6) {
                allzero = false;
                break;
            }
        }
        if (allzero && fabs(a[i][n]) > 1e-6) {
            fout << "No solution" << endl;
            return 0;
        } else if (allzero && fabs(a[i][n]) < 1e-6) {
            fout << "Infinite solution" << endl;
            return 0;
        }
    }

    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = a[i][n];
        for (int j = i + 1; j < n; j++)
            sum -= a[i][j] * x[j];
        x[i] = sum / a[i][i];
    }

    for (int i = 0; i < n; i++)
        fout << "x" << i + 1 << " = " << fixed << setprecision(6) << x[i] << endl;

    fin.close();
    fout.close();
    return 0;
}
```

#### Gauss Elimination Input
```
3
2 1 -1 8
-3 -1 2 -11
-2 1 2 -3
```
#### Input Format:
```
## Input Description

3  → Number of variables (order of the system)

2 1 -1 8  
→ Coefficients of the first equation and its constant term  
→ (2x₁ + x₂ − x₃ = 8)

-3 -1 2 -11  
→ Coefficients of the second equation and its constant term  
→ (−3x₁ − x₂ + 2x₃ = −11)

-2 1 2 -3  
→ Coefficients of the third equation and its constant term  
→ (−2x₁ + x₂ + 2x₃ = −3)

```

#### Gauss Elimination Output
```
Upper Triangular Matrix:
2.000 1.000 -1.000 8.000 
0 0.500 0.500 1.000 
0 0 -1.000 1.000 
x1 = 2.000000
x2 = 3.000000
x3 = -1.000000

```
#### [Back to Contents](#table-of-contents)
---

### Gauss Jordan Elimination Method

#### Gauss Jordan Theory
#### Gauss Jordan Introduction
The Gauss–Jordan method is a numerical technique used to solve a system of linear equations. It is an extension of the Gauss Elimination method. Unlike Gauss Elimination, which reduces the coefficient matrix to an upper triangular form and requires back substitution, the Gauss–Jordan method reduces the coefficient matrix completely to a diagonal (identity) matrix.
The main advantage of this method is that it directly provides the solution without any further computation once the augmented matrix is in reduced form. It is particularly useful for solving multiple systems of equations with the same coefficient matrix or for finding the inverse of a matrix.
A system of linear equations:
a₁₁x₁ + a₁₂x₂ + ⋯ + a₁ₙxₙ = b₁  
a₂₁x₁ + a₂₂x₂ + ⋯ + a₂ₙxₙ = b₂  
⋮  
aₙ₁x₁ + aₙ₂x₂ + ⋯ + aₙₙxₙ = bₙ
can be written in augmented matrix form:
[A | b] =

⎡ a₁₁  a₁₂  ⋯  a₁ₙ | b₁ ⎤  
⎢ a₂₁  a₂₂  ⋯  a₂ₙ | b₂ ⎥  
⎢  ⋮     ⋮   ⋱   ⋮  | ⋮ ⎥  
⎣ aₙ₁  aₙ₂  ⋯  aₙₙ | bₙ ⎦


#### Gauss Jordan Formula
The main principle is to apply **elementary row operations** to reduce the augmented matrix into the identity matrix:

[A | b]  ⟶  [I | x]

Where:
- `I` is the identity matrix of size `n × n`
- `x = [x₁, x₂, ⋮, xₙ]ᵀ` is the solution vector

Elementary Row Operations include:
- Swap two rows: `Rᵢ ↔ Rⱼ`
- Multiply a row by a nonzero scalar: `Rᵢ → kRᵢ`
- Add/subtract a multiple of one row to another: `Rᵢ → Rᵢ + kRⱼ`

Direct formula for each pivot step:
- `Rᵢ → Rᵢ / aᵢᵢ`   (make pivot = 1)  
- `Rⱼ → Rⱼ − aⱼᵢ Rᵢ`   (make other elements in column = 0)
#### Gauss Jordan Algorithm steps
Form the augmented matrix: Write the system [A∣b].
Make the leading diagonal 1: For each row i, divide the row by the pivot element a_ii.
Eliminate other elements in the pivot column: For each row j≠i, make the elements in the pivot column 0 using the operation:
R_j→R_j-a_ji R_i

Repeat for all pivot elements: Apply steps 2 and 3 sequentially for all rows/columns.
Obtain the solution: Once the augmented matrix is in the form [I∣x], the solution vector xis directly obtained from the last column.
#### Gauss Jordan Application
#### Used for:
•	Solving systems of linear equations
•	Finding matrix inverse using row operations

#### Gauss Jordan Code
```cpp
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


```

#### Gauss Jordan Input
```
3
2 1 -3
1 3 2 
3 -2 4 
9 5 1
```
#### Input Format;
```
3  
→ Number of equations / variables (`n`)

2 1 -3  
→ Coefficients of the first equation  
→ (2x₁ + x₂ − 3x₃)

1 3 2  
→ Coefficients of the second equation  
→ (x₁ + 3x₂ + 2x₃)

3 -2 4  
→ Coefficients of the third equation  
→ (3x₁ − 2x₂ + 4x₃)

9 5 1  
→ Constants of the equations  
→ Right-hand side vector **b**

Corresponding System of Equations:

2x₁ + x₂ − 3x₃ = 9  
x₁ + 3x₂ + 2x₃ = 5  
3x₁ − 2x₂ + 4x₃ = 1  

This input represents a system of **three linear equations with three unknowns**, solved using the **Gauss–Jordan Elimination method** to obtain the **Reduced Row Echelon Form (RREF)** and the **unique solution**.

```

#### Gauss Jordan Output
```
Reduced Row Echelon Form of the Augmented Matrix:
     1.000      0.000      0.000  |      2.463
     0.000      1.000      0.000  |      1.433
    -0.000     -0.000      1.000  |     -0.881

Solution for each variable rounded to 3 decimal places:
x[1] = 2.463
x[2] = 1.433
x[3] = -0.881

```
#### [Back to Contents](#table-of-contents)
---

### LU Decomposition Method

#### LU Decomposition Theory
#### LU Decomposition Introduction
LU Decomposition is a matrix factorization technique used to solve systems of linear equations efficiently. In this method, a square matrix A is decomposed into the product of two triangular matrices:

A = L * U

Where:
- L is a lower triangular matrix (all elements above the diagonal are 0)
- U is an upper triangular matrix (all elements below the diagonal are 0)

Once A is decomposed, the system Ax = b can be solved in two steps using an intermediate vector y:

Ly = b
Ux = y

LU Decomposition is particularly efficient when the same coefficient matrix is used to solve multiple systems with different right-hand side vectors.

#### LU Decomposition Formula
Matrix decomposition:

A = L * U

Where:

L = 
| 1     0     0   ...  0   |
| l21   1     0   ...  0   |
| l31  l32    1   ...  0   |
| ...   ...  ...  ...  ...  |
| ln1  ln2  ln3  ...  1    |

U =
| u11  u12  u13  ...  u1n |
| 0    u22  u23  ...  u2n |
| 0     0   u33  ...  u3n |
| ...  ...  ...  ...  ... |
| 0     0    0   ...  unn |

Forward substitution (solve Ly = b):

y1 = b1
y2 = b2 - l21*y1
y3 = b3 - l31*y1 - l32*y2
...
yn = bn - ln1*y1 - ln2*y2 - ... - ln(n,n-1)*y(n-1)

Back substitution (solve Ux = y):

xn = yn / unn
x(n-1) = (y(n-1) - u(n-1,n)*xn) / u(n-1,n-1)
...
x1 = (y1 - u12*x2 - u13*x3 - ... - u1n*xn) / u11
#### LU Decomposition Algorithm Steps
1. Decompose the matrix A into L and U using elimination steps.  
2. Solve Ly = b using forward substitution.  
3. Solve Ux = y using back substitution.  
4. The solution vector x is obtained from step 3.
#### LU Decomposition Application
**Used For :**  
LU Decomposition is used for:  
- Solving systems of linear equations efficiently  
- Computing the inverse of a matrix  
- Determining the determinant of a matrix  
- Problems where repeated solutions with the same coefficient matrix but different constants are needed  
- Simplifying matrix computations in numerical analysis and engineering applications  

**Suitable For :**  
LU Decomposition is suitable for:  
- Square matrices (n × n)  
- Matrices that are non-singular and preferably well-conditioned  
- Problems requiring multiple solutions with the same coefficient matrix  
- Situations where computational efficiency is important over direct methods like Gaussian elimination


#### LU Decomposition Code
```cpp
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
```

#### LU Decomposition Input
```
2
1 1 2
2 2 4
2
1 1 2
2 2 5
3
2 3 -1 5
4 1 2 6
-2 5 3 12
```
#### Input Format:
```
n
a11 a12 ... a1n b1
a21 a22 ... a2n b2
...
an1 an2 ... ann bn
(repeated for multiple test cases)
```
**Where:**
- `n` is the number of variables / equations  
- Each of the next `n` lines represents one linear equation  
- `aij` are the coefficients of the variables  
- `bi` is the constant term  
- The input represents an **augmented matrix of size n × (n+1)**  
- Multiple systems can be provided sequentially in the same input file


#### LU Decomposition Output
```
The system has infinite solutions.

The system has no solution.

Lower Matrix L:
     1.000      0.000      0.000 
     0.500      1.000      0.000 
    -0.500      0.778      1.000 
Upper Matrix U:
     4.000      1.000      2.000 
     0.000      4.500      2.000 
     0.000      0.000     -1.556 
Solution:
x1 = 1.250
x2 = 2.286
x3 = -0.643
The system has a unique solution.


```
#### [Back to Contents](#table-of-contents)
---

### Matrix Inversion

#### Matrix Inversion Theory
#### Matrix Inversion Introduction
Matrix inversion is the process of finding a matrix `A⁻¹` such that multiplying it with the original matrix `A` gives the identity matrix `I`.  
It is a fundamental concept in linear algebra, used to solve systems of equations, analyze transformations, and support applications in engineering, physics, and computer science.

#### Matrix Inversion Formula
A square matrix `A` is invertible if and only if its determinant is non-zero:

det(A) ≠ 0

The inverse of matrix `A` is given by:

A⁻¹ = (1 / det(A)) · adj(A)

Where:
- `adj(A)` is the **adjoint** (transpose of the cofactor matrix) of `A`.

#### Matrix Inversion Steps
1. Compute the determinant of matrix `A`.  
   - If `det(A) = 0`, the inverse does not exist.  
2. Form the cofactor matrix by calculating cofactors of each element.  
3. Take the transpose of the cofactor matrix → this gives the adjoint matrix.  
4. Divide the adjoint matrix by `det(A)` → this gives `A⁻¹`.  
5. Verify by checking that `A · A⁻¹ = I`.

#### Matrix Inversion Application
- **Solving systems of linear equations**: For `Ax = b`, the solution is `x = A⁻¹b`.  
- **Computing determinants indirectly**: Inverse methods can simplify certain determinant calculations.  
- **Engineering & Physics**: Used in control systems, circuit analysis, and mechanics.  
- **Computer Science**: Applied in graphics transformations, optimization problems, and machine learning algorithms.  

#### Matrix Inversion Code
```cpp
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

```

#### Matrix Inversion Input
```
3
2 1 1
1 3 2
1 2 2
4 7 5
```
#### Input Format:
```
n
a11 a12 a13 ... a1n
a21 a22 a23 ... a2n
...
an1 an2 an3 ... ann
b1 b2 b3 ... bn
```

#### Matrix Inversion Output
```
Here N = 3

Determinant of A: 3.0000
Inverse exists.

 TRANSPOSE MATRIX
    2.000000     1.000000     1.000000 
    1.000000     3.000000     2.000000 
    1.000000     2.000000     2.000000 

FINAL INVERSE MATRIX
    0.666667    -0.000000    -0.333333 
   -0.000000     1.000000    -1.000000 
   -0.333333    -1.000000     1.666667 
The solution for Ax=b is :
x[1] = 1.0000
x[2] = 2.0000
x[3] = 0.0000

```
#### [Back to Contents](#table-of-contents)
---

### Solution of Non-Linear Equations

### Bisection Method

#### Bisection Theory

#### Bisection Introduction
#### Bisection Introduction
The Bisection Method is a numerical root-finding technique used to determine an approximate solution of the equation `f(x) = 0`.

It is a **bracketing method**, which means it requires two initial guesses that enclose a root.  
If a function `f(x)` is continuous in the interval `[a, b]` and `f(a) · f(b) < 0`, then, according to the **Intermediate Value Theorem**, at least one root lies within that interval.  

The method works by repeatedly dividing the interval into two equal halves and selecting the subinterval in which the sign change occurs. It is simple, reliable, and always converges, though it may be slower compared to other methods.

---

#### Bisection Formula
The midpoint of the interval [aⓜ,b]is calculated using the formula:
c=(a+b)/2

Where:
a= left endpoint
b= right endpoint
c= midpoint (approximate root)
The interval is updated based on the sign of the function:
If f(a)⋅f(c)<0, then set b=c
Otherwise, set a=c
This process continues until the desired accuracy is achieved.
#### Bisection Algorithm Steps
Step 1: Choose initial values aand bsuch that
f(a)⋅f(b)<0

Step 2: Compute the midpoint
c=(a+b)/2

Step 3: Evaluate f(c)
Step 4:
If f(a)⋅f(c)<0, set b=c
Else, set a=c
Step 5: Repeat Steps 2–4 until
∣b-a∣<ε

where ε is the allowable error.
Step 6: The value of cis taken as the approximate root.
#### Bisection Application
#### Used for:
The Bisection Method is used for:
	Finding real roots of nonlinear equations f(x)=0
	Solving equations where the function is continuous over a given interval
	Obtaining a rough or initial approximation of a root
	Situations where guaranteed convergence is more important than speed
	Educational purposes to understand the basic concept of numerical root-finding
#### Suitable For:
The Bisection Method is suitable for:
	Functions that are continuous in the interval [aⓜ,b]
	Problems where two initial guesses can be chosen such that
f(a)⋅f(b)<0

	Cases where the root lies near the boundary of the interval
	Problems requiring simple and stable algorithms
	Situations where derivatives are not available or difficult to compute

#### Bisection Code
```cpp
#include <bits/stdc++.h>
using namespace std;

vector<double> polyCoeff;
int degree;

// Function to evaluate the polynomial
double evaluatePolynomial(double x)
{
    double result = 0.0;
    for (int i = 0; i <= degree; i++)
    {
        result += polyCoeff[i] * pow(x, i);
    }
    return result;
}

int main()
{
    ifstream inputFile("input.txt");
    ofstream outputFile("output.txt");

    outputFile << "=== Bisection Method ===" << endl;

    // Read polynomial degree
    inputFile >> degree;
    polyCoeff.resize(degree + 1);

    // Read coefficients
    for (int i = 0; i <= degree; i++)
    {
        inputFile >> polyCoeff[i];
    }

    // Print the polynomial equation
    outputFile << "Equation is: ";
    bool firstTerm = true;

    for (int i = degree; i >= 0; i--)
    {
        if (polyCoeff[i] == 0) continue;

        if (!firstTerm)
            outputFile << (polyCoeff[i] > 0 ? " + " : " - ");
        else
        {
            if (polyCoeff[i] < 0) outputFile << "-";
            firstTerm = false;
        }

        if (fabs(polyCoeff[i]) != 1.0 || i == 0)
            outputFile << fabs(polyCoeff[i]);

        if (i > 0) outputFile << "x";
        if (i > 1) outputFile << "^" << i;
    }

    outputFile << " = 0" << endl;

    // Read interval, step size and tolerance
    double intervalStart, intervalEnd, stepSize, tolerance;
    inputFile >> intervalStart >> intervalEnd;
    inputFile >> stepSize;
    inputFile >> tolerance;

    vector<pair<double, double>> rootIntervals;

    // Find intervals containing roots
    for (double x = intervalStart; x < intervalEnd; x += stepSize)
    {
        double fx = evaluatePolynomial(x);
        double fxNext = evaluatePolynomial(x + stepSize);

        // Exact root check
        if (fabs(fx) < 1e-6)
        {
            rootIntervals.push_back({x, x});
        }
        // Sign change check
        else if (fx * fxNext < 0)
        {
            rootIntervals.push_back({x, x + stepSize});
        }
    }

    outputFile << "Intervals containing roots: ";
    for (auto interval : rootIntervals)
    {
        outputFile << "{" << interval.first << "," << interval.second << "}  ";
    }
    outputFile << endl;

    // Apply Bisection Method
    vector<double> roots;
    outputFile << "Roots are:" << endl;

    for (auto interval : rootIntervals)
    {
        double left = interval.first;
        double right = interval.second;
        double mid;

        // If exact root already found
        if (left == right)
        {
            roots.push_back(left);
            continue;
        }

        while (fabs(right - left) > tolerance)
        {
            mid = (left + right) / 2.0;

            if (fabs(evaluatePolynomial(mid)) < 1e-6)
                break;
            else if (evaluatePolynomial(mid) * evaluatePolynomial(left) < 0)
                right = mid;
            else
                left = mid;
        }

        roots.push_back(mid);
    }

    // Print all roots
    for (int i = 0; i < roots.size(); i++)
    {
        outputFile << "Root " << i + 1 << " = "
                   << fixed << setprecision(2) << roots[i] << endl;
    }

    inputFile.close();
    outputFile.close();
    return 0;
}
```

#### Bisection Input
```
3
-6 11 -6 1
0 4
0.5
0.001
```
#### Input Format:
- `degree` → The degree of the polynomial.  
  Example: `3` means a cubic polynomial.
- `c0 c1 ... c_degree` → Coefficients of the polynomial starting from the constant term up to the highest degree term.  
  Example: `-6 11 -6 1` represents the polynomial:  
  `-6 + 11x − 6x² + 1x³ = 0`
- `intervalStart intervalEnd` → The search interval for roots.  
  Example: `0 4` means roots will be searched between `x = 0` and `x = 4`.
- `stepSize` → The increment used to scan the interval for sign changes.  
  Example: `0.5` means the interval is checked in steps of 0.5.
- `tolerance` → The stopping criterion for the bisection iterations (accuracy of the root).  
  Example: `0.001` means the algorithm stops when the interval size is less than 0.001.


#### Bisection Output
```
Equation is: x^3 - 6x^2 + 11x - 6 = 0
Intervals containing roots: {1,1}  {2,2}  {2.5,3}  {3,3}  
Roots are:
Root 1 = 1.00
Root 2 = 2.00
Root 3 = 3.00
Root 4 = 3.00

```
#### [Back to Contents](#table-of-contents)
---

### False Position Method

#### False Position Theory
#### False Position Introduction

The False Position Method, also known as Regula Falsi, is a bracketing technique used to find roots of nonlinear equations.  
It improves upon the Bisection Method by using **linear interpolation** between two points instead of simply taking the midpoint.  
If `f(a) · f(b) < 0`, it indicates that a root lies between `a` and `b`. The method then uses the x-intercept of the line joining `(a, f(a))` and `(b, f(b))` as the next approximation.

---

#### False Position Formula

The formula used to compute the next approximation `x` is:

x = (a·f(b) − b·f(a)) / (f(b) − f(a))

This gives the x-intercept of the straight line connecting the points `(a, f(a))` and `(b, f(b))`.

---

#### False Position Algorithm Steps

1. Choose two initial guesses `a` and `b` such that `f(a) · f(b) < 0`.
2. Compute the next approximation `x` using:

  x = (a·f(b) − b·f(a)) / (f(b) − f(a))

3. Evaluate `f(x)`:
   - If `f(a) · f(x) < 0`, set `b = x`.
   - If `f(b) · f(x) < 0`, set `a = x`.
4. Repeat steps 2–3 until the value of `x` converges within a desired tolerance.

---

#### False Position Application

- **Root finding in nonlinear equations** where bracketing is possible.
- Commonly used in **engineering problems**, such as fluid mechanics, thermodynamics, and electrical circuits.
- Useful in **physics** for solving transcendental equations.
- Applied in **numerical analysis** when a reliable bracketing method is preferred over open methods like Newton-Raphson.

⚠️ Note: While more accurate than Bisection in many cases, False Position may converge slowly if one side of the bracket remains fixed for many iterations.

#### False Position Code
```cpp
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

```

#### False Position Input
```
1 2
0.0001
100

```
#### Input Format;
- `a b` → The initial interval endpoints.  
  Example: `1 2` means the algorithm will search for a root between `x = 1` and `x = 2`.
- `tolerance` → The acceptable error threshold for stopping the iteration.  
  Example: `0.0001` means the algorithm stops when `|f(c)| < 0.0001`.
- `maxIterations` → The maximum number of iterations allowed.  
  Example: `100` means the algorithm will perform at most 100 iterations if convergence is not reached earlier.


#### False Position Output
```
Iter	 a		 b		 c		 f(c)
1	 1.000000	 2.000000	 1.333333	 -0.962963
2	 1.333333	 2.000000	 1.462687	 -0.333339
3	 1.462687	 2.000000	 1.504019	 -0.101818
4	 1.504019	 2.000000	 1.516331	 -0.029895
5	 1.516331	 2.000000	 1.519919	 -0.008675
6	 1.519919	 2.000000	 1.520957	 -0.002509
7	 1.520957	 2.000000	 1.521258	 -0.000725
8	 1.521258	 2.000000	 1.521344	 -0.000209
9	 1.521344	 2.000000	 1.521370	 -0.000060

Root = 1.521370

```
#### [Back to Contents](#table-of-contents)
---

### Secant Method

#### Secant Theory
#### Secant Introduction
The Secant Method is an **open root-finding technique** that does not require the function to change sign within an interval.  
It is similar to the Newton–Raphson method but instead of using the derivative, it approximates the slope of the tangent by using a **secant line** through two previous approximations.  
This makes the method useful when the derivative of the function is difficult or expensive to compute.

---

#### Secant Formula
The iterative formula for the Secant Method is:

xₙ₊₁ = xₙ − f(xₙ) · (xₙ − xₙ₋₁) / (f(xₙ) − f(xₙ₋₁))

Where:
- `xₙ` and `xₙ₋₁` are the two most recent approximations.
- `f(x)` is the function whose root is being sought.

---

#### Secant Steps
1. Choose two initial guesses `x₀` and `x₁`.
2. Compute the next approximation using the secant formula:

   xₙ₊₁ = xₙ − f(xₙ) · (xₙ − xₙ₋₁) / (f(xₙ) − f(xₙ₋₁))
   
4. Update the values:
   - Replace `xₙ₋₁` with `xₙ`.
   - Replace `xₙ` with `xₙ₊₁`.
5. Repeat the process until the difference between successive approximations is less than the desired tolerance (convergence).

---

#### Secant Application
- **Root finding** in nonlinear equations where derivatives are difficult to compute.
- Used in **engineering problems**, such as solving equations in fluid mechanics, thermodynamics, and electrical circuits.
- Applied in **physics** for transcendental equations and iterative models.
- Useful in **numerical analysis** as a faster alternative to bracketing methods like Bisection or False Position.
- ⚠️ Note: Convergence is not guaranteed for all functions, but when it converges, it is often faster than bracketing methods.

#### Secant Code
```cpp
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
```

#### Secant Input
```
4
1 0 -5 0 4
4 
1 -3 2 6 0
```
#### Input Format;
```
- `degree` → The degree of the polynomial.  
  Example: `4` means a quartic polynomial.
- `c0 c1 ... c_degree` → Coefficients of the polynomial starting from the highest degree term down to the constant term.  
  - `1 0 -5 0 4` represents:  
    `x⁴ − 5x² + 4 = 0`
  - `1 -3 2 6 0` represents:  
    `x⁴ − 3x³ + 2x² + 6x = 0`
- Multiple polynomials can be provided one after another in the same `input.txt`.  
  The program processes each polynomial separately, detects intervals with sign changes, and applies the **Secant Method** to approximate roots.

```

#### Secant Output
```
Polynomial degree 4
Coefficients: 1 0 -5 0 4 
Intervals detected: 4
Root ≈ -2.000000  in 2 iterations
Root ≈ -1.000000  in 2 iterations
Root ≈ 1.000000  in 2 iterations
Root ≈ 2.000000  in 2 iterations
-----------------------------------
Polynomial degree 4
Coefficients: 1.000000 -3.000000 2.000000 6.000000 0.000000 
Intervals detected: 2
Root ≈ -1.000000  in 2 iterations
Root ≈ -0.000000  in 2 iterations
-----------------------------------

```
#### [Back to Contents](#table-of-contents)
---

### Newton Raphson Method

#### Newton Raphson Theory
#### Newton Raphson Introduction

The Newton–Raphson Method is an efficient **open root-finding technique** that uses the tangent of a function to approximate its root.  
Starting from an initial guess, the method iteratively refines the approximation by following the tangent line at each point to its x-intercept.  
It is widely used due to its fast convergence when the initial guess is close to the actual root.

---

#### Newton Raphson Formula

The iterative formula for the Newton–Raphson Method is:

xₙ₊₁ = xₙ − f(xₙ) / f′(xₙ)

Where:
- `xₙ` is the current approximation.
- `f(xₙ)` is the function value.
- `f′(xₙ)` is the derivative of the function at `xₙ`.

---

#### Newton Raphson Steps

1. Choose an initial guess `x₀`.
2. Evaluate `f(xₙ)` and `f′(xₙ)`.
3. Compute the next approximation using:

  xₙ₊₁ = xₙ − f(xₙ) / f′(xₙ)

4. Repeat the process until the difference between successive approximations is less than the desired tolerance.

⚠️ **Note**: The method requires that `f′(xₙ)` is not zero and that the function is differentiable near the root.

---

#### Newton Raphson Application

- **Solving nonlinear equations** in engineering, physics, and applied mathematics.
- Used in **optimization problems** where finding critical points is essential.
- Applied in **computer science** for numerical methods and algorithmic root-finding.
- Common in **control systems**, **mechanics**, and **electrical circuit analysis**.
- ⚠️ May fail or diverge if the initial guess is poor or if the derivative is zero or undefined near the root.

#### Newton Raphson Code
```cpp
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace std;


const float TOLERANCE = 0.0001;
const int MAX_ITER = 100;

float f(float x)
{
    return (3 * x - cos(x) - 1);
}

float df(float x) //derivative of the func
{
    return (3 + sin(x));
}

float newton_raphson_solve(float initial_guess, int& iterations_needed) {
    float xn = initial_guess;
    float xn_1 = initial_guess;
    iterations_needed = 0;

    for (int n_iter = 0; n_iter < MAX_ITER; ++n_iter) {
        float derivative_val = df(xn);

        if (fabs(derivative_val) < 1e-6) {
            return NAN;
        }


        xn_1 = xn - (f(xn) / derivative_val);

        if (fabs(xn_1 - xn) < TOLERANCE) {
            iterations_needed = n_iter + 1;
            return xn_1;
        }

        xn = xn_1;
    }

    return NAN;
}


int main(){
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    if (!fin.is_open() || !fout.is_open()) {
        cerr << "Error! Could not open input/output files." << endl;
        return 1;
    }

    vector<float> v;
    float x0;
    float j = 1.0f;

    if (!(fin >> j)) {

        fout << "Error: Failed to read search step size from input.txt. Using default j=1.0." << endl;
    }

     for(float i = -200.0f; i < 200.0f; i += j)
    {
        if(f(i) * f(i + j) < 0)
        {
            v.push_back(i);
        }
    }


    if(v.empty()){

        if (fin >> x0) {
            v.push_back(x0);
            fout << "WARNING: No sign-change interval found. Using fallback guess x0 = " << x0 << endl;
        } else {
            fout << "ERROR: No sign-change interval found and no fallback x0 provided in input.txt." << endl;
            fin.close();
            fout.close();
            return 1;
        }
    }


    fout << fixed << setprecision(10);

    fout << "Function: 3x - cos(x) - 1 = 0" << endl;
    fout << "Tolerance: " << TOLERANCE << endl;
    fout << "Search Step: " << j << endl;

    fout << "Initial Guesses : ";
    for(auto p : v)
    {
        fout << "{" << p << "}" << " ";
    }
    fout << endl;

    int root_count = 0;

    for(auto it : v)
    {
        float xn = it;
        float xn_1 = 0.0f;
        int n_iterations = 0;
        float root_result = newton_raphson_solve(it, n_iterations);

        root_count++;
        float right_endpoint = it + j;

        fout << "Root " << root_count << ":" << endl;
        fout << "  Search Interval = [" << it <<","<<right_endpoint<<"]"<< endl;

        if (isnan(root_result)) {
            fout << "  Root Value = FAILED." << endl;
            fout << "  Iteration needed for the root " << root_count << " = >" << MAX_ITER << endl;
        } else {
            fout << "  Root Value= " << root_result << endl;
            fout << "  Iteration needed for the root " << root_count << " = " << n_iterations << endl;
        }
    }

    fout << "Total roots attempted: " << root_count << endl;


    fin.close();
    fout.close();

    return 0;
}

```

#### Newton Raphson Input
```
3 -2 5 -4 1 0.8 2.5

```
#### Input Format;
```
- `degree` → The degree of the polynomial.  
  Example: `3` means a cubic polynomial.
- `c0 c1 ... c_degree` → Coefficients of the polynomial starting from the highest degree term down to the constant term.  
  Example: `-2 5 -4 1` represents the polynomial:  
  `-2x³ + 5x² − 4x + 1 = 0`
- `stepSize` → The search step used to scan for sign changes in the interval.  
  Example: `0.8` means the program checks in increments of 0.8.
- `initialGuess` → A fallback starting point for Newton–Raphson if no sign-change interval is found.  
  Example: `2.5` means the algorithm will start iterations at `x = 2.5` if needed.

```

#### Newton Raphson Output
```
Function: 3x - cos(x) - 1 = 0
Tolerance: 0.0001000000
Search Step: 3.0000000000
Initial Guesses : {-2.0000000000} 
Root 1:
  Search Interval = [-2.0000000000,1.0000000000]
  Root Value= 0.6071016192
  Iteration needed for the root 1 = 4
Total roots attempted: 1

```
#### [Back to Contents](#table-of-contents)
---

### Solution of Interpolation

### Newton's Forward Interpolation Method

#### Newton's Forward Interpolation Theory
##### Newton's Forward Interpolation Introduction

Newton’s Forward Interpolation method is a numerical technique used to estimate the value of a function at a point when the independent variable values are equally spaced. It is particularly effective when the required value lies near the beginning of the data table.
The method uses forward differences of the function values to construct the interpolation polynomial.


##### Newton's Forward Interpolation Formula
```
Interpolation formula:

y = y0 + p*Δy0 + (p*(p-1)/2!)*Δ²y0 + (p*(p-1)*(p-2)/3!)*Δ³y0 + ...

where:

p = (x - x0) / h

Explanation of terms:

- y0   : The value of the function at the first data point.
- Δy0, Δ²y0, Δ³y0, ... : Forward differences of the function values.
- h    : Spacing between consecutive values of x.
- x    : The point at which interpolation is required.
- p    : Ratio of distance from the first data point in units of h.

This formula estimates y at x by successively adding terms involving forward differences,
where each term accounts for higher-order variations of the data.
```


##### Newton's Forward Interpolation Algorithm Steps

1. Arrange the data:
   Ensure that the independent variable values x0, x1, ..., xn are equally spaced.

2. Construct a forward difference table:
   Compute the forward differences Δy0, Δ²y0, Δ³y0, ... from the given data.

3. Calculate p:
   p = (x - x0) / h
   where x is the value at which interpolation is required
   and h is the spacing between consecutive x values.

4. Apply the interpolation formula:
   y = y0 + p*Δy0 + (p*(p-1)/2!)*Δ²y0 + (p*(p-1)*(p-2)/3!)*Δ³y0 + ...
   Substitute y0, Δy0, Δ²y0, ... and p to compute the interpolated value of y.

5. Evaluate higher-order terms if necessary:
   Include as many terms as needed for the desired accuracy.



##### Newton's Forward Interpolation Application
```
Uses:
- Estimating intermediate values
- Function approximation from tabulated data
- Numerical computations in engineering problems
- Scientific data analysis
```

#### Newton's Forward Interpolation Code

```python
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
        for (int i = 0; i < n - j; i++)
            diff[i][j] = diff[i+1][j-1] - diff[i][j-1];

    double value;
    fin >> value;
    double h = x[1] - x[0];
    double u = (value - x[0]) / h;
    double result = y[0];
    double term = 1.0;

    for (int i = 1; i < n; i++) {
        term *= (u - (i - 1));
        result += (term * diff[0][i]) / tgamma(i + 1);
    }

    fout << fixed << setprecision(6) << result << endl;
    return 0;
}
```
- [Newton's Forward Interpolation](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Interpolation/Newton%27s%20Forward%20Interpolation/code.cpp)

#### Newton's Forward Interpolation Input
```
5
0 1 2 3 4
1 2 4 8 16
2.5
```
```
5
→ Number of data points (n)

0 1 2 3 4
→ Values of the independent variable x (equally spaced)

1 2 4 8 16
→ Corresponding values of the dependent variable y

2.5
→ Value of x at which interpolation is required
```
#### Newton's Forward Interpolation Output
```
5.648438
```




### Newton's Backward Interpolation Method

#### Newton's Backward Interpolation Theory
##### Newton's Backward Interpolation Introduction

Newton’s Backward Interpolation method is used when the data points are equally spaced
and the required value lies near the end of the data table.
This method uses backward differences to estimate the value of the function.



##### Newton's Backward Interpolation Formula
```
y = yn + p*∇y_n + (p*(p+1)/2!)*∇²y_n + (p*(p+1)*(p+2)/3!)*∇³y_n + ...

where:

p = (x - xn) / h

Explanation of terms:

- yn    : The value of the function at the last data point.
- ∇y_n, ∇²y_n, ∇³y_n, ... : Backward differences of the function values.
- h     : Spacing between consecutive values of x.
- x     : The point at which interpolation is required.
- p     : Ratio of distance from the last data point in units of h.

```

##### Newton's Backward Interpolation Algorithm Steps
```
1. Arrange the data:
   Ensure that the independent variable values x0, x1, ..., xn are equally spaced.

2. Construct a backward difference table:
   Compute the backward differences ∇y_n, ∇²y_n, ∇³y_n, ... from the given data.

3. Calculate p:
   p = (x - xn) / h
   where x is the value at which interpolation is required
   and h is the spacing between consecutive x values.

4. Apply the interpolation formula:
   y = yn + p*∇y_n + (p*(p+1)/2!)*∇²y_n + (p*(p+1)*(p+2)/3!)*∇³y_n + ...
   Substitute yn, ∇y_n, ∇²y_n, ... and p to compute the interpolated value of y.

5. Evaluate higher-order terms if necessary:
   Include as many terms as needed for the desired accuracy.

```
##### Newton's Backward Interpolation Application
```

Uses:
- Estimating values near the end of datasets
- Predicting values close to the last data point
- Numerical analysis and table-based computation
```

#### Newton's Backward Interpolation Code
```python
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
```
- [Newton's Backward Interpolation](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Interpolation/Newton%27s%20Backward%20Interpolation/code.cpp)
#### Newton's Backward Interpolation Input
```
5
0 1 2 3 4
1 2 4 8 16
3.5
```
```
5
→ Number of data points (n)

0 1 2 3 4
→ Values of the independent variable x (equally spaced)

1 2 4 8 16
→ Corresponding values of the dependent variable y

3.5
→ Value of x at which interpolation is required (near the end of the table)

```
#### Newton's Backward Interpolation Output
```
11.313708

```



### divided difference  Method

#### divided difference Theory
##### divided difference Introduction
Newton’s Divided Difference Interpolation is used when the data points are unequally spaced.
It constructs an interpolation polynomial that passes through all the given data points.
This method uses divided differences to compute the coefficients of the polynomial.


##### divided difference Formula
```
P(x) = y0 
       + (x - x0) * f[x0, x1] 
       + (x - x0)*(x - x1) * f[x0, x1, x2] 
       + (x - x0)*(x - x1)*(x - x2) * f[x0, x1, x2, x3] 
       + ...

where:

- y0                  : The value of the function at the first data point.
- f[x0, x1], f[x0,x1,x2], ... : First, second, and higher-order divided differences.
- x                    : The point at which interpolation is required.

```

##### divided difference Steps
```
1. Arrange the data:
   List the data points (x0, y0), (x1, y1), ..., (xn, yn).

2. Construct a divided difference table:
   - Compute first-order divided differences: f[xi, xi+1] = (yi+1 - yi) / (xi+1 - xi)
   - Compute second-order divided differences: f[xi, xi+1, xi+2] = (f[xi+1, xi+2] - f[xi, xi+1]) / (xi+2 - xi)
   - Continue calculating higher-order divided differences as needed.

3. Form the interpolation polynomial:
   P(x) = y0 + (x - x0)*f[x0,x1] + (x - x0)*(x - x1)*f[x0,x1,x2] + ...

4. Evaluate P(x) at the required value of x:
   Substitute the computed divided differences and the value of x to get the interpolated value of y.

```

##### divided difference Application
```
Uses:
- Interpolation with irregular datasets
- Polynomial construction from scattered data
- Scientific and engineering calculations
- Function approximation without equal spacing
```

#### divided difference Code
```python
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
```
- [Newton's Divided Difference Interpolation](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Interpolation/Newton%27s%20Divided_difference/code.cpp)


#### divided difference Input
```
5
1 2 3 4 5
3 4 5 6 7
5.5
```
```
5
→ Number of data points (n)

0 1 2 3 4
→ Values of the independent variable x 

3 4 5 6 7
→ Corresponding values of the dependent variable y

5.5
→ Value of x at which interpolation is required 

```



#### divided difference Output
```
7.500000
```


### solution-of-curve-fitting-model

### least square regression method for linear equations
#### least square regression method for linear equations theory

##### least square regression method for linear equations Introduction
Linear regression is used to model the relationship between a dependent variable \(y\) and an independent variable \(x\) by fitting a straight line to the observed data.  
```
The line is represented as:

\[
y = a + bx
\]

Where:  
- \(a\) = y-intercept  
- \(b\) = slope of the line  
```
The objective is to find the best-fit line that minimizes the sum of squared errors between observed and predicted values.

---

#####  least square regression method for linear equations Formula 
```
The coefficients \(a\) and \(b\) are obtained from the **normal equations**:

\[
\sum y = n a + b \sum x
\]

\[
\sum xy = a \sum x + b \sum x^2
\]

Where:  
- \(n\) = number of data points  
- \(\sum x\) = sum of x-values  
- \(\sum y\) = sum of y-values  
- \(\sum xy\) = sum of product of x and y  
- \(\sum x^2\) = sum of squares of x-values  

Solving these equations gives the values of \(a\) and \(b\).
```
---


#####  least square regression method for linear equations steps
1. Input data points \((x_i, y_i)\) for \(i = 1, 2, ..., n\).  
2. Compute the sums: \(\sum x\), \(\sum y\), \(\sum xy\), \(\sum x^2\).  
3. Form the normal equations:  
   - \(\sum y = n a + b \sum x\)  
   - \(\sum xy = a \sum x + b \sum x^2\)  
4. Solve the equations simultaneously to find \(a\) and \(b\).  
5. The best-fit line is \(y = a + bx\).  
6. Optionally, calculate predicted \(y_i\) for each \(x_i\) to compare with observed values.

---
#####  least square regression method for linear equations application
```
Uses:
- Predicting values based on a straight-line trend
- Modeling relationships between two variables
- Data fitting in experimental and statistical analysis
- Estimating unknown values from observed data
- Simple forecasting problems
```
####  least square regression method for linear equations Code
```python
#include <bits/stdc++.h>
using namespace std;

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int n;
    fin >> n;

    vector<double> x(n), y(n);

    for (int i = 0; i < n; i++) fin >> x[i];
    for (int i = 0; i < n; i++) fin >> y[i];

    double sumx = 0, sumy = 0, sumxy = 0, sumx2 = 0;

    for (int i = 0; i < n; i++) {
        sumx += x[i];
        sumy += y[i];
        sumxy += x[i] * y[i];
        sumx2 += x[i] * x[i];
    }

    double a = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx);
    double b = (sumy - a * sumx) / n;

    fout << fixed << setprecision(2);
    fout << "Linear Regression Line: y = " << a <<  " x + " << b << "\n";

    fin.close();
    fout.close();

    return 0;
}
```
- [Linear Regression](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Curve%20Fitting/Linear%20Regression/linear.cpp)
####  least square regression method for linear equations Input
```
6
1 2 3 4 5 6
3 4 5 6 7 8

```
```
6
→ Number of data points (n)

1 2 3 4 5 6
→ Values of the independent variable x

3 4 5 6 7 8
→ Corresponding values of the dependent variable y

```
####  least square regression method for linear equations Output
```
Linear Regression Line: y = 1.00 x + 2.00

```








### least square regression method for transcendental equations
#### least square regression method for transcendental equations theory

##### least square regression method for transcendental equations Introduction
Transcendental equations are equations involving transcendental functions such as exponential, logarithmic, trigonometric, or combinations of these, e.g.,  
```
\[
f(x) = e^x - 3x = 0
\]
```
These equations cannot be solved analytically in most cases. Numerical methods are used to approximate the roots of the equation. Common methods include:  
- Bisection Method  
- False Position (Regula Falsi) Method  
- Newton-Raphson Method  
- Secant Method  

---

#####  least square regression method for transcendental equations Formula 
```
y=ae^bx
y=ax^b
equations are similar to linear equaton
```
#####  least square regression method for transcendental equations steps
1. Select an initial guess or interval depending on the method.  
2. Evaluate the function \(f(x)\) at required points.  
3. Apply the iterative formula specific to the chosen method.  
4. Check for convergence:  
   \(|x_{n+1} - x_n| < \text{tolerance}\) or \(|f(x_{n+1})| < \text{tolerance}\).  
5. Repeat steps 2–4 until convergence.  
6. Report \(x_{root}\) as the approximate solution.

---

#####  least square regression method for transcendental equations application
```
Uses:
- Population growth modeling
- Radioactive decay analysis
- Chemical reaction rate modeling
- Economic growth analysis
- Biological and physical processes
```
####  least square regression method for transcendental equations Code
```python
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
```
- [Transcendental Equation](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Curve%20Fitting/Transcendental%20Equation/exponential.cpp)
####  least square regression method for transcendental equations Input
```
5
1 2 3 4 5
50 80 96 120 145
6

```
```
5
→ Number of observations (n)

1 2 3 4 5
→ Values of the independent variable t

50 80 96 120 145
→ Corresponding values of the dependent variable T

6
→ Value of t at which T is to be estimated

```
####  least square regression method for transcendental equations Output
```
Computed parameters:
a = 5.7492444036
b = 41.0586716201

Estimated T(6.0000000000) = 189.7614442461

```















### least square regression method for Polynomial equations
#### least square regression method for Polynomial equations theory

##### least square regression method for Polynomial equations – Introduction
Polynomial regression is a statistical method used to fit a higher-degree polynomial to a set of data points. Unlike linear regression, which fits a straight line, polynomial regression can capture curvature in the data.  

The assumed form of the polynomial is:

y = a0 + a1*x + a2*x^2 + a3*x^3 + ... + an*x^n

Where:  
- a0, a1, a2, ..., an are the coefficients of the polynomial  
- n is the degree of the polynomial  

The coefficients are determined such that the sum of the squares of the differences between the observed values and the predicted values is minimized.

---

##### least square regression method for Polynomial equations – Formula
```
For a polynomial of degree n, the **normal equations** are:

Σy = n*a0 + a1*Σx + a2*Σx^2 + ... + an*Σx^n

Σ(x*y) = a0*Σx + a1*Σx^2 + a2*Σx^3 + ... + an*Σx^(n+1)

Σ(x^2*y) = a0*Σx^2 + a1*Σx^3 + a2*Σx^4 + ... + an*Σx^(n+2)

...

Σ(x^n*y) = a0*Σx^n + a1*Σx^(n+1) + a2*Σx^(n+2) + ... + an*Σx^(2n)

Solve these equations simultaneously to determine the coefficients a0, a1, ..., an.
```
---

##### least square regression method for Polynomial equations – steps
```
1. Collect the data points (x, y).  
2. Decide the degree n of the polynomial.  
3. Compute the required sums: Σx, Σx^2, ..., Σx^(2n), Σy, Σ(x*y), Σ(x^2*y), ..., Σ(x^n*y).  
4. Form the normal equations:

Σy = n*a0 + a1*Σx + a2*Σx^2 + ... + an*Σx^n  
Σ(x*y) = a0*Σx + a1*Σx^2 + a2*Σx^3 + ... + an*Σx^(n+1)  
Σ(x^2*y) = a0*Σx^2 + a1*Σx^3 + a2*Σx^4 + ... + an*Σx^(n+2)  
...  
Σ(x^n*y) = a0*Σx^n + a1*Σx^(n+1) + a2*Σx^(n+2) + ... + an*Σx^(2n)  

5. Solve the normal equations simultaneously to find the coefficients a0, a1, ..., an.  
6. Construct the regression polynomial:

y = a0 + a1*x + a2*x^2 + ... + an*x^n  

7. Use the polynomial to predict y for any given x.
 ```
##### least-square-regression-method-for-polynomial-equations-application
```
Uses:
- Modeling non-linear relationships
- Curve fitting in engineering and scientific data
- Approximating complex functions
- Data smoothing and interpolation
- Reducing error where linear regression fails
```
 #### least square regression method for Polynomial equations Code
```python
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
```
- [Polynomial Equation](https://github.com/Hirobi98/Numerical_methods_terminal_app/blob/main/Curve%20Fitting/Polynomial%20Equation/polynomial.cpp)
#### least square regression method for Polynomial equations Input
```
5 2
1 2 3 4 5
6 11 18 27 38

```
```
5 2
→ n = Number of data points
→ degree = Degree of the polynomial

1 2 3 4 5
→ Values of the independent variable x

6 11 18 27 38
→ Corresponding values of the dependent variable y

```
#### least square regression method for Polynomial equations Output
```
a0 = 3.000000
a1 = 2.000000
a2 = 1.000000

```


### Solution of Differential Equations

### Equal Interval Interpolation Method

#### Equal Interval Interpolation Theory
[Add your theory content here]

#### Equal Interval Interpolation Code
```python
# Add your code here
```

#### Equal Interval Interpolation Input
```
[Add your input format here]
```

#### Equal Interval Interpolation Output
```
[Add your output format here]
```

---

### Second Order Derivative Method 

#### Second Order Derivative Theory
[Add your theory content here]

#### Second Order Derivative Code
```python
# Add your code here
```

#### Second Order Derivative Input
```
[Add your input format here]
```

#### Second Order Derivative Output
```
[Add your output format here]
```

---

### Runge Kutta Method 

#### Runge Kutta Theory
[Add your theory content here]

#### Runge Kutta Code
```python
# Add your code here
```

#### Runge Kutta Input
```
[Add your input format here]
```

#### Runge Kutta Output
```
[Add your output format here]
```

---

### Numerical Differentiation Method

#### Numerical Differentiation Theory
[Add your theory content here]

#### Numerical Differentiation Code
```python
# Add your code here
```

#### Numerical Differentiation Input
```
[Add your input format here]
```

#### Numerical Differentiation Output
```
[Add your output format here]
```

---

### Solution of Numerical Integrations

### Simpson's One-Third Rule

#### Simpson's One-Third Rule Theory
[Add your theory content here]

#### Simpson's One-Third Rule Code
```python
# Add your code here
```

#### Simpson's One-Third Rule Input
```
[Add your input format here]
```

#### Simpson's One-Third Rule Output
```
[Add your output format here]
```

---

### Simpson's Three-Eighths Rule 

#### Simpson's Three-Eighths Rule Theory
[Add your theory content here]

#### Simpson's Three-Eighths Rule Code
```python
# Add your code here
```

#### Simpson's Three-Eighths Rule Input
```
[Add your input format here]
```

#### Simpson's Three-Eighths Rule Output
```
[Add your output format here]
```

---

## Numerical Differentiation
### Numerical Differentiation By Forward Interpolation Method

#### Numerical Differentiation By Forward Interpolation Theory

#### Numerical Differentiation By Forward Interpolation Introduction
```
[Add your input format here]
```
#### Numerical Differentiation By Forward Interpolation Algorithm steps
```
[Add your input format here]
```
#### Numerical Differentiation By Forward Interpolation Applications
```
[Add your input format here]
```

#### Numerical Differentiation By Forward Interpolation Code
```cpp
# Add your code here
```

#### Numerical Differentiation By Forward Interpolation Input
```
[Add your input format here]
```

#### Numerical Differentiation By Forward Interpolation Output
```
[Add your output format here]
```

---

### Numerical Differentiation By Backward Interpolation Method

#### Numerical Differentiation By Backward Interpolation Theory

#### Numerical Differentiation By Backward Interpolation Introduction
```
[Add your input format here]
```
#### Numerical Differentiation By Backward Interpolation Algorithm steps
```
[Add your input format here]
```
#### Numerical Differentiation By Backward Interpolation Applications
```
[Add your input format here]
```

#### Numerical Differentiation By Backward Interpolation Code
```cpp
# Add your code here
```

#### Numerical Differentiation By Backward Interpolation Input
```
[Add your input format here]
```

#### Numerical Differentiation By Backward Interpolation Output
```
[Add your output format here]
```

---

