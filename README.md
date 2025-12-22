### Numerical methods

## Table of Contents

- [Solution of Linear Equations](#solution-of-linear-equations)
  - [Gauss Elimination Method](#gauss-elimination-method)
    - [Theory](#gauss-elimination-theory)
      - [Intro](#introduction)
      - [Formula](#formula)
      - [Algorithm](#algorithm-steps)
      - [Application](#application)
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
    - [Code](#lu-decomposition-code)
    - [Input](#lu-decomposition-input)
    - [Output](#lu-decomposition-output)
  - [Matrix Inversion](#matrix-inversion)
    - [Theory](#matrix-inversion-theory)
    - [Code](#matrix-inversion-code)
    - [Input](#matrix-inversion-input)
    - [Output](#matrix-inversion-output)

- [Solution of Non-Linear Equations](#solution-of-non-linear-equations)
  - [Bisection Method](#bisection-method)
    - [Theory](#bisection-theory)
    - [Code](#bisection-code)
    - [Input](#bisection-input)
    - [Output](#bisection-output)
  - [False Position Method](#false-position-method)
    - [Theory](#false-position-theory)
    - [Code](#false-position-code)
    - [Input](#false-position-input)
    - [Output](#false-position-output)
  - [Secant Method](#secant-method)
    - [Theory](#secant-theory)
    - [Code](#secant-code)
    - [Input](#secant-input)
    - [Output](#secant-output)
  - [Newton Raphson Method](#newton-raphson-method)
    - [Theory](#newton-raphson-theory)
    - [Code](#newton-raphson-code)
    - [Input](#newton-raphson-input)
    - [Output](#newton-raphson-output)

- [Solution of Interpolation](#solution-of-interpolation)
  - [Newton's Forward Interpolation Method](#newtons-forward-interpolation-method)
    - [Theory](#newtons-forward-interpolation-theory)
    - [Code](#newtons-forward-interpolation-code)
    - [Input](#newtons-forward-interpolation-input)
    - [Output](#newtons-forward-interpolation-output)
  - [Newton's Backward Interpolation Method](#newtons-backward-interpolation-method)
    - [Theory](#newtons-backward-interpolation-theory)
    - [Code](#newtons-backward-interpolation-code)
    - [Input](#newtons-backward-interpolation-input)
    - [Output](#newtons-backward-interpolation-output)
  - [Divided Difference Method](#divided-difference-method)
    - [Theory](#divided-difference-theory)
    - [Code](#divided-difference-code)
    - [Input](#divided-difference-input)
    - [Output](#divided-difference-output)

- [Solution of Curve Fitting Model](#solution-of-curve-fitting-model)
  - [Least Square Regression Method For Linear Equations](#least-square-regression-method-for-linear-equations-method)
    - [Theory](#least-square-regression-method-for-linear-equations-theory)
    - [Code](#least-square-regression-method-for-linear-equations-code)
    - [Input](#least-square-regression-method-for-linear-equations-input)
    - [Output](#least-square-regression-method-for-linear-equations-output)
  - [Least Square Regression Method For Transcendental Equations](#least-square-regression-method-for-transcendental-equations)
    - [Theory](#least-square-regression-method-for-transcendental-equations-theory)
    - [Code](#least-square-regression-method-for-transcendental-equations-code)
    - [Input](#least-square-regression-method-for-transcendental-equations-input)
    - [Output](#least-square-regression-method-for-transcendental-equations-output)
  - [Least Square Regression Method For Polynomial Equations](#least-square-regression-method-for-polynomial-equations)
    - [Theory](#least-square-regression-method-for-polynomial-equations-theory)
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
```python
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
[Add your input format here]
```

#### Gauss Elimination Output
```
[Add your output format here]
```

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
#### Gauss Jordan Algoritm steps
Form the augmented matrix: Write the system [A∣b].
Make the leading diagonal 1: For each row i, divide the row by the pivot element a_ii.
Eliminate other elements in the pivot column: For each row j≠i, make the elements in the pivot column 0 using the operation:
R_j→R_j-a_ji R_i

Repeat for all pivot elements: Apply steps 2 and 3 sequentially for all rows/columns.
Obtain the solution: Once the augmented matrix is in the form [I∣x], the solution vector xis directly obtained from the last column.
#### Gauss Jordan Aplication
#### Used for:
•	Solving systems of linear equations
•	Finding matrix inverse using row operations

#### Gauss Elimination Code
```python
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

#### Gauss Elimination Input
```
[Add your input format here]
```

#### Gauss Elimination Output
```
[Add your output format here]
```


#### Gauss Jordan Code
```python
# Add your code here
```

#### Gauss Jordan Input
```
[Add your input format here]
```

#### Gauss Jordan Output
```
[Add your output format here]
```

---

### LU Decomposition Method

#### LU Decomposition Theory
[Add your theory content here]

#### LU Decomposition Code
```python
# Add your code here
```

#### LU Decomposition Input
```
[Add your input format here]
```

#### LU Decomposition Output
```
[Add your output format here]
```

---

### Matrix Inversion

#### Matrix Inversion Theory
[Add your theory content here]

#### Matrix Inversion Code
```python
# Add your code here
```

#### Matrix Inversion Input
```
[Add your input format here]
```

#### Matrix Inversion Output
```
[Add your output format here]
```

---

### Solution of Non-Linear Equations

### Bisection Method

#### Bisection Theory

#### Introduction
The Bisection Method is a numerical root-finding technique used to determine an approximate solution of the equation
f(x)=0

It is a bracketing method, which means it requires two initial guesses that enclose a root.
If a function f(x)is continuous in the interval [aⓜ,b]and
f(a)⋅f(b)<0

then, according to the Intermediate Value Theorem, at least one root lies within that interval.
The method works by repeatedly dividing the interval into two equal halves and selecting the subinterval in which the sign change occurs. It is simple, reliable, and always converges, though it may be slower compared to other methods.
#### Formula
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
#### Algorithm Steps
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
#### Application
#### Used for:
The Bisection Method is used for:
	Finding real roots of nonlinear equations f(x)=0
	Solving equations where the function is continuous over a given interval
	Obtaining a rough or initial approximation of a root
	Situations where guaranteed convergence is more important than speed
	Educational purposes to understand the basic concept of numerical root-finding
#### Suitable For
The Bisection Method is suitable for:
	Functions that are continuous in the interval [aⓜ,b]
	Problems where two initial guesses can be chosen such that
f(a)⋅f(b)<0

	Cases where the root lies near the boundary of the interval
	Problems requiring simple and stable algorithms
	Situations where derivatives are not available or difficult to compute

#### Bisection Code
```python
# Add your code here
```

#### Bisection Input
```
[Add your input format here]
```

#### Bisection Output
```
[Add your output format here]
```

---

### False Position Method

#### False Position Theory
[Add your theory content here]

#### False Position Code
```python
# Add your code here
```

#### False Position Input
```
[Add your input format here]
```

#### False Position Output
```
[Add your output format here]
```

---

### Secant Method

#### Secant Theory
[Add your theory content here]

#### Secant Code
```python
# Add your code here
```

#### Secant Input
```
[Add your input format here]
```

#### Secant Output
```
[Add your output format here]
```

---

### Newton Raphson Method

#### Newton Raphson Theory
[Add your theory content here]

#### Newton Raphson Code
```python
# Add your code here
```

#### Newton Raphson Input
```
[Add your input format here]
```

#### Newton Raphson Output
```
[Add your output format here]
```

---

### Solution of Interpolation

### Newton's Forward Interpolation Method

#### Newton's Forward Interpolation Theory
[Add your theory content here]

#### Newton's Forward Interpolation Code
```python
# Add your code here
```

#### Newton's Forward Interpolation Input
```
[Add your input format here]
```

#### Newton's Forward Interpolation Output
```
[Add your output format here]
```

---

### Newton's Backward Interpolation Method

#### Newton's Backward Interpolation Theory
[Add your theory content here]

#### Newton's Backward Interpolation Code
```python
# Add your code here
```

#### Newton's Backward Interpolation Input
```
[Add your input format here]
```

#### Newton's Backward Interpolation Output
```
[Add your output format here]
```

---

### Divided Difference Method

#### Divided Difference Theory
[Add your theory content here]

#### Divided Difference Code
```python
# Add your code here
```

#### Divided Difference Input
```
[Add your input format here]
```

#### Divided Difference Output
```
[Add your output format here]
```

---

### Solution of Curve Fitting Model

### Least Square Regression Method For Linear Equations Method

#### Least Square Regression Method For Linear Equations Theory
[Add your theory content here]

#### Least Square Regression Method For Linear Equations Code
```python
# Add your code here
```

#### Least Square Regression Method For Linear Equations Input
```
[Add your input format here]
```

#### Least Square Regression Method For Linear Equations Output
```
[Add your output format here]
```

---

### Least Square Regression Method For Transcendental Equations 

#### Least Square Regression Method For Transcendental Equations Theory
[Add your theory content here]

#### Least Square Regression Method For Transcendental Equations Code
```python
# Add your code here
```

#### Least Square Regression Method For Transcendental Equations Input
```
[Add your input format here]
```

#### Least Square Regression Method For Transcendental Equations Output
```
[Add your output format here]
```

---

### Least Square Regression Method For Polynomial Equations 

#### Least Square Regression Method For Polynomial Equations Theory
[Add your theory content here]

#### Least Square Regression Method For Polynomial Equations Code
```python
# Add your code here
```

#### Least Square Regression Method For Polynomial Equations Input
```
[Add your input format here]
```

#### Least Square Regression Method For Polynomial Equations Output
```
[Add your output format here]
```

---

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