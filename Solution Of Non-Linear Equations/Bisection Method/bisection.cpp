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