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
