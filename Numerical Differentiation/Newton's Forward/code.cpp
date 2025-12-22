#include<bits/stdc++.h>
using namespace std;

long long factr(int numbr)
{
    if(numbr == 0 || numbr == 1) return 1;
    return numbr * factr(numbr - 1);
}

double funcv(double xvalu)
{
    return exp(xvalu) + xvalu*xvalu*xvalu + sin(xvalu);
}

double der1v(double xvalu)
{
    return exp(xvalu) + 3*xvalu*xvalu + cos(xvalu);
}

double der2v(double xvalu)
{
    return exp(xvalu) + 6*xvalu - sin(xvalu);
}

vector<vector<double>> crfdt(vector<double>& yvalu)
{
    int nsize = yvalu.size();
    vector<vector<double>> difft(nsize, vector<double>(nsize, 0.0));

    for(int i = 0; i < nsize; i++)
        difft[i][0] = yvalu[i];

    for(int j = 1; j < nsize; j++)
        for(int i = 0; i < nsize - j; i++)
            difft[i][j] = difft[i + 1][j - 1] - difft[i][j - 1];

    return difft;
}

void slvcs(int casno, ifstream& fin, ofstream& fout)
{
    int nintv;
    fin >> nintv;

    double lowbd, upbdx;
    fin >> lowbd >> upbdx;

    double xeval;
    fin >> xeval;

    int nptss = nintv + 1;
    double hstep = (upbdx - lowbd) / nintv;

    vector<double> xgrid(nptss), yvalu(nptss);
    for(int i = 0; i < nptss; i++)
    {
        xgrid[i] = lowbd + i * hstep;
        yvalu[i] = funcv(xgrid[i]);
    }

    vector<vector<double>> difft = crfdt(yvalu);

    double uvalu = (xeval - xgrid[0]) / hstep;

    double numf1 =
        ( difft[0][1]
        + (2*uvalu - 1) * difft[0][2] / factr(2)
        + (3*uvalu*uvalu - 6*uvalu + 2) * difft[0][3] / factr(3)
        ) / hstep;

    double numf2 =
        ( difft[0][2]
        + (uvalu - 1) * difft[0][3]
        ) / (hstep * hstep);

    double exfdt = der1v(xeval);
    double exsdt = der2v(xeval);

    double errf1 = fabs((exfdt - numf1) / exfdt) * 100;
    double errf2 = fabs((exsdt - numf2) / exsdt) * 100;

    fout << "\nTEST CASE #" << casno << "\n";
    fout << fixed << setprecision(6);
    fout << "Numerical f'(x)  = " << numf1 << "\n";
    fout << "Exact f'(x)      = " << exfdt << "\n";
    fout << "Numerical f''(x) = " << numf2 << "\n";
    fout << "Exact f''(x)     = " << exsdt << "\n";
    fout << "Error in f'(x)   = " << errf1 << "%\n";
    fout << "Error in f''(x)  = " << errf2 << "%\n";
}

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int tcase;
    fin >> tcase;

    for(int i = 1; i <= tcase; i++)
        slvcs(i, fin, fout);

    fin.close();
    fout.close();
    return 0;
}
