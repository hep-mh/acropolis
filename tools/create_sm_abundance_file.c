#include "src/include.h"

/**
This file must be used with a local copy of AlterBBN.
Simply copy it into your main AlterBBN directoy and
compile it with

            make create_sm_abundance_file

Afterwards, it can be executed via

    ./create_sm_abundance_file.x <failsafe> <eta10>
**/

#define NY 9

void ratioH_to_Y0(double ratioH[], double Y0[]) {
    // Extract the baryon-to-photon ratio
    Y0[0] = ratioH[0];

    // Handle the special case 'p'/'H'
    Y0[2] = ratioH[2];
    // Handle the secial case 'He4'
    Y0[6] = ratioH[6]/4;

    for ( int i = 1; i <= NY; i++ ) {
        // Skip the special cases from above
        if ( i == 2 || i == 6) continue;

        Y0[i] = ratioH[i]*Y0[2];
    }

    // Revert the decays of 'H3' and 'Be7'
    Y0[5] = Y0[5] - ratioH[4]*Y0[2]; // H3
    Y0[8] = Y0[8] - ratioH[9]*Y0[2]; // Be7

    // Perform the neutron decay
    Y0[2] = Y0[2] + Y0[1];           // p
    Y0[1] = 0.;                      // n

    /* The results of this function are:
        Y0[0]       eta_final
        Y0[1]       0
        Y0[2]       (n_n + n_p) / n_b
        Y0[3]       n_D   / n_b
        Y0[4]       n_H3  / n_b
        Y0[5]       n_He3 / n_b
        Y0[6]       n_He4 / n_b = Yp/4
        Y0[7]       n_Li6 / n_b
        Y0[8]       n_Li7 / n_b
        Y0[9]       n_Be7 / n_b
    */

}

int main( int argc, char** argv ) {
    int failsafe;
    float eta;

    if ( argc != 3 ) {
        printf("\x1B[1;31mERROR  \x1B[0m: Would you kindly specify the following command-line arguments:\n"
          "         1. failsafe  0       = fast\n"
          "                      1...3   = more precise, stiff method\n"
          "                      5...7   = stiff method with precision tests   ( 5=5%%,  6=1%%,     7=0.1%%)\n"
          "                      10...12 = RK4 method with adaptative stepsize (10=5%%, 11=1%%,    12=0.1%%)\n"
          "                      20...22 = Fehlberg RK4-5 method               (20=5%%, 21=1%%,    22=0.1%%)\n"
          "                      30...32 = Cash-Karp RK4-5 method              (30=1%%, 31=1e-4,  32=1e-5).\n"
          "         2. eta10     The baryon-to-photon ratio times 1e10.\n");

        return 1;
    } else {
        sscanf(argv[1], "%d", &failsafe);
        sscanf(argv[2], "%f", &eta);
    }

    struct relicparam paramrelic;

    double ratioH[NNUC+1];
    double Y0m[NY+1], Y0h[NY+1], Y0l[NY+1];

    Init_cosmomodel(&paramrelic);

    paramrelic.failsafe = failsafe;
    paramrelic.eta0     = eta*1e-10;

    // Mean
    paramrelic.err=0;
    nucl(&paramrelic, ratioH);
    ratioH_to_Y0(ratioH, Y0m);

    // High
    paramrelic.err=1;
    nucl(&paramrelic, ratioH);
    ratioH_to_Y0(ratioH, Y0h);

    // Low
    paramrelic.err=2;
    nucl(&paramrelic, ratioH);
    ratioH_to_Y0(ratioH, Y0l);

    // Print to ...
    FILE* abundance_file = fopen("abundance_file.dat", "w");
    for ( int i = 1; i <= NY; ++i ) {
        // ... screen and ...
        printf("%.6e %.6e %.6e\n", Y0m[i], Y0h[i], Y0l[i]);
        // ... file
        fprintf(abundance_file, "%.6e %.6e %.6e\n", Y0m[i], Y0h[i], Y0l[i]);
    }
    fclose(abundance_file);

    return 0;
}
