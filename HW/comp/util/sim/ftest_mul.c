#include <stdio.h>
#include <stdlib.h>

int main() {

    FILE* fp_ref;
    FILE* fp_input;
    fp_input = fopen("input.txt", "w");
    fp_ref = fopen("fmul_cases.txt", "w");

    unsigned x, y;
    unsigned ex, ey;
    float* px = (float*)(&x);
    float* py = (float*)(&y);

    float z;
    unsigned* pz = (unsigned*)(&z);

    srand(12);

    char c[] = "x*y";

    // fprintf(fp_ref, "\tx\t\t\ty\t\t\t%s\t\t\t\t(hex)%s\t.sv result\t\tmismatch\n", c, c);

    for(int  i=0; i<10; i+=1){
        x = rand() & 0x007fffff;
        ex = rand() & 10 + 122;
        x = x | (ex << 23);

        y = rand() & 0x007fffff;
        ey = rand() & 10 + 122;
        y = y | (ey << 23);

        z = (*px) * (*py);
        
        fprintf(fp_input, "%08x%08x\n", x, y);
        fprintf(fp_ref, "\t%f\t%f\t%f\t%08x\t\t\n", *px, *py, z, *pz);

    }
    
    return 0;   
}

