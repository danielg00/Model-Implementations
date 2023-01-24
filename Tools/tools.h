#include <stdio.h>

typedef struct {
    int dim;
    float * vector;
}  VEC;

typedef struct {
    int dims = malloc(sizeof(int) * 2);
    float ** matrix;
}  MAT;  

float VEC_VEC_MUL(VEC * v1, VEC * v2);
VEC VEC_MAT_MUL(VEC * v, MAT ** mat);
MAT MAT_MAT_MUL(MAT ** mat1, MAT ** mat2);
void read_IDX_file(FILE * file);


