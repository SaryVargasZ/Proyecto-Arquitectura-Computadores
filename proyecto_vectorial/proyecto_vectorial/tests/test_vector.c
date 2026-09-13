#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

extern float sum_array(const float *arr, int n);
extern void compute_stats(const float *arr, int n, float *mean, float *var, float *min, float *max);
extern void normalize_array(const float *in, float *out, int n, float mean, float stddev);

static int float_equals(float a, float b) {
    return fabsf(a - b) <= EPSILON;
}

void test_sum_vector() {
    printf("[RUN] Testing sum_array (AVX2)...\n");
    float arr[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float res = sum_array(arr, 10);
    assert(float_equals(res, 55.0f));
    printf("      [PASS] sum_array AVX2 OK (N=10)\n");
}

void test_stats_remanente() {
    printf("[RUN] Testing compute_stats con remanente (N=15)...\n");
    float arr[15];
    for (int i = 0; i < 15; i++) arr[i] = (float)(i + 1);
    
    float mean, var, min, max;
    compute_stats(arr, 15, &mean, &var, &min, &max);
    
    assert(float_equals(mean, 8.0f));
    assert(float_equals(min, 1.0f));
    assert(float_equals(max, 15.0f));
    printf("      [PASS] N=15 (8 vector + 7 remanente) OK\n");
}

void test_edge_cases() {
    printf("[RUN] Testing Casos Borde (N=0, N=1, stddev=0)...\n");
    
    // N = 0
    float arr0[] = {0.0f};
    float mean=99, var=99, min=99, max=99;
    compute_stats(arr0, 0, &mean, &var, &min, &max);
    assert(mean == 0.0f && var == 0.0f && min == 0.0f && max == 0.0f);

    // N = 1
    float arr1[] = {12.5f};
    compute_stats(arr1, 1, &mean, &var, &min, &max);
    assert(float_equals(mean, 12.5f) && float_equals(var, 0.0f));

    // stddev = 0
    float in_const[] = {3.0f, 3.0f, 3.0f, 3.0f};
    float out_const[4] = {0};
    normalize_array(in_const, out_const, 4, 3.0f, 0.0f);
    assert(float_equals(out_const[0], 3.0f));

    printf("      [PASS] Casos borde validados correctamente\n");
}

int main(void) {
    printf("====================================================\n");
    printf("   PRUEBAS UNITARIAS KERNEL VECTORIAL (AVX2)       \n");
    printf("====================================================\n\n");

    test_sum_vector();
    test_stats_remanente();
    test_edge_cases();

    printf("\n====================================================\n");
    printf("   ¡TODAS LAS PRUEBAS VECTORIALES PASARON!          \n");
    printf("====================================================\n");
    return 0;
}
