#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4f

// ---------------------------------------------------------------
// Declaración de prototipos según la convención System V AMD64 ABI
// ---------------------------------------------------------------
extern float sum_array(const float *arr, int n);
extern void compute_stats(const float *arr, int n, float *mean, float *var, float *min, float *max);
extern void normalize_array(const float *in, float *out, int n, float mean, float stddev);

// Función auxiliar para comparar flotantes considerando tolerancia (1e-4)
static int float_equals(float a, float b) {
    return fabsf(a - b) <= EPSILON;
}

// ---------------------------------------------------------------
// 1. Prueba sum_array
// ---------------------------------------------------------------
void test_sum_array() {
    printf("[RUN] Testing sum_array...\n");
    float arr[] = {1.5f, 2.5f, 3.0f, -1.0f};
    float res = sum_array(arr, 4);
    assert(float_equals(res, 6.0f) && "FAIL: sum_array con arreglo estandar");
    
    // N = 0
    assert(float_equals(sum_array(arr, 0), 0.0f) && "FAIL: sum_array con N=0");
    printf("      [PASS] sum_array OK\n");
}

// ---------------------------------------------------------------
// 2. Prueba Caso Borde: N = 0
// ---------------------------------------------------------------
void test_n_zero() {
    printf("[RUN] Testing compute_stats (N = 0)...\n");
    float arr[] = {0.0f};
    float mean = 99.0f, var = 99.0f, min = 99.0f, max = 99.0f;
    
    compute_stats(arr, 0, &mean, &var, &min, &max);
    
    assert(mean == 0.0f && "FAIL: mean debe ser 0.0 cuando N=0");
    assert(var == 0.0f  && "FAIL: var debe ser 0.0 cuando N=0");
    assert(min == 0.0f  && "FAIL: min debe ser 0.0 cuando N=0");
    assert(max == 0.0f  && "FAIL: max debe ser 0.0 cuando N=0");
    printf("      [PASS] N = 0 gestionado correctamente (sin division por cero)\n");
}

// ---------------------------------------------------------------
// 3. Prueba Caso Borde: N = 1
// ---------------------------------------------------------------
void test_n_one() {
    printf("[RUN] Testing compute_stats (N = 1)...\n");
    float arr[] = {42.5f};
    float mean, var, min, max;
    
    compute_stats(arr, 1, &mean, &var, &min, &max);
    
    assert(float_equals(mean, 42.5f) && "FAIL: mean erroneo para N=1");
    assert(float_equals(var, 0.0f)  && "FAIL: varianza debe ser 0.0 para N=1");
    assert(float_equals(min, 42.5f) && "FAIL: min erroneo para N=1");
    assert(float_equals(max, 42.5f) && "FAIL: max erroneo para N=1");
    printf("      [PASS] N = 1 OK\n");
}

// ---------------------------------------------------------------
// 4. Prueba N no múltiplo de 8 (Ej: N = 7)
// ---------------------------------------------------------------
void test_n_seven() {
    printf("[RUN] Testing compute_stats (N = 7, no multiplo de 8)...\n");
    float arr[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float mean, var, min, max;
    
    // Esperados calculados a mano:
    // Suma = 28.0 | Mean = 4.0
    // Var = ((3^2 + 2^2 + 1^2 + 0 + 1^2 + 2^2 + 3^2) / 7) = 28 / 7 = 4.0
    compute_stats(arr, 7, &mean, &var, &min, &max);
    
    assert(float_equals(mean, 4.0f) && "FAIL: mean erroneo para N=7");
    assert(float_equals(var, 4.0f)  && "FAIL: varianza erronea para N=7");
    assert(float_equals(min, 1.0f)  && "FAIL: min erroneo para N=7");
    assert(float_equals(max, 7.0f)  && "FAIL: max erroneo para N=7");
    printf("      [PASS] N = 7 OK\n");
}

// ---------------------------------------------------------------
// 5. Prueba Valores Negativos
// ---------------------------------------------------------------
void test_negative_values() {
    printf("[RUN] Testing compute_stats (Valores Negativos)...\n");
    float arr[] = {-10.0f, -2.0f, -5.0f, -1.0f};
    float mean, var, min, max;
    
    // Suma = -18.0 | Mean = -4.5
    // Var = ((-5.5)^2 + (2.5)^2 + (-0.5)^2 + (3.5)^2) / 4
    //     = (30.25 + 6.25 + 0.25 + 12.25) / 4 = 49 / 4 = 12.25
    compute_stats(arr, 4, &mean, &var, &min, &max);
    
    assert(float_equals(mean, -4.5f)  && "FAIL: mean para valores negativos");
    assert(float_equals(var, 12.25f)  && "FAIL: var para valores negativos");
    assert(float_equals(min, -10.0f)  && "FAIL: min para valores negativos");
    assert(float_equals(max, -1.0f)   && "FAIL: max para valores negativos");
    printf("      [PASS] Valores negativos OK\n");
}

// ---------------------------------------------------------------
// 6. Prueba normalize_array (Caso Estándar)
// ---------------------------------------------------------------
void test_normalize_standard() {
    printf("[RUN] Testing normalize_array (Estandar)...\n");
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    float out[7] = {0};
    int n = 7;
    
    float mean = 4.0f;
    float stddev = 2.0f; // sqrt(4.0)
    
    normalize_array(in, out, n, mean, stddev);
    
    // Esperado z[i] = (x[i] - 4) / 2
    float expected[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f};
    for (int i = 0; i < n; i++) {
        assert(float_equals(out[i], expected[i]) && "FAIL: elemento normalizado no coincide");
    }
    printf("      [PASS] normalize_array estandar OK\n");
}

// ---------------------------------------------------------------
// 7. Prueba Caso Borde: Elementos Constantes (stddev = 0.0)
// ---------------------------------------------------------------
void test_normalize_zero_stddev() {
    printf("[RUN] Testing normalize_array (Caso stddev = 0.0)...\n");
    float in[] = {5.0f, 5.0f, 5.0f, 5.0f};
    float out[4] = {0};
    int n = 4;
    
    float mean = 5.0f;
    float stddev = 0.0f; // Todos los valores son iguales
    
    // Debe copiar in[i] a out[i] sin dividir por cero
    normalize_array(in, out, n, mean, stddev);
    
    for (int i = 0; i < n; i++) {
        assert(float_equals(out[i], 5.0f) && "FAIL: copia directa por stddev=0 fallo");
    }
    printf("      [PASS] Evitada division por cero cuando stddev = 0.0 OK\n");
}

// ---------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------
int main(void) {
    printf("====================================================\n");
    printf("   EJECUTANDO PRUEBAS UNITARIAS (KERNEL ESCALAR)   \n");
    printf("====================================================\n\n");

    test_sum_array();
    test_n_zero();
    test_n_one();
    test_n_seven();
    test_negative_values();
    test_normalize_standard();
    test_normalize_zero_stddev();

    printf("\n====================================================\n");
    printf("   ¡TODAS LAS PRUEBAS PASARON SATISFACTORIAMENTE!   \n");
    printf("====================================================\n");

    return 0;
}
