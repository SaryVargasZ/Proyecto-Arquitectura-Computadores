; =============================================================
; stats_vector.asm
; Implementación VECTORIZADA con AVX2.
;
; Objetivo:
;   - Procesar 8 valores float32 por iteración usando registros YMM.
;   - Calcular suma, media, varianza, mínimo y máximo.
;   - Normalizar el arreglo de forma vectorizada.
;   - Procesar con un bucle escalar los elementos sobrantes cuando
;     n no sea múltiplo de 8.
;
; Misma interfaz y ABI que la versión escalar.
;
; Requisito:
;   Confirmar soporte AVX2 antes de ejecutar:
;       lscpu | grep avx2
;       cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; =============================================================
; float sum_array(const float *arr, int n)
;
; Entradas:
;   rdi = dirección base de arr
;   esi = cantidad de elementos n
;
; Salida:
;   xmm0 = suma total
;
; Estrategia:
;   1) Procesar bloques de 8 floats con AVX2.
;   2) Reducir horizontalmente las 8 sumas parciales.
;   3) Procesar escalarmente el remanente n % 8.
; =============================================================
sum_array:
    xor     eax, eax                    ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0            ; 8 acumuladores parciales en cero

    ; ecx = mayor múltiplo de 8 menor o igual que n.
    mov     ecx, esi
    and     ecx, ~7
    test    ecx, ecx
    jle     .sum_reduce                 ; si n < 8, no hay bloque vectorial

    ; ---------------------------------------------------------
    ; Bucle vectorial: 8 floats por iteración
    ; ---------------------------------------------------------
.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce

    vmovups ymm1, [rdi + rax*4]         ; cargar arr[i] ... arr[i+7]
    vaddps  ymm0, ymm0, ymm1            ; sumar por carriles

    add     eax, 8                      ; i += 8
    jmp     .sum_vec_loop

    ; ---------------------------------------------------------
    ; Reducción horizontal:
    ; convertir 8 acumuladores en un único escalar
    ; ---------------------------------------------------------
.sum_reduce:
    vextractf128 xmm2, ymm0, 1          ; xmm2 = mitad alta de ymm0
    vaddps  xmm0, xmm0, xmm2            ; combinar mitad baja + mitad alta
    vhaddps xmm0, xmm0, xmm0            ; reducir 4 valores a 2
    vhaddps xmm0, xmm0, xmm0            ; reducir 2 valores a 1

    ; ---------------------------------------------------------
    ; Remanente escalar: elementos n % 8
    ; ---------------------------------------------------------
.sum_scalar_tail:
    cmp     eax, esi
    jge     .sum_done

    vmovss  xmm1, [rdi + rax*4]         ; cargar un float sobrante
    vaddss  xmm0, xmm0, xmm1            ; sumarlo al resultado escalar

    inc     eax                         ; i++
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                          ; limpiar parte alta de registros YMM
    ret


; =============================================================
; void compute_stats(const float *arr, int n,
;                    float *mean, float *var,
;                    float *min, float *max)
;
; Entradas según System V AMD64 ABI:
;   rdi = arr
;   esi = n
;   rdx = mean*
;   rcx = var*
;   r8  = min*
;   r9  = max*
;
; Estrategia:
;   1) Calcular la media con sum_array.
;   2) Replicar la media en los 8 carriles de ymm6.
;   3) Procesar bloques de 8 valores:
;        - mínimo con vminps
;        - máximo con vmaxps
;        - (x - mean)^2 con vsubps + vmulps
;        - acumulación de varianza con vaddps
;   4) Reducir los registros vectoriales a escalares.
;   5) Procesar el remanente con instrucciones escalares.
;   6) Dividir la suma de cuadrados entre n.
;
; Caso borde:
;   Si n <= 0, escribe 0.0 en mean, var, min y max.
; =============================================================
compute_stats:

    ; ---------------------------------------------------------
    ; Prólogo: preservar registros callee-saved utilizados
    ; ---------------------------------------------------------
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    push    r9                      ; guardar max* temporalmente

    ; Copias permanentes de los parámetros.
    mov     rbx, rdi                ; rbx  = arr
    mov     r12d, esi              ; r12d = n
    mov     r13, rdx               ; r13  = mean*
    mov     r14, rcx               ; r14  = var*
    mov     r15, r8                ; r15  = min*

    ; Caso borde: n <= 0.
    test    r12d, r12d
    jle     .zero_n_case

    ; ---------------------------------------------------------
    ; 1) Calcular la media
    ; ---------------------------------------------------------
.mean_vectorial:
    call    sum_array               ; xmm0 = suma total

    cvtsi2ss xmm1, r12d             ; xmm1 = float(n)
    vdivss  xmm0, xmm0, xmm1        ; xmm0 = mean
    vmovss  [r13], xmm0             ; *mean = mean

    ; Replicar la media en 8 carriles para restarla a un bloque
    ; completo con una sola instrucción vectorial.
    vbroadcastss ymm6, xmm0         ; ymm6 = [mean, mean, ... mean]

    ; Inicializar mínimo y máximo con el primer elemento.
    vmovss  xmm3, [rbx]             ; mínimo escalar inicial
    vmovss  xmm4, [rbx]             ; máximo escalar inicial

    ; Acumulador vectorial para sum((x - mean)^2).
    vxorps  ymm0, ymm0, ymm0

    ; ecx = mayor múltiplo de 8 <= n.
    mov     ecx, r12d
    and     ecx, ~7
    xor     eax, eax                ; eax = i = 0

    test    ecx, ecx
    jle     .scalar_operation       ; si n < 8, usar solo ruta escalar

    ; Replicar el valor inicial de min/max en los 8 carriles.
    vbroadcastss ymm4, xmm3
    vbroadcastss ymm5, xmm4

    ; ---------------------------------------------------------
    ; 2) Bucle vectorial: procesa 8 floats por iteración
    ; ---------------------------------------------------------
.loop_vectorization:
    cmp     eax, ecx
    jge     .stats_reduce

    ; Los arreglos están alineados a 32 bytes, por eso se usa vmovaps.
    vmovaps ymm1, [rbx + rax*4]     ; ymm1 = arr[i] ... arr[i+7]

    ; Actualizar mínimos y máximos parciales por cada carril.
    vminps  ymm4, ymm4, ymm1
    vmaxps  ymm5, ymm5, ymm1

    ; Acumular los cuadrados de las diferencias respecto de la media.
    vsubps  ymm3, ymm1, ymm6        ; ymm3 = x - mean
    vmulps  ymm3, ymm3, ymm3        ; ymm3 = (x - mean)^2
    vaddps  ymm0, ymm0, ymm3        ; acumulador += cuadrados

    add     eax, 8                  ; i += 8
    jmp     .loop_vectorization

    ; ---------------------------------------------------------
    ; 3) Reducciones horizontales
    ; ---------------------------------------------------------
.stats_reduce:

    ; --- Varianza: ymm0 -> xmm0[0] ---
    vextractf128 xmm2, ymm0, 1      ; extraer mitad alta
    vaddps  xmm0, xmm0, xmm2        ; combinar ambas mitades
    vhaddps xmm0, xmm0, xmm0        ; 4 parciales -> 2
    vhaddps xmm0, xmm0, xmm0        ; 2 parciales -> 1

    ; --- Mínimo: ymm4 -> xmm3[0] ---
    vextractf128 xmm2, ymm4, 1      ; mitad alta del vector de mínimos
    vminps  xmm4, xmm4, xmm2        ; combinar mitades
    vpshufd xmm2, xmm4, 0x4E        ; intercambiar pares para comparar
    vminps  xmm4, xmm4, xmm2
    vpshufd xmm2, xmm4, 0xB1        ; intercambiar vecinos
    vminps  xmm3, xmm4, xmm2        ; xmm3 queda con el mínimo final

    ; --- Máximo: ymm5 -> xmm4[0] ---
    vextractf128 xmm2, ymm5, 1      ; mitad alta del vector de máximos
    vmaxps  xmm5, xmm5, xmm2        ; combinar mitades
    vpshufd xmm2, xmm5, 0x4E
    vmaxps  xmm5, xmm5, xmm2
    vpshufd xmm2, xmm5, 0xB1
    vmaxps  xmm4, xmm5, xmm2        ; xmm4 queda con el máximo final

    ; ---------------------------------------------------------
    ; 4) Remanente escalar
    ;    Procesa desde i = múltiplo_de_8 hasta n-1.
    ; ---------------------------------------------------------
.scalar_operation:
    cmp     eax, r12d
    jge     .scalar_operation_done

    vmovss  xmm1, [rbx + rax*4]     ; cargar arr[i]

    vminss  xmm3, xmm3, xmm1        ; actualizar mínimo final
    vmaxss  xmm4, xmm4, xmm1        ; actualizar máximo final

    vmovss  xmm5, [r13]             ; xmm5 = mean
    vsubss  xmm1, xmm1, xmm5        ; x[i] - mean
    vmulss  xmm1, xmm1, xmm1        ; (x[i] - mean)^2
    vaddss  xmm0, xmm0, xmm1        ; agregar al acumulador reducido

    inc     eax                     ; i++
    jmp     .scalar_operation

    ; ---------------------------------------------------------
    ; 5) Finalizar y guardar resultados
    ; ---------------------------------------------------------
.scalar_operation_done:
    cvtsi2ss xmm1, r12d             ; xmm1 = float(n)
    vdivss  xmm0, xmm0, xmm1        ; var = suma_cuadrados / n

    vmovss  [r14], xmm0             ; *var = varianza
    vmovss  [r15], xmm3             ; *min = mínimo

    pop     r9                      ; recuperar max*
    vmovss  [r9], xmm4              ; *max = máximo

    jmp     .done

    ; ---------------------------------------------------------
    ; Caso borde: n <= 0
    ; ---------------------------------------------------------
.zero_n_case:
    pop     r9                      ; recuperar max* y balancear la pila
    vxorps  xmm0, xmm0, xmm0        ; xmm0 = 0.0

    vmovss  [r13], xmm0             ; *mean = 0.0
    vmovss  [r14], xmm0             ; *var  = 0.0
    vmovss  [r15], xmm0             ; *min  = 0.0
    vmovss  [r9],  xmm0             ; *max  = 0.0

    ; ---------------------------------------------------------
    ; Epílogo
    ; ---------------------------------------------------------
.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx

    vzeroupper                      ; evitar penalización al volver a SSE/C
    ret


; =============================================================
; void normalize_array(const float *in, float *out, int n,
;                      float mean, float stddev)
;
; Entradas:
;   rdi  = in
;   rsi  = out
;   edx  = n
;   xmm0 = mean
;   xmm1 = stddev
;
; Operación:
;   out[i] = (in[i] - mean) / stddev
;
; Estrategia:
;   - Replicar mean y stddev en registros YMM.
;   - Procesar 8 elementos por iteración con AVX2.
;   - Procesar escalarmente el remanente.
;
; Casos borde:
;   - Si n <= 0, no se procesa nada.
;   - Si stddev == 0.0, se copia in[] en out[] para evitar
;     una división por cero.
; =============================================================
normalize_array:

    ; Si no hay elementos, salir.
    test    edx, edx
    jle     .norm_done

    ; Guardar los escalares originales antes de reutilizar xmm0/xmm1.
    vmovss  xmm8, xmm0              ; xmm8 = mean
    vmovss  xmm9, xmm1              ; xmm9 = stddev

    ; Comprobar si stddev == 0.0.
    vxorps  xmm0, xmm0, xmm0        ; xmm0 = 0.0
    vcomiss xmm9, xmm0
    je      .zero_case_stddev_init

    ; Replicar ambos escalares en los ocho carriles.
    vbroadcastss ymm1, xmm8         ; ymm1 = [mean x 8]
    vbroadcastss ymm2, xmm9         ; ymm2 = [stddev x 8]

    ; ecx = cantidad de elementos que pueden procesarse de 8 en 8.
    mov     ecx, edx
    and     ecx, ~7
    xor     eax, eax                ; eax = i = 0

    test    ecx, ecx
    jle     .norm_scalar            ; si n < 8, saltar al bucle escalar

    ; ---------------------------------------------------------
    ; Bucle vectorial de normalización
    ; ---------------------------------------------------------
.loop_norm:
    cmp     eax, ecx
    jge     .norm_scalar

    vmovaps ymm4, [rdi + rax*4]     ; cargar 8 valores de entrada
    vsubps  ymm4, ymm4, ymm1        ; x - mean
    vdivps  ymm4, ymm4, ymm2        ; (x - mean) / stddev
    vmovaps [rsi + rax*4], ymm4     ; guardar 8 resultados

    add     eax, 8                  ; i += 8
    jmp     .loop_norm

    ; ---------------------------------------------------------
    ; Remanente escalar: elementos que no completan un bloque de 8
    ; ---------------------------------------------------------
.norm_scalar:
    cmp     eax, edx
    jge     .norm_done

    vmovss  xmm0, [rdi + rax*4]     ; xmm0 = in[i]
    vsubss  xmm0, xmm0, xmm8        ; xmm0 -= mean
    vdivss  xmm0, xmm0, xmm9        ; xmm0 /= stddev
    vmovss  [rsi + rax*4], xmm0     ; out[i] = resultado

    inc     eax                     ; i++
    jmp     .norm_scalar

    ; ---------------------------------------------------------
    ; Caso stddev == 0.0
    ; ---------------------------------------------------------
.zero_case_stddev_init:
    xor     eax, eax                ; comenzar copia desde i = 0

.zero_case_stddev:
    cmp     eax, edx
    jge     .norm_done

    vmovss  xmm0, [rdi + rax*4]     ; cargar in[i]
    vmovss  [rsi + rax*4], xmm0     ; copiar a out[i]

    inc     eax                     ; i++
    jmp     .zero_case_stddev

.norm_done:
    vzeroupper                      ; limpiar parte alta de los YMM
    ret


; Marcar la pila como no ejecutable.
section .note.GNU-stack noalloc noexec nowrite progbits
