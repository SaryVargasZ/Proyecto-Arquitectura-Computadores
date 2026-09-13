; =============================================================
; stats_vector.asm
; Version VECTORIZADA (AVX2, 8 floats por iteracion) de los
; kernels de computo. Misma ABI que la version escalar.
;
; Antes de compilar/ejecutar en su maquina, confirme soporte AVX2:
;   lscpu | grep avx2
;   cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n -> retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO. Fijense especialmente en:
;   (1) como se calcula cuantos elementos entran en bucles de 8
;       ("and ecx, ~7" redondea n hacia abajo al multiplo de 8),
;   (2) la REDUCCION HORIZONTAL para pasar de 8 sumas parciales
;       (un YMM) a un unico escalar,
;   (3) el BUCLE ESCALAR DE CIERRE para el remanente (n % 8 != 0).
; Reutilicen este mismo patron en compute_stats y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador vectorial (8 carriles) = 0

    mov     ecx, esi
    and     ecx, ~7                ; ecx = n redondeado hacia abajo, multiplo de 8
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]    ; carga 8 floats (unaligned: siempre valido)
    vaddps  ymm0, ymm0, ymm1       ; acumula por carril
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    ; --- reduccion horizontal: 8 carriles de ymm0 -> un escalar ---
    vextractf128 xmm2, ymm0, 1     ; xmm2 = mitad alta (carriles 4-7)
    vaddps  xmm0, xmm0, xmm2       ; xmm0 = 4 sumas parciales (carriles 0-3 + 4-7)
    vhaddps xmm0, xmm0, xmm0       ; suma horizontal dentro de 128 bits
    vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = suma total de los 8 carriles originales

.sum_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                     ; evita penalizacion de transicion AVX/SSE
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
; TODO (estudiante):
;   1) mean = suma(arr) / n (puede llamar a sum_array; recuerde
;      guardar arr/n/mean*/var*/min*/max* en registros callee-saved
;      antes, porque la llamada destruye registros caller-saved).
;   2) Segunda pasada VECTORIZADA para acumular sum((x-mean)^2):
;        - "broadcast" de mean a los 8 carriles con vbroadcastss.
;        - vsubps + vmulps (o vfmadd231ps si quieren ir mas alla)
;          para acumular los cuadrados de las diferencias,
;        - misma reduccion horizontal que en sum_array,
;        - bucle escalar para el remanente (subss/mulss/addss).
;   3) Min/max VECTORIZADOS con vminps/vmaxps a lo largo del bucle
;      principal, reduccion final con vextractf128 + vminps/vmaxps
;      (y shuffles si quieren reducir los 4 restantes a 1), mas
;      bucle escalar de cierre con minss/maxss o comiss.
;   4) Guarde los resultados en [rdx]=mean, [rcx]=var, [r8]=min,
;      [r9]=max. Si n == 0, escriba 0.0 en los cuatro.
;   5) 'vzeroupper' antes de cualquier 'ret' en una funcion que usa
;      registros YMM.
; ---------------------------------------------------------------
compute_stats:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    mov     rbx, rdi                ; rbx = arr
    mov     r12d, esi               ; r12d = n
    mov     r13, rdx                ; r13 = mean*
    mov     r14, rcx                ; r14 = var*
    mov     r15, r8                 ; r15 = min*
    push    r9                      ; max* a la pila (6to argumento)

    ; Caso Borde: N <= 0
    test    r12d, r12d
    jle     .stats_zero

    ; Paso 1: Calcular la Media usando sum_array
    call    sum_array               ; Retorna suma en xmm0
    vcvtsi2ss xmm1, xmm1, r12d      ; xmm1 = (float) n
    vdivss  xmm0, xmm0, xmm1        ; xmm0 = mean = suma / n
    vmovss  [r13], xmm0             ; Guardar *mean

    ; Paso 2: Configurar acumuladores vectoriales
    vbroadcastss ymm2, xmm0         ; ymm2 = [mean, mean, ..., mean] (8 carriles)
    vxorps  ymm0, ymm0, ymm0        ; ymm0 = acumulador varianza = 0
    vmovups ymm3, [rbx]             ; ymm3 = min inicial (primeros 8 elementos)
    vmovups ymm4, [rbx]             ; ymm4 = max inicial (primeros 8 elementos)

    mov     ecx, r12d
    and     ecx, ~7                 ; ecx = n redondeado al múltiplo de 8
    test    ecx, ecx
    jle     .stats_scalar_prep

    xor     eax, eax                ; eax = i = 0

.stats_vec_loop:
    cmp     eax, ecx
    jge     .stats_vec_reduce

    vmovups ymm1, [rbx + rax*4]     ; Cargar 8 floats

    vminps  ymm3, ymm3, ymm1        ; Mínimo por carril
    vmaxps  ymm4, ymm4, ymm1        ; Máximo por carril

    vsubps  ymm5, ymm1, ymm2        ; ymm5 = x[i] - mean
    vmulps  ymm5, ymm5, ymm5        ; ymm5 = (x[i] - mean)^2
    vaddps  ymm0, ymm0, ymm5        ; Acumular cuadrados

    add     eax, 8
    jmp     .stats_vec_loop

.stats_vec_reduce:
    ; Reducción Varianza (ymm0 -> xmm0)
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    ; Reducción Mínimo (ymm3 -> xmm3)
    vextractf128 xmm1, ymm3, 1
    vminps  xmm3, xmm3, xmm1
    vpshufd xmm1, xmm3, 0x4E        ; Intercambiar palabras dobles 0-1 con 2-3
    vminps  xmm3, xmm3, xmm1
    vpshufd xmm1, xmm3, 0xB1        ; Intercambiar adyacentes
    vminps  xmm3, xmm3, xmm1

    ; Reducción Máximo (ymm4 -> xmm4)
    vextractf128 xmm1, ymm4, 1
    vmaxps  xmm4, xmm4, xmm1
    vpshufd xmm1, xmm4, 0x4E
    vmaxps  xmm4, xmm4, xmm1
    vpshufd xmm1, xmm4, 0xB1
    vmaxps  xmm4, xmm4, xmm1

    jmp     .stats_scalar_tail

.stats_scalar_prep:
    ; Preparación si N < 8
    vmovss  xmm3, [rbx]
    vmovss  xmm4, [rbx]
    vxorps  xmm0, xmm0, xmm0
    xor     eax, eax

.stats_scalar_tail:
    cmp     eax, r12d
    jge     .stats_done

    vmovss  xmm1, [rbx + rax*4]

    vminss  xmm3, xmm3, xmm1
    vmaxss  xmm4, xmm4, xmm1

    vsubss  xmm5, xmm1, [r13]       ; xmm5 = x[i] - mean
    vmulss  xmm5, xmm5, xmm5        ; xmm5 = (x[i] - mean)^2
    vaddss  xmm0, xmm0, xmm5

    inc     eax
    jmp     .stats_scalar_tail

.stats_done:
    vcvtsi2ss xmm1, xmm1, r12d      ; xmm1 = (float) n
    vdivss  xmm0, xmm0, xmm1        ; Varianza = suma_cuadrados / n

    vmovss  [r14], xmm0             ; *var
    vmovss  [r15], xmm3             ; *min
    pop     r9
    vmovss  [r9], xmm4              ; *max
    jmp     .stats_epilogue

.stats_zero:
    pop     r9
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0             ; *mean = 0.0
    vmovss  [r14], xmm0             ; *var  = 0.0
    vmovss  [r15], xmm0             ; *min  = 0.0
    vmovss  [r9], xmm0              ; *max  = 0.0

.stats_epilogue:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    vzeroupper
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual.
;
; TODO (estudiante):
;   - "Broadcast" mean y stddev a registros YMM con vbroadcastss
;     (guarde antes xmm0/xmm1 en otros registros o en la pila, ya
;     que planea usar xmm0/xmm1 tambien como temporales del bucle).
;   - Bucle vectorial de 8 en 8: vmovups/vmovaps carga, vsubps,
;     vdivps (o vmulps por el reciproco de stddev si quieren
;     optimizar), vmovups/vmovaps guarda.
;   - Bucle escalar de cierre para el remanente (n % 8), igual que
;     en sum_array.
;   - 'vzeroupper' antes del 'ret'.
; ---------------------------------------------------------------
normalize_array:
    ; TODO: implementar
    test    edx, edx
    jle     .norm_done

    vbroadcastss ymm2, xmm0         ; ymm2 = [mean, ..., mean]
    vxorps  xmm3, xmm3, xmm3
    vcomiss xmm1, xmm3
    je      .norm_zero_stddev

    ; Recíproco de stddev para multiplicar (1.0 / stddev)
    mov     eax, 0x3f800000         ; 1.0f en IEEE 754
    vmovd   xmm3, eax
    vdivss  xmm3, xmm3, xmm1        ; xmm3 = 1.0 / stddev
    vbroadcastss ymm3, xmm3         ; ymm3 = [1/stddev, ..., 1/stddev]

    mov     ecx, edx
    and     ecx, ~7                 ; ecx = n redondeado a múltiplo de 8
    xor     eax, eax                ; eax = i = 0

    test    ecx, ecx
    jle     .norm_scalar_tail

.norm_vec_loop:
    cmp     eax, ecx
    jge     .norm_scalar_tail

    vmovups ymm0, [rdi + rax*4]     ; Cargar 8 floats
    vsubps  ymm0, ymm0, ymm2        ; x[i] - mean
    vmulps  ymm0, ymm0, ymm3        ; (x[i] - mean) * (1/stddev)
    vmovups [rsi + rax*4], ymm0     ; Guardar 8 floats

    add     eax, 8
    jmp     .norm_vec_loop

.norm_scalar_tail:
    cmp     eax, edx
    jge     .norm_done

    vmovss  xmm0, [rdi + rax*4]
    vsubss  xmm0, xmm0, [rsp - 8]   ; Restar mean
    ; Usar xmm3 scalar para el remanente
    vextractf128 xmm4, ymm3, 0
    vmulss  xmm0, xmm0, xmm4
    vmovss  [rsi + rax*4], xmm0

    inc     eax
    jmp     .norm_scalar_tail

.norm_zero_stddev:
    ; Si stddev == 0, copiar arreglo in -> out directamente
    xor     eax, eax

.copy_loop:
    cmp     eax, edx
    jge     .norm_done

    vmovss  xmm0, [rdi + rax*4]
    vmovss  [rsi + rax*4], xmm0

    inc     eax
    jmp     .copy_loop

.norm_done:
    vzeroupper
    ret
