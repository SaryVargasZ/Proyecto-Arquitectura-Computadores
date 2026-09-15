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
    push rbx
    push r12
    push r13
    push r14
    push r15
    push r9                  ; Guardar max* en pila

    mov rbx, rdi             ; arr
    mov r12d, esi            ; n
    mov r13, rdx             ; mean*
    mov r14, rcx             ; var*
    mov r15, r8              ; min*

    ; Caso Borde: N <= 0
    test r12d, r12d
    jle .zero_n_case

.mean_vectorial:
    call sum_array           ; xmm0 = suma total
    cvtsi2ss xmm1, r12d      ; Convertir n a float
    vdivss xmm0, xmm0, xmm1   ; xmm0 = mean
    vmovss [r13], xmm0       ; Guardar *mean

    ; Preparar media replicada para la varianza en ymm6 (¡No se sobrescribe!)
    vbroadcastss ymm6, xmm0 

    ; Inicializar min (xmm3) y max (xmm4) con el primer elemento arr[0]
    vmovss xmm3, [rbx]
    vmovss xmm4, [rbx]

    ; Inicializar acumulador de varianza a 0
    vxorps ymm0, ymm0, ymm0

    mov ecx, r12d
    and ecx, ~7              ; ecx = múltiplos de 8
    xor eax, eax             ; i = 0

    test ecx, ecx
    jle .scalar_operation    ; Si n < 8, ir directo a escalar

    ; Para min/max vectoriales, broadcasting del primer elemento a ymm4 y ymm5
    vbroadcastss ymm4, xmm3
    vbroadcastss ymm5, xmm4

.loop_vectorization:
    cmp eax, ecx
    jge .stats_reduce

    vmovups ymm1, [rbx + rax*4]

    vminps ymm4, ymm4, ymm1   ; Mínimo vectorial
    vmaxps ymm5, ymm5, ymm1   ; Máximo vectorial

    vsubps ymm3, ymm1, ymm6   ; x - mean
    vmulps ymm3, ymm3, ymm3   ; (x - mean)^2
    vaddps ymm0, ymm0, ymm3   ; Acumular varianza en ymm0

    add eax, 8
    jmp .loop_vectorization

.stats_reduce:
    ; --- Reducción Varianza (ymm0 -> xmm0) ---
    vextractf128 xmm2, ymm0, 1
    vaddps  xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    ; --- Reducción Mínimo (ymm4 -> xmm3) ---
    vextractf128 xmm2, ymm4, 1
    vminps  xmm4, xmm4, xmm2
    vpshufd xmm2, xmm4, 0x4E
    vminps  xmm4, xmm4, xmm2
    vpshufd xmm2, xmm4, 0xB1
    vminps  xmm3, xmm4, xmm2

    ; --- Reducción Máximo (ymm5 -> xmm4) ---
    vextractf128 xmm2, ymm5, 1
    vmaxps  xmm5, xmm5, xmm2
    vpshufd xmm2, xmm5, 0x4E
    vmaxps  xmm5, xmm5, xmm2
    vpshufd xmm2, xmm5, 0xB1
    vmaxps  xmm4, xmm5, xmm2

.scalar_operation:
    cmp eax, r12d
    jge .scalar_operation_done

    vmovss xmm1, [rbx + rax*4]
    vminss xmm3, xmm3, xmm1   ; Actualizar min
    vmaxss xmm4, xmm4, xmm1   ; Actualizar max

    vmovss xmm5, [r13]        ; Cargar mean
    vsubss xmm1, xmm1, xmm5   ; x[i] - mean
    vmulss xmm1, xmm1, xmm1   ; (x[i] - mean)^2
    vaddss xmm0, xmm0, xmm1   ; <--- CORREGIDO: Acumular xmm1 en xmm0

    inc eax
    jmp .scalar_operation

.scalar_operation_done:
    cvtsi2ss xmm1, r12d
    vdivss xmm0, xmm0, xmm1   ; Varianza final = suma_cuadrados / n
    vmovss [r14], xmm0        ; Guardar *var
    vmovss [r15], xmm3        ; Guardar *min
    pop r9                    ; Restaurar r9 (max*)
    vmovss [r9],  xmm4        ; Guardar *max
    jmp .done

.zero_n_case:
    pop r9                    ; Limpiar pila
    vxorps xmm0, xmm0, xmm0
    vmovss [r13], xmm0        ; *mean = 0.0
    vmovss [r14], xmm0        ; *var  = 0.0
    vmovss [r15], xmm0        ; *min  = 0.0
    vmovss [r9],  xmm0        ; *max  = 0.0

.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
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
    test edx, edx
    jle .norm_done

    vmovss xmm8, xmm0         ; xmm8 = mean
    vmovss xmm9, xmm1         ; xmm9 = stddev

    vxorps xmm0, xmm0, xmm0
    vcomiss xmm9, xmm0        ; Si stddev == 0.0
    je .zero_case_stddev_init

    vbroadcastss ymm1, xmm8   ; ymm1 = mean replicado
    vbroadcastss ymm2, xmm9   ; ymm2 = stddev replicado

    mov ecx, edx
    and ecx, ~7               ; múltiplos de 8
    xor eax, eax              ; i = 0

    test ecx, ecx
    jle .norm_scalar

.loop_norm:
    cmp eax, ecx
    jge .norm_scalar

    vmovups ymm4, [rdi + rax*4]
    vsubps  ymm4, ymm4, ymm1  ; in[i] - mean
    vdivps  ymm4, ymm4, ymm2  ; / stddev
    vmovups [rsi + rax*4], ymm4
    add eax, 8
    jmp .loop_norm

.norm_scalar:
    cmp eax, edx
    jge .norm_done

    vmovss xmm0, [rdi + rax*4]
    vsubss xmm0, xmm0, xmm8
    vdivss xmm0, xmm0, xmm9
    vmovss [rsi + rax*4], xmm0
    inc eax
    jmp .norm_scalar

.zero_case_stddev_init:
    xor eax, eax

.zero_case_stddev:
    cmp eax, edx
    jge .norm_done

    vmovss xmm0, [rdi + rax*4]
    vmovss [rsi + rax*4], xmm0
    inc eax
    jmp .zero_case_stddev

.norm_done:
    vzeroupper
    ret
