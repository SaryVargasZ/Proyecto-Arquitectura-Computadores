; =============================================================
; stats_scalar.asm
; Version ESCALAR (referencia) de los kernels de computo.
;
; Convencion de llamada: System V AMD64 ABI
;   enteros/punteros: rdi, rsi, rdx, rcx, r8, r9
;   flotantes:        xmm0, xmm1, xmm2, ...
;   retorno float:    xmm0
;   callee-saved:     rbx, rbp, r12-r15 (si los usa, debe preservarlos)
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n
;   retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO: estudien este patron (recorrido,
; acumulador, condicion de salida) antes de escribir compute_stats
; y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador = 0.0

.sum_loop:
    cmp     eax, esi
    jge     .sum_done
    movss   xmm1, [rdi + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .sum_loop

.sum_done:
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
;   var = varianza POBLACIONAL = sum((x - mean)^2) / n
;   Caso borde: si n == 0, escriba 0.0 en mean/var/min/max.
;
; TODO (estudiante):
;   1) Calcular mean = suma(arr) / n. Puede reutilizar sum_array con
;      'call sum_array', pero recuerde que eso destruye los
;      registros caller-saved (rax, rcx, rdx, rsi, rdi, r8-r11):
;      guarde arr/n/mean*/var*/min*/max* en registros callee-saved
;      (rbx, r12-r15) ANTES de llamar.
;   2) Recorrer el arreglo una segunda vez para acumular
;      sum((x - mean)^2) y obtener var = esa suma / n.
;   3) Recorrer el arreglo (puede combinarlo con el paso 1) llevando
;      min y max con comiss + saltos condicionales (ja/jb, etc.)
;      o con las instrucciones minss/maxss.
;   4) Guardar los resultados en las direcciones recibidas por
;      puntero: [rdx]=mean, [rcx]=var, [r8]=min, [r9]=max.
;   5) No olvide restaurar los registros callee-saved en el epilogo.
; ---------------------------------------------------------------
compute_stats:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; Respaldar argumentos en registros callee-saved (preservados tras call sum_array)
    mov     rbx, rdi            ; rbx = arr
    mov     r12d, esi           ; r12d = n
    mov     r13, rdx            ; r13 = mean*
    mov     r14, rcx            ; r14 = var*
    mov     r15, r8             ; r15 = min*
    push    r9                  ; Guardar max* en la pila (6to argumento)

    ; Caso borde: N <= 0
    test    r12d, r12d
    jle     .stats_zero

    ; 1) Calcular mean = sum_array(arr, n) / n
    call    sum_array           ; Retorna la suma en xmm0
    cvtsi2ss xmm1, r12d         ; xmm1 = (float) n
    divss   xmm0, xmm1          ; xmm0 = mean = suma / n
    movss   [r13], xmm0         ; Escribir *mean

    ; 2) Preparación para acumular varianza, min y max
    movss   xmm2, xmm0          ; xmm2 = mean
    xorps   xmm0, xmm0          ; xmm0 = acumulador varianza = 0.0

    movss   xmm3, [rbx]         ; xmm3 = min = arr[0]
    movss   xmm4, [rbx]         ; xmm4 = max = arr[0]

    xor     eax, eax            ; eax = i = 0

.stats_loop:
    cmp     eax, r12d
    jge     .stats_loop_done

    movss   xmm1, [rbx + rax*4] ; xmm1 = arr[i]

    ; Actualizar min y max
    minss   xmm3, xmm1          ; min = min(min, arr[i])
    maxss   xmm4, xmm1          ; max = max(max, arr[i])

    ; Acumular (arr[i] - mean)^2
    subss   xmm1, xmm2          ; xmm1 = arr[i] - mean
    mulss   xmm1, xmm1          ; xmm1 = (arr[i] - mean)^2
    addss   xmm0, xmm1          ; acumulador += (arr[i] - mean)^2

    inc     eax
    jmp     .stats_loop

.stats_loop_done:
    cvtsi2ss xmm1, r12d         ; xmm1 = (float) n
    divss   xmm0, xmm1          ; xmm0 = varianza = acum / n

    movss   [r14], xmm0         ; Escribir *var
    movss   [r15], xmm3         ; Escribir *min
    pop     r9                  ; Recuperar max* de la pila
    movss   [r9], xmm4          ; Escribir *max
    jmp     .stats_epilogue

.stats_zero:
    pop     r9                  ; Ajustar la pila (sacar max* sin usar)
    xorps   xmm0, xmm0          ; xmm0 = 0.0
    movss   [r13], xmm0         ; *mean = 0.0
    movss   [r14], xmm0         ; *var  = 0.0
    movss   [r15], xmm0         ; *min  = 0.0
    movss   [r9], xmm0          ; *max  = 0.0

.stats_epilogue:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual
;   (evite division por cero).
;
; TODO (estudiante): implementar el bucle escalar.
; Sugerencia: guarde mean (xmm0) y stddev (xmm1) en registros que no
; se sobrescriban dentro del bucle (por ejemplo xmm8/xmm9, que en
; System V no se usan para pasar argumentos), o vuelva a cargarlos
; en cada iteracion desde una copia guardada en la pila.
; ---------------------------------------------------------------
normalize_array:
    ; TODO: implementar
    test    edx, edx
    jle     .norm_done          ; Si n <= 0, salir

    movaps  xmm8, xmm0          ; xmm8 = mean (caller-saved seguro)
    movaps  xmm9, xmm1          ; xmm9 = stddev (caller-saved seguro)

    ; Comprobar si stddev == 0.0
    xorps   xmm0, xmm0
    comiss  xmm9, xmm0
    je      .norm_copy_loop     ; Si stddev == 0, copiar directamente

    xor     eax, eax            ; eax = i = 0

.norm_loop:
    cmp     eax, edx
    jge     .norm_done

    movss   xmm0, [rdi + rax*4] ; xmm0 = in[i]
    subss   xmm0, xmm8          ; xmm0 = in[i] - mean
    divss   xmm0, xmm9          ; xmm0 = (in[i] - mean) / stddev
    movss   [rsi + rax*4], xmm0 ; out[i] = xmm0

    inc     eax
    jmp     .norm_loop

.norm_copy_loop:
    xor     eax, eax            ; eax = i = 0

.copy_loop:
    cmp     eax, edx
    jge     .norm_done

    movss   xmm0, [rdi + rax*4]
    movss   [rsi + rax*4], xmm0 ; copia directa out[i] = in[i]

    inc     eax
    jmp     .copy_loop

.norm_done:
    ret
