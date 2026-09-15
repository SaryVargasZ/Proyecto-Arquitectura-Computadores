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
;Primero se debe guardar en registros callee-saved

push rbx
push r12
push r13
push r14
push r15

;mover los registros:

mov rbx, rdi
mov r12d, esi
mov r13, rdx
mov r14, rcx
mov r15, r8
push r9

.evaluate_zero_case:
test r12d, r12d
jle .zero_case

.mean:
call sum_array
cvtsi2ss xmm1, r12d
divss xmm0, xmm1
movss [r13], xmm0

;Se prepara lo que se ocupa para recorrer el arreglo:
movss xmm3, [rbx]
movss xmm4, [rbx]
movss xmm2, xmm0
xorps xmm0, xmm0
xor eax, eax
jmp .stats_loop


.zero_case:
pop r9
xorps xmm0, xmm0
movss [r13], xmm0
movss [r14], xmm0
movss [r15], xmm0
movss [r9], xmm0

jmp .epilogue



;Se recorre el arreglo para acumular la suma y min, max
.stats_loop:

cmp eax, r12d
jge .operation_done

movss xmm1, [rbx + rax*4]
minss xmm3, xmm1
maxss xmm4, xmm1
subss xmm1, xmm2
mulss xmm1, xmm1
addss xmm0, xmm1

inc eax

jmp .stats_loop



.operation_done:

cvtsi2ss xmm1, r12d
divss xmm0, xmm1
movss [r14], xmm0
movss [r13], xmm2
movss [r15], xmm3
pop r9
movss [r9], xmm4
jmp .epilogue


.epilogue: 
pop r15
pop r14
pop r13
pop r12
pop rbx
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

test edx, edx
jle .done

movss xmm8, xmm0
movss xmm9, xmm1

xor eax, eax
xorps xmm0, xmm0
xorps xmm1, xmm1

.zero_stddev:
comiss xmm0, xmm9
je .zero_case_stddev



.loop_norm:
cmp eax, edx
jge .done

movss xmm1, [rdi + rax*4]
subss xmm1, xmm8
movss xmm0, xmm1
divss xmm0, xmm9
movss [rsi+rax*4], xmm0
inc eax

jmp .loop_norm



.done: 
ret


.zero_case_stddev:

cmp     eax, edx
jge     .done

movss   xmm0, [rdi + rax*4]   
movss   [rsi + rax*4], xmm0   
inc     eax
jmp     .zero_case_stddev

