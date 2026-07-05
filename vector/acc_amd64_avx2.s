// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm

#include "textflag.h"

// ============================================================================
// AVX2 Prefix Sum Macros
//
// Inclusive parallel prefix sum (scan) for 8 elements across two 128-bit lanes
// in a YMM register.  Algorithm: Hillis-Steele style, per-lane then cross-lane.
//
// Input:  YDST = [a0, a1, a2, a3, a4, a5, a6, a7]
// Output: YDST = [a0, a0+a1, ..., a0+a1+...+a7]
//
// WORKAROUNDS for Go assembler limitations:
// 1) VPSLLDQ is encoded as 128-bit (VEX.L=0) even with YMM registers.
//    Fix: compute prefix sum per-lane in XMM registers.
// 2) VEXTRACTF128 encodes the wrong register in VEX.vvvv.
//    Fix: use VMOVDQU to store/load YMM through 32-byte stack buffer at (SP).
//
// prefix_sum_8ps: float32 variant (VADDPS)
// prefix_sum_8pd: int32 variant   (VPADDD)
//
// Args: YDST = input/output YMM.
//       Requires 32 bytes of stack buffer at (SP) for lane extraction.
// Clobbers: YDST, (SP)-31, X0, X1, X7.

#define prefix_sum_8ps(YDST) \
	/* Store YDST to stack buffer, then load each lane as XMM */ \
	VMOVDQU YDST, (SP) \
	VMOVDQU (SP), X0 \
	VMOVDQU 16(SP), X1 \
	/* --- Lower lane prefix sum (X0) --- */ \
	VPSLLDQ $4, X0, X7 \
	VADDPS  X7, X0, X0 \
	VPSLLDQ $8, X0, X7 \
	VADDPS  X7, X0, X0 \
	/* --- Upper lane prefix sum (X1) --- */ \
	VPSLLDQ $4, X1, X7 \
	VADDPS  X7, X1, X1 \
	VPSLLDQ $8, X1, X7 \
	VADDPS  X7, X1, X1 \
	/* --- Cross-lane: broadcast lower-lane total to upper lane --- */ \
	VSHUFPS $0xFF, X0, X0, X7 \
	VADDPS  X7, X1, X1 \
	/* --- Combine back into YMM via stack buffer --- */ \
	VMOVDQU X0, (SP) \
	VMOVDQU X1, 16(SP) \
	VMOVDQU (SP), YDST

#define prefix_sum_8pd(YDST) \
	/* Store YDST to stack buffer, then load each lane as XMM */ \
	VMOVDQU YDST, (SP) \
	VMOVDQU (SP), X0 \
	VMOVDQU 16(SP), X1 \
	/* --- Lower lane prefix sum (X0) --- */ \
	VPSLLDQ $4, X0, X7 \
	VPADDD  X7, X0, X0 \
	VPSLLDQ $8, X0, X7 \
	VPADDD  X7, X0, X0 \
	/* --- Upper lane prefix sum (X1) --- */ \
	VPSLLDQ $4, X1, X7 \
	VPADDD  X7, X1, X1 \
	VPSLLDQ $8, X1, X7 \
	VPADDD  X7, X1, X1 \
	/* --- Cross-lane: broadcast lower-lane total to upper lane --- */ \
	VSHUFPS $0xFF, X0, X0, X7 \
	VPADDD  X7, X1, X1 \
	/* --- Combine back into YMM via stack buffer --- */ \
	VMOVDQU X0, (SP) \
	VMOVDQU X1, 16(SP) \
	VMOVDQU (SP), YDST

// ----------------------------------------------------------------------------
// XMM Constants for blend operations (file-local scope, not accessible
// across .s files).

DATA scatterAndMulBy0x101<>+0x00(SB)/8, $0x8080010180800000
DATA scatterAndMulBy0x101<>+0x08(SB)/8, $0x8080030380800202
GLOBL scatterAndMulBy0x101<>(SB), (NOPTR+RODATA), $16

DATA gather<>+0x00(SB)/8, $0x808080800d090501
DATA gather<>+0x08(SB)/8, $0x8080808080808080
GLOBL gather<>(SB), (NOPTR+RODATA), $16

DATA fxAlmost65536<>+0x00(SB)/8, $0x0000ffff0000ffff
DATA fxAlmost65536<>+0x08(SB)/8, $0x0000ffff0000ffff
GLOBL fxAlmost65536<>(SB), (NOPTR+RODATA), $16

DATA inverseFFFF<>+0x00(SB)/8, $0x8000800180008001
DATA inverseFFFF<>+0x08(SB)/8, $0x8000800180008001
GLOBL inverseFFFF<>(SB), (NOPTR+RODATA), $16

// ----------------------------------------------------------------------------
// YMM Constants for AVX2 (32 bytes each)

DATA ymm_flSignMask<>+0x00(SB)/8, $0x7fffffff7fffffff
DATA ymm_flSignMask<>+0x08(SB)/8, $0x7fffffff7fffffff
DATA ymm_flSignMask<>+0x10(SB)/8, $0x7fffffff7fffffff
DATA ymm_flSignMask<>+0x18(SB)/8, $0x7fffffff7fffffff
GLOBL ymm_flSignMask<>(SB), (NOPTR+RODATA), $32

DATA ymm_flOne<>+0x00(SB)/8, $0x3f8000003f800000
DATA ymm_flOne<>+0x08(SB)/8, $0x3f8000003f800000
DATA ymm_flOne<>+0x10(SB)/8, $0x3f8000003f800000
DATA ymm_flOne<>+0x18(SB)/8, $0x3f8000003f800000
GLOBL ymm_flOne<>(SB), (NOPTR+RODATA), $32

DATA ymm_flAlmost65536<>+0x00(SB)/8, $0x477fffff477fffff
DATA ymm_flAlmost65536<>+0x08(SB)/8, $0x477fffff477fffff
DATA ymm_flAlmost65536<>+0x10(SB)/8, $0x477fffff477fffff
DATA ymm_flAlmost65536<>+0x18(SB)/8, $0x477fffff477fffff
GLOBL ymm_flAlmost65536<>(SB), (NOPTR+RODATA), $32

DATA ymm_fxAlmost65536<>+0x00(SB)/8, $0x0000ffff0000ffff
DATA ymm_fxAlmost65536<>+0x08(SB)/8, $0x0000ffff0000ffff
DATA ymm_fxAlmost65536<>+0x10(SB)/8, $0x0000ffff0000ffff
DATA ymm_fxAlmost65536<>+0x18(SB)/8, $0x0000ffff0000ffff
GLOBL ymm_fxAlmost65536<>(SB), (NOPTR+RODATA), $32

// ----------------------------------------------------------------------------
// XMM Constants for OpOver blend

DATA xmm_blend_101<>+0x00(SB)/8, $0x0000010100000101
DATA xmm_blend_101<>+0x08(SB)/8, $0x0000010100000101
GLOBL xmm_blend_101<>(SB), (NOPTR+RODATA), $16

DATA xmm_blend_one<>+0x00(SB)/8, $0x0000000100000001
DATA xmm_blend_one<>+0x08(SB)/8, $0x0000000100000001
GLOBL xmm_blend_one<>(SB), (NOPTR+RODATA), $16

// ----------------------------------------------------------------------------
// func haveAVX2() bool
TEXT ·haveAVX2(SB), NOSPLIT, $0
	MOVQ $7, AX
	XORQ CX, CX
	CPUID
	SHRQ $5, BX
	ANDQ $1, BX
	MOVB BX, ret+0(FP)
	RET

// ============================================================================
// func floatingAccumulateMaskAVX2(dst []uint32, src []float32)
TEXT ·floatingAccumulateMaskAVX2(SB), NOSPLIT, $32-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  flAccMaskAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_flSignMask<>(SB), Y3
	VMOVDQU ymm_flOne<>(SB), Y4
	VMOVDQU ymm_flAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

flAccMaskAvx2Loop8:
	CMPQ R9, R10
	JAE  flAccMaskAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8ps(Y1)

	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	VMOVDQU Y2, (DI)

	// Offset: broadcast Y1[7] to all 8 lanes
	VMOVDQU Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $32, DI
	ADDQ $32, SI
	JMP  flAccMaskAvx2Loop8

flAccMaskAvx2Loop1:
	CMPQ R9, R11
	JAE  flAccMaskAvx2End

	MOVL  (SI), X1
	VADDPS X6, X1, X1

	VPAND      X3, X1, X2
	VMINPS     X4, X2, X2
	VMULPS     X5, X2, X2
	VCVTTPS2DQ X2, X2

	VMOVD X2, (DI)
	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $4, DI
	ADDQ $4, SI
	JMP  flAccMaskAvx2Loop1

flAccMaskAvx2End:
	VZEROUPPER
	RET

// ============================================================================
// func floatingAccumulateOpOverAVX2(dst []uint8, src []float32)
//
// Vectorized OpOver blend: SIMD prefix sum + mask conversion, then fully
// vectorized Porter-Duff compositing using VPUNPCKLBW to unpack dst bytes
// to 32-bit, VPMULLD for exact 32-bit multiply, and approximate division
// x/0xffff ≈ (x + (x>>16) + 1) >> 16 (error ≤ 1 in output byte).
TEXT ·floatingAccumulateOpOverAVX2(SB), NOSPLIT, $32-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  flAccOpOverAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_flSignMask<>(SB), Y3
	VMOVDQU ymm_flOne<>(SB), Y4
	VMOVDQU ymm_flAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

flAccOpOverAvx2Loop8:
	CMPQ R9, R10
	JAE  flAccOpOverAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8ps(Y1)
	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	// Store 8 mask values to stack for blend to read
	VMOVDQU Y2, (SP)

	// ---- SIMD OpOver blend (8 pixels) ----
	// Register usage: X8-X15 as temporaries. Y1 (prefix sum) and Y6 (carry)
	// are preserved. X0 must be zero for byte unpacking.

	VPXOR    X0, X0, X0         // Zero X0 for unpacking with zeros

	// Load 8 dst bytes, unpack to 8 × 32-bit
	VMOVQ    (DI), X8           // X8 = [b0..b7, 0..0] (8 bytes in lower 64)
	VPUNPCKLBW X0, X8, X8       // bytes → 16-bit: [b0,0, b1,0, ..., b7,0]
	VPUNPCKLWD X0, X8, X9       // lower 4 → 32-bit: [b0,0,0,0, ..., b3,0,0,0]
	VPUNPCKHWD X0, X8, X8       // upper 4 → 32-bit: [b4,0,0,0, ..., b7,0,0,0]

	// Load 8 mask values (lower/upper 4 × uint32)
	VMOVDQU (SP), X10
	VMOVDQU 16(SP), X11

	// dstA = byte × 0x101 (replicate byte to 16-bit)
	VMOVDQU xmm_blend_101<>(SB), X12
	VPMULLD  X12, X9, X9        // dstA lower 4
	VPMULLD  X12, X8, X8        // dstA upper 4

	// complement = 0xffff − mask
	VMOVDQU fxAlmost65536<>(SB), X12
	VPSUBD   X10, X12, X14      // complement lower 4
	VPSUBD   X11, X12, X15      // complement upper 4

	// product = dstA × complement (32-bit exact, max ~4.28B)
	VPMULLD  X14, X9, X9
	VPMULLD  X15, X8, X8

	// quotient ≈ (product + (product >> 16) + 1) >> 16  (÷ 0xffff)
	VPSRLD   $16, X9, X14
	VPADDD   X14, X9, X9        // product + (product >> 16)
	VPSRLD   $16, X8, X14
	VPADDD   X14, X8, X8
	VMOVDQU  xmm_blend_one<>(SB), X14
	VPADDD   X14, X9, X9        // + 1
	VPADDD   X14, X8, X8
	VPSRLD   $16, X9, X9        // quotient lower 4
	VPSRLD   $16, X8, X8        // quotient upper 4

	// out16 = quotient + mask
	VPADDD   X10, X9, X9
	VPADDD   X11, X8, X8

	// outByte = out16 >> 8
	VPSRLD   $8, X9, X9
	VPSRLD   $8, X8, X8

	// Pack 8 × 32-bit → 8 × 8-bit: 32→16 (PACKUSDW) then 16→8 (PACKUSWB)
	PACKUSDW X8, X9             // 8 × 16-bit (lower 4 from X9, upper 4 from X8)
	PACKUSWB X8, X9             // 8 × 8-bit  (lower 8 bytes from X9's words)
	VMOVQ    X9, (DI)           // Store 8 result bytes

	// Carry broadcast: Y6 = [Y1[7], ..., Y1[7]] for next iteration
	VMOVDQU  Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $8, DI
	ADDQ $32, SI
	JMP  flAccOpOverAvx2Loop8

flAccOpOverAvx2Loop1:
	CMPQ R9, R11
	JAE  flAccOpOverAvx2End

	MOVL  (SI), X1
	VADDPS X6, X1, X1

	VPAND      X3, X1, X2
	VMINPS     X4, X2, X2
	VMULPS     X5, X2, X2
	VCVTTPS2DQ X2, X2

	MOVBLZX (DI), R12
	IMULL   $0x101, R12
	MOVL    X2, R13
	MOVL    $0xffff, AX
	SUBL    R13, AX
	MULL    R12
	MOVL    $0x80008001, BX
	MULL    BX
	SHRL    $15, DX
	ADDL    DX, R13
	SHRL    $8, R13
	MOVB    R13, (DI)

	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $1, DI
	ADDQ $4, SI
	JMP  flAccOpOverAvx2Loop1

flAccOpOverAvx2End:
	VZEROUPPER
	RET

// ============================================================================
// func floatingAccumulateOpSrcAVX2(dst []uint8, src []float32)
TEXT ·floatingAccumulateOpSrcAVX2(SB), NOSPLIT, $32-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  flAccOpSrcAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_flSignMask<>(SB), Y3
	VMOVDQU ymm_flOne<>(SB), Y4
	VMOVDQU ymm_flAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

flAccOpSrcAvx2Loop8:
	CMPQ R9, R10
	JAE  flAccOpSrcAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8ps(Y1)
	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	// Extract both lanes via stack
	VMOVDQU Y2, (SP)
	VMOVDQU (SP), X12
	VMOVDQU 16(SP), X7

	// Load gather mask INSIDE loop (Y6/X6 was zeroed by VPXOR above)
	MOVOU gather<>(SB), X6
	PSHUFB X6, X12
	MOVD   X12, (DI)

	PSHUFB X6, X7
	MOVD   X7, 4(DI)

	// Offset: broadcast Y1[7] to all 8 lanes
	VMOVDQU Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $8, DI
	ADDQ $32, SI
	JMP  flAccOpSrcAvx2Loop8

flAccOpSrcAvx2Loop1:
	CMPQ R9, R11
	JAE  flAccOpSrcAvx2End

	MOVL  (SI), X1
	VADDPS X6, X1, X1

	VPAND      X3, X1, X2
	VMINPS     X4, X2, X2
	VMULPS     X5, X2, X2
	VCVTTPS2DQ X2, X2

	MOVL    X2, BX
	SHRL    $8, BX
	MOVB    BX, (DI)

	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $1, DI
	ADDQ $4, SI
	JMP  flAccOpSrcAvx2Loop1

flAccOpSrcAvx2End:
	VZEROUPPER
	RET

// ============================================================================
// func fixedAccumulateMaskAVX2(buf []uint32)
TEXT ·fixedAccumulateMaskAVX2(SB), NOSPLIT, $32-24
	MOVQ buf_base+0(FP), DI
	MOVQ buf_len+8(FP), BX
	MOVQ buf_base+0(FP), SI
	MOVQ buf_len+8(FP), R10

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_fxAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

fxAccMaskAvx2Loop8:
	CMPQ R9, R10
	JAE  fxAccMaskAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8pd(Y1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	VMOVDQU Y2, (DI)

	// Offset: broadcast Y1[7] to all 8 lanes
	VMOVDQU Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $32, DI
	ADDQ $32, SI
	JMP  fxAccMaskAvx2Loop8

fxAccMaskAvx2Loop1:
	CMPQ R9, R11
	JAE  fxAccMaskAvx2End

	MOVL  (SI), X1
	VPADDD X6, X1, X1

	VPABSD  X1, X2
	VPSRLD  $2, X2, X2
	VPMINUD X5, X2, X2

	VMOVD X2, (DI)
	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $4, DI
	ADDQ $4, SI
	JMP  fxAccMaskAvx2Loop1

fxAccMaskAvx2End:
	VZEROUPPER
	RET

// ============================================================================
// func fixedAccumulateOpOverAVX2(dst []uint8, src []uint32)
//
// Vectorized OpOver blend: SIMD prefix sum + mask conversion, then fully
// vectorized Porter-Duff compositing using VPUNPCKLBW to unpack dst bytes
// to 32-bit, VPMULLD for exact 32-bit multiply, and approximate division
// x/0xffff ≈ (x + (x>>16) + 1) >> 16 (error ≤ 1 in output byte).
TEXT ·fixedAccumulateOpOverAVX2(SB), NOSPLIT, $32-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  fxAccOpOverAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_fxAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

fxAccOpOverAvx2Loop8:
	CMPQ R9, R10
	JAE  fxAccOpOverAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8pd(Y1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	// Store 8 mask values to stack for blend to read
	VMOVDQU Y2, (SP)

	// ---- SIMD OpOver blend (8 pixels) ----
	// Register usage: X8-X15 as temporaries. Y1 (prefix sum) and Y6 (carry)
	// are preserved. X0 must be zero for byte unpacking.

	VPXOR    X0, X0, X0         // Zero X0 for unpacking with zeros

	// Load 8 dst bytes, unpack to 8 × 32-bit
	VMOVQ    (DI), X8           // X8 = [b0..b7, 0..0] (8 bytes in lower 64)
	VPUNPCKLBW X0, X8, X8       // bytes → 16-bit: [b0,0, b1,0, ..., b7,0]
	VPUNPCKLWD X0, X8, X9       // lower 4 → 32-bit: [b0,0,0,0, ..., b3,0,0,0]
	VPUNPCKHWD X0, X8, X8       // upper 4 → 32-bit: [b4,0,0,0, ..., b7,0,0,0]

	// Load 8 mask values (lower/upper 4 × uint32)
	VMOVDQU (SP), X10
	VMOVDQU 16(SP), X11

	// dstA = byte × 0x101 (replicate byte to 16-bit)
	VMOVDQU xmm_blend_101<>(SB), X12
	VPMULLD  X12, X9, X9        // dstA lower 4
	VPMULLD  X12, X8, X8        // dstA upper 4

	// complement = 0xffff − mask
	VMOVDQU fxAlmost65536<>(SB), X12
	VPSUBD   X10, X12, X14      // complement lower 4
	VPSUBD   X11, X12, X15      // complement upper 4

	// product = dstA × complement (32-bit exact, max ~4.28B)
	VPMULLD  X14, X9, X9
	VPMULLD  X15, X8, X8

	// quotient ≈ (product + (product >> 16) + 1) >> 16  (÷ 0xffff)
	VPSRLD   $16, X9, X14
	VPADDD   X14, X9, X9        // product + (product >> 16)
	VPSRLD   $16, X8, X14
	VPADDD   X14, X8, X8
	VMOVDQU  xmm_blend_one<>(SB), X14
	VPADDD   X14, X9, X9        // + 1
	VPADDD   X14, X8, X8
	VPSRLD   $16, X9, X9        // quotient lower 4
	VPSRLD   $16, X8, X8        // quotient upper 4

	// out16 = quotient + mask
	VPADDD   X10, X9, X9
	VPADDD   X11, X8, X8

	// outByte = out16 >> 8
	VPSRLD   $8, X9, X9
	VPSRLD   $8, X8, X8

	// Pack 8 × 32-bit → 8 × 8-bit: 32→16 (PACKUSDW) then 16→8 (PACKUSWB)
	PACKUSDW X8, X9             // 8 × 16-bit (lower 4 from X9, upper 4 from X8)
	PACKUSWB X8, X9             // 8 × 8-bit  (lower 8 bytes from X9's words)
	VMOVQ    X9, (DI)           // Store 8 result bytes

	// Carry broadcast: Y6 = [Y1[7], ..., Y1[7]] for next iteration
	VMOVDQU  Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $8, DI
	ADDQ $32, SI
	JMP  fxAccOpOverAvx2Loop8

fxAccOpOverAvx2Loop1:
	CMPQ R9, R11
	JAE  fxAccOpOverAvx2End

	MOVL  (SI), X1
	VPADDD X6, X1, X1

	VPABSD  X1, X2
	VPSRLD  $2, X2, X2
	VPMINUD X5, X2, X2

	MOVBLZX (DI), R12
	IMULL   $0x101, R12
	MOVL    X2, R13
	MOVL    $0xffff, AX
	SUBL    R13, AX
	MULL    R12
	MOVL    $0x80008001, BX
	MULL    BX
	SHRL    $15, DX
	ADDL    DX, R13
	SHRL    $8, R13
	MOVB    R13, (DI)

	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $1, DI
	ADDQ $4, SI
	JMP  fxAccOpOverAvx2Loop1

fxAccOpOverAvx2End:
	VZEROUPPER
	RET

// ============================================================================
// func fixedAccumulateOpSrcAVX2(dst []uint8, src []uint32)
TEXT ·fixedAccumulateOpSrcAVX2(SB), NOSPLIT, $32-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  fxAccOpSrcAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_fxAlmost65536<>(SB), Y5

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

fxAccOpSrcAvx2Loop8:
	CMPQ R9, R10
	JAE  fxAccOpSrcAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8pd(Y1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	// Extract both lanes via stack
	VMOVDQU Y2, (SP)
	VMOVDQU (SP), X12
	VMOVDQU 16(SP), X7

	// Load gather mask INSIDE loop (Y6/X6 was zeroed by VPXOR above)
	MOVOU gather<>(SB), X6
	PSHUFB X6, X12
	MOVD   X12, (DI)

	PSHUFB X6, X7
	MOVD   X7, 4(DI)

	// Offset: broadcast Y1[7] to all 8 lanes
	VMOVDQU Y1, (SP)
	VPBROADCASTD 28(SP), Y6

	ADDQ $8, R9
	ADDQ $8, DI
	ADDQ $32, SI
	JMP  fxAccOpSrcAvx2Loop8

fxAccOpSrcAvx2Loop1:
	CMPQ R9, R11
	JAE  fxAccOpSrcAvx2End

	MOVL  (SI), X1
	VPADDD X6, X1, X1

	VPABSD  X1, X2
	VPSRLD  $2, X2, X2
	VPMINUD X5, X2, X2

	MOVL X2, BX
	SHRL $8, BX
	MOVB BX, (DI)

	MOVOU X1, X6

	ADDQ $1, R9
	ADDQ $1, DI
	ADDQ $4, SI
	JMP  fxAccOpSrcAvx2Loop1

fxAccOpSrcAvx2End:
	VZEROUPPER
	RET
