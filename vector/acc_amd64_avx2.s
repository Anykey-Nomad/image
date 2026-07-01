// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm

#include "textflag.h"

// ============================================================================
// AVX2 Prefix Sum Macros
//
// Inclusive parallel prefix sum (scan) for 8 elements in a YMM register.
// Algorithm: Hillis-Steele style, 3 stages.
//
// Input:  YDST = [a0, a1, a2, a3, a4, a5, a6, a7]
// Output: YDST = [a0, a0+a1, ..., a0+a1+...+a7]
//
// Stage 1 (shift-by-1 + add): pairwise sums within each 128-bit lane
// Stage 2 (shift-by-2 + add): 4-wide prefix sums within each lane
// Stage 3 (cross-lane):       broadcast lane-0 total to lane-1, add
//
// prefix_sum_8ps: float32 variant (VADDPS)
// prefix_sum_8pd: int32 variant   (VPADDD)
// Clobbers: YTMP, XTMP0, XTMP1.

#define prefix_sum_8ps(YDST, YTMP, XTMP0, XTMP1) \
	VPSLLDQ $4, YDST, YTMP \
	VADDPS  YTMP, YDST, YDST \
	VPXOR   YTMP, YTMP, YTMP \
	VSHUFPS $0x40, YDST, YTMP, YTMP \
	VADDPS  YTMP, YDST, YDST \
	VSHUFPS      $0xFF, YDST, YDST, YTMP \
	VEXTRACTF128 $0, YTMP, XTMP0 \
	VEXTRACTF128 $1, YDST, XTMP1 \
	VADDPS       XTMP0, XTMP1, XTMP1 \
	VINSERTF128  $1, XTMP1, YDST, YDST

#define prefix_sum_8pd(YDST, YTMP, XTMP0, XTMP1) \
	VPSLLDQ $4, YDST, YTMP \
	VPADDD  YTMP, YDST, YDST \
	VPXOR   YTMP, YTMP, YTMP \
	VSHUFPS $0x40, YDST, YTMP, YTMP \
	VPADDD  YTMP, YDST, YDST \
	VSHUFPS      $0xFF, YDST, YDST, YTMP \
	VEXTRACTF128 $0, YTMP, XTMP0 \
	VEXTRACTF128 $1, YDST, XTMP1 \
	VPADDD       XTMP0, XTMP1, XTMP1 \
	VINSERTF128  $1, XTMP1, YDST, YDST

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
TEXT ·floatingAccumulateMaskAVX2(SB), NOSPLIT, $0-48
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

	prefix_sum_8ps(Y1, Y0, X0, X1)

	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	VMOVDQU Y2, (DI)

	// Offset: broadcast Y1[7] to all 8 positions
	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
// KEY: Extract both lanes to X12/X7 BEFORE the blend to avoid
// VEXTRACTF128 clearing the source YMM's upper bits.
TEXT ·floatingAccumulateOpOverAVX2(SB), NOSPLIT, $0-48
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

	MOVOU scatterAndMulBy0x101<>(SB), X8
	MOVOU fxAlmost65536<>(SB), X9
	MOVOU inverseFFFF<>(SB), X10

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

flAccOpOverAvx2Loop8:
	CMPQ R9, R10
	JAE  flAccOpOverAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8ps(Y1, Y0, X0, X1)
	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	// Extract both lanes to SEPARATE registers BEFORE blend.
	// VEXTRACTF128 to X12 clears Y12 upper bits (don't care).
	// VEXTRACTF128 to X7 clears Y7 upper bits (don't care).
	// Neither modifies Y2.
	VEXTRACTF128 $0, Y2, X12
	VEXTRACTF128 $1, Y2, X7

	// Batch 1: elements 0-3 (using X12)
	MOVL   (DI), X0
	PSHUFB X8, X0
	MOVOU  X9, X11
	PSUBL  X12, X11
	PMULLD X11, X0
	MOVOU  X0, X11
	PSRLQ  $32, X11
	PMULULQ X10, X0
	PMULULQ X10, X11
	PSRLQ  $47, X0
	PSRLQ  $47, X11
	PSLLQ  $32, X11
	XORPS  X0, X11
	PADDD  X11, X12
	PSHUFB X6, X12
	MOVL   X12, (DI)

	// Batch 2: elements 4-7 (using X7)
	MOVL   4(DI), X0
	PSHUFB X8, X0
	MOVOU  X9, X11
	PSUBL  X7, X11
	PMULLD X11, X0
	MOVOU  X0, X11
	PSRLQ  $32, X11
	PMULULQ X10, X0
	PMULULQ X10, X11
	PSRLQ  $47, X0
	PSRLQ  $47, X11
	PSLLQ  $32, X11
	XORPS  X0, X11
	PADDD  X11, X7
	PSHUFB X6, X7
	MOVL   X7, 4(DI)

	// Offset
	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
TEXT ·floatingAccumulateOpSrcAVX2(SB), NOSPLIT, $0-48
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
	MOVOU gather<>(SB), X6

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

flAccOpSrcAvx2Loop8:
	CMPQ R9, R10
	JAE  flAccOpSrcAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8ps(Y1, Y0, X0, X1)
	VADDPS Y6, Y1, Y1

	// Convert to mask
	VPAND      Y3, Y1, Y2
	VMINPS     Y4, Y2, Y2
	VMULPS     Y5, Y2, Y2
	VCVTTPS2DQ Y2, Y2

	// Extract both lanes BEFORE any VEXTRACTF128 clears Y2
	VEXTRACTF128 $0, Y2, X12
	PSHUFB X6, X12
	MOVL   X12, (DI)

	VEXTRACTF128 $1, Y2, X7
	PSHUFB X6, X7
	MOVL   X7, 4(DI)

	// Offset
	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
TEXT ·fixedAccumulateMaskAVX2(SB), NOSPLIT, $0-24
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

	prefix_sum_8pd(Y1, Y0, X0, X1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	VMOVDQU Y2, (DI)

	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
TEXT ·fixedAccumulateOpOverAVX2(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  fxAccOpOverAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_fxAlmost65536<>(SB), Y5

	MOVOU scatterAndMulBy0x101<>(SB), X8
	MOVOU fxAlmost65536<>(SB), X9
	MOVOU inverseFFFF<>(SB), X10

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

fxAccOpOverAvx2Loop8:
	CMPQ R9, R10
	JAE  fxAccOpOverAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8pd(Y1, Y0, X0, X1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	// Extract both lanes BEFORE blend
	VEXTRACTF128 $0, Y2, X12
	VEXTRACTF128 $1, Y2, X7

	// Batch 1 (X12)
	MOVL   (DI), X0
	PSHUFB X8, X0
	MOVOU  X9, X11
	PSUBL  X12, X11
	PMULLD X11, X0
	MOVOU  X0, X11
	PSRLQ  $32, X11
	PMULULQ X10, X0
	PMULULQ X10, X11
	PSRLQ  $47, X0
	PSRLQ  $47, X11
	PSLLQ  $32, X11
	XORPS  X0, X11
	PADDD  X11, X12
	PSHUFB X6, X12
	MOVL   X12, (DI)

	// Batch 2 (X7)
	MOVL   4(DI), X0
	PSHUFB X8, X0
	MOVOU  X9, X11
	PSUBL  X7, X11
	PMULLD X11, X0
	MOVOU  X0, X11
	PSRLQ  $32, X11
	PMULULQ X10, X0
	PMULULQ X10, X11
	PSRLQ  $47, X0
	PSRLQ  $47, X11
	PSLLQ  $32, X11
	XORPS  X0, X11
	PADDD  X11, X7
	PSHUFB X6, X7
	MOVL   X7, 4(DI)

	// Offset
	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
TEXT ·fixedAccumulateOpSrcAVX2(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ dst_len+8(FP), BX
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R10

	CMPQ BX, R10
	JLT  fxAccOpSrcAvx2End

	MOVQ R10, R11
	ANDQ $-8, R10

	VMOVDQU ymm_fxAlmost65536<>(SB), Y5
	MOVOU gather<>(SB), X6

	VPXOR Y6, Y6, Y6
	MOVQ  $0, R9

fxAccOpSrcAvx2Loop8:
	CMPQ R9, R10
	JAE  fxAccOpSrcAvx2Loop1

	VMOVDQU (SI), Y1

	prefix_sum_8pd(Y1, Y0, X0, X1)

	VPADDD Y6, Y1, Y1

	VPABSD  Y1, Y2
	VPSRLD  $2, Y2, Y2
	VPMINUD Y5, Y2, Y2

	// Extract both lanes BEFORE any VEXTRACTF128 clears Y2
	VEXTRACTF128 $0, Y2, X12
	PSHUFB X6, X12
	MOVL   X12, (DI)

	VEXTRACTF128 $1, Y2, X7
	PSHUFB X6, X7
	MOVL   X7, 4(DI)

	// Offset
	VSHUFPS      $0xFF, Y1, Y1, Y6
	VEXTRACTF128 $1, Y6, X7
	VINSERTF128  $0, X7, Y6, Y6

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
