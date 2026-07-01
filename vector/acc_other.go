// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || appengine || !gc || noasm

package vector

const haveAccumulateSIMD = false

func fixedAccumulateOpOverSIMD(dst []uint8, src []uint32)     {}
func fixedAccumulateOpSrcSIMD(dst []uint8, src []uint32)      {}
func fixedAccumulateMaskSIMD(buf []uint32)                    {}
func floatingAccumulateOpOverSIMD(dst []uint8, src []float32) {}
func floatingAccumulateOpSrcSIMD(dst []uint8, src []float32)  {}
func floatingAccumulateMaskSIMD(dst []uint32, src []float32)  {}

func haveAVX2() bool                                             { return false }
func floatingAccumulateMaskAVX2(dst []uint32, src []float32)     {}
func floatingAccumulateOpOverAVX2(dst []uint8, src []float32)    {}
func floatingAccumulateOpSrcAVX2(dst []uint8, src []float32)     {}
