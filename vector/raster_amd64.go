//go:build amd64 && !noasm
// +build amd64,!noasm

package vector

func haveAVX2() bool
func fixedAccumulateMaskAVX2(buf []uint32)
func fixedAccumulateOpSrcAVX2(dst []uint8, src []uint32)
func fixedAccumulateOpOverAVX2(dst []uint8, src []uint32)
func floatingAccumulateMaskAVX2(dst []uint32, src []float32)
func floatingAccumulateOpOverAVX2(dst []uint8, src []float32)
func floatingAccumulateOpSrcAVX2(dst []uint8, src []float32)
