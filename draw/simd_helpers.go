//go:build goexperiment.simd && amd64

package draw

import "simd/archsimd"

// accumulateHorizontalRGBA performs SIMD-accelerated horizontal weighted accumulation
// of RGBA pixels from src into accumulators (pr, pg, pb, pa).
// Uses Float64x4 to process all 4 channels simultaneously.
func accumulateHorizontalRGBA(src []uint8, contribs []contrib, srcStride, y, startIdx int32) [4]float64 {
	var sum archsimd.Float64x4
	var buf [4]float64

	for i := startIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		buf[0] = float64(src[pi]) * 0x101 * c.weight
		buf[1] = float64(src[pi+1]) * 0x101 * c.weight
		buf[2] = float64(src[pi+2]) * 0x101 * c.weight
		buf[3] = float64(src[pi+3]) * 0x101 * c.weight
		pixelVec := archsimd.LoadFloat64x4(&buf)
		sum = sum.Add(pixelVec)
	}

	var out [4]float64
	sum.Store(&out)
	return out
}

// accumulateHorizontalNRGBA performs SIMD-accelerated horizontal weighted accumulation
// of NRGBA pixels (with alpha correction) into accumulators.
func accumulateHorizontalNRGBA(src []uint8, contribs []contrib, srcStride, y, startIdx int32) [4]float64 {
	var sum archsimd.Float64x4
	var buf [4]float64

	for i := startIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		pau := float64(src[pi+3]) * 0x101
		buf[0] = float64(src[pi]) * 0x101 * pau / 0xffff * c.weight
		buf[1] = float64(src[pi+1]) * 0x101 * pau / 0xffff * c.weight
		buf[2] = float64(src[pi+2]) * 0x101 * pau / 0xffff * c.weight
		buf[3] = pau * c.weight
		pixelVec := archsimd.LoadFloat64x4(&buf)
		sum = sum.Add(pixelVec)
	}

	var out [4]float64
	sum.Store(&out)
	return out
}

// accumulateVertical performs SIMD-accelerated vertical weighted accumulation
// from the tmp buffer into accumulators (pr, pg, pb, pa).
func accumulateVertical(tmp [][4]float64, contribs []contrib, dw, dx, sStart, sEnd int32) [4]float64 {
	var sum archsimd.Float64x4
	var buf [4]float64

	for i := sStart; i < sEnd; i++ {
		c := &contribs[i]
		p := &tmp[c.coord*int32(dw)+dx]
		buf[0] = p[0] * c.weight
		buf[1] = p[1] * c.weight
		buf[2] = p[2] * c.weight
		buf[3] = p[3] * c.weight
		pixelVec := archsimd.LoadFloat64x4(&buf)
		sum = sum.Add(pixelVec)
	}

	var out [4]float64
	sum.Store(&out)
	return out
}
