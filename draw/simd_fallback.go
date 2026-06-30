//go:build !goexperiment.simd || !amd64

package draw

// accumulateHorizontalRGBA is the scalar fallback for horizontal RGBA accumulation.
func accumulateHorizontalRGBA(src []uint8, contribs []contrib, srcStride, y, tIdx int32) [4]float64 {
	var pr, pg, pb, pa float64
	for i := tIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		pr += float64(src[pi]) * 0x101 * c.weight
		pg += float64(src[pi+1]) * 0x101 * c.weight
		pb += float64(src[pi+2]) * 0x101 * c.weight
		pa += float64(src[pi+3]) * 0x101 * c.weight
	}
	return [4]float64{pr, pg, pb, pa}
}

// accumulateHorizontalNRGBA is the scalar fallback for horizontal NRGBA accumulation.
func accumulateHorizontalNRGBA(src []uint8, contribs []contrib, srcStride, y, tIdx int32) [4]float64 {
	var pr, pg, pb, pa float64
	for i := tIdx; i < int32(len(contribs)); i++ {
		c := &contribs[i]
		pi := y*srcStride + c.coord*4
		pau := float64(src[pi+3]) * 0x101
		pr += float64(src[pi]) * 0x101 * pau / 0xffff * c.weight
		pg += float64(src[pi+1]) * 0x101 * pau / 0xffff * c.weight
		pb += float64(src[pi+2]) * 0x101 * pau / 0xffff * c.weight
		pa += pau * c.weight
	}
	return [4]float64{pr, pg, pb, pa}
}

// accumulateVertical is the scalar fallback for vertical accumulation.
func accumulateVertical(tmp [][4]float64, contribs []contrib, dw, dx, sStart, sEnd int32) [4]float64 {
	var pr, pg, pb, pa float64
	for i := sStart; i < sEnd; i++ {
		c := &contribs[i]
		p := &tmp[c.coord*int32(dw)+dx]
		pr += p[0] * c.weight
		pg += p[1] * c.weight
		pb += p[2] * c.weight
		pa += p[3] * c.weight
	}
	return [4]float64{pr, pg, pb, pa}
}
