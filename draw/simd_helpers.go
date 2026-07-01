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

		// 1. Скалярно делаем только приведение типов (это дешевле умножений)
		buf[0] = float64(src[pi])
		buf[1] = float64(src[pi+1])
		buf[2] = float64(src[pi+2])
		buf[3] = float64(src[pi+3])

		pixelVec := archsimd.LoadFloat64x4(&buf)

		// 2. Вычисляем итоговый вес для этого пикселя один раз
		// 257.0 — это float64 эквивалент 0x101 (ускоряет конвертацию int -> float)
		w := c.weight * 257.0

		// 3. Бродкаст веса в вектор
		wBuf := [4]float64{w, w, w, w}
		weightVec := archsimd.LoadFloat64x4(&wBuf)

		// 4. Векторное умножение + сложение (SIMD берет на себя всю тяжелую математику)
		sum = sum.Add(pixelVec.Mul(weightVec))
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

		// Загружаем каналы
		buf[0] = float64(src[pi])
		buf[1] = float64(src[pi+1])
		buf[2] = float64(src[pi+2])
		buf[3] = float64(src[pi+3])
		pixelVec := archsimd.LoadFloat64x4(&buf)

		// Считаем скалярные множители
		pau := buf[3] * 257.0
		rgbMultiplier := (pau / 65535.0) * c.weight * 257.0
		alphaMultiplier := pau * c.weight

		// Заполняем вектор множителей: [R_mult, G_mult, B_mult, A_mult]
		// Обратите внимание: для RGB множитель один, для Alpha - другой
		// (В вашем исходном коде альфа не домножалась на src[pi+3] повторно, она просто равна pau * c.weight)
		wBuf := [4]float64{rgbMultiplier, rgbMultiplier, rgbMultiplier, alphaMultiplier}

		// Для альфы исходное значение src[pi+3] в векторе pixelVec нам нужно обнулить или игнорировать,
		// так как формула для альфы buf[3] = pau * c.weight не использует умножение на саму альфу дважды.
		// Чтобы не ломать математику (так как мы делаем Mul), подставим 1.0 в позицию альфы исходного вектора:
		buf[3] = 1.0
		pixelVec = archsimd.LoadFloat64x4(&buf)

		weightVec := archsimd.LoadFloat64x4(&wBuf)
		sum = sum.Add(pixelVec.Mul(weightVec))
	}

	var out [4]float64
	sum.Store(&out)
	return out
}

// accumulateVertical performs SIMD-accelerated vertical weighted accumulation
func accumulateVertical(tmp [][4]float64, contribs []contrib, dw, dx, sStart, sEnd int32) [4]float64 {
	var sum archsimd.Float64x4

	for i := sStart; i < sEnd; i++ {
		c := &contribs[i]

		// 1. Загружаем пиксель напрямую из памяти (никаких промежуточных buf)
		pixelVec := archsimd.LoadFloat64x4(&tmp[c.coord*int32(dw)+dx])

		// 2. Бродкаст веса. Так как c.weight — скаляр, заполняем им вектор.
		// Если в archsimd есть метод типа BroadcastFloat64, используйте его.
		// Если нет, инициализируем массив:
		wBuf := [4]float64{c.weight, c.weight, c.weight, c.weight}
		weightVec := archsimd.LoadFloat64x4(&wBuf)

		// 3. Векторное умножение и сложение.
		// Если archsimd поддерживает FMA (например, sum.FMA(pixelVec, weightVec)), это будет в 2 раза быстрее!
		// Иначе делаем Mul, затем Add:
		sum = sum.Add(pixelVec.Mul(weightVec))
	}

	var out [4]float64
	sum.Store(&out)
	return out
}
