# image-simd

[English](#english) | [Русский](#русский)

---

## English

A SIMD-accelerated fork of [golang.org/x/image](https://pkg.go.dev/golang.org/x/image) — supplementary Go image packages.

### What's different from the original

This fork adds **SIMD-accelerated image drawing** via Go's auto-vectorization, specifically in the `draw` sub-package:

| Feature | Original (`golang.org/x/image`) | image-simd |
|---|---|---|
| `draw.Draw` (RGBA/NRGBA) | Scalar pixel-by-pixel | SIMD auto-vectorized accumulation |
| Horizontal accumulation | Scalar float64 loops | Vectorized with `archsimd.Float64x4` |
| Vertical accumulation | Scalar float64 loops | Vectorized with `archsimd.Float64x4` |

#### How it works

The `draw` sub-package contains two build-tagged implementations:

- **`draw/simd_helpers.go`** (`GOEXPERIMENT=simd && amd64`) — Uses `archsimd.Float64x4` vectors for parallel weighted accumulation of RGBA/NRGBA pixel channels:
  - `accumulateHorizontalRGBA` — processes all 4 channels simultaneously
  - `accumulateHorizontalNRGBA` — handles alpha correction + 4-channel accumulation
  - `accumulateVertical` — vectorized vertical weighted sum

- **`draw/simd_fallback.go`** (`!GOEXPERIMENT=simd || !amd64`) — Standard scalar loops for platforms without SIMD support.

### Usage

Drop-in replacement for `golang.org/x/image`:

```go
import "github.com/Anykey-Nomad/image-simd/draw"

// Same API as golang.org/x/image/draw
draw.Draw(dst, src.Bounds(), src, src.Min, draw.Over)
```

To enable SIMD acceleration:

```bash
GOEXPERIMENT=simd go build
```

### API

The API is identical to [golang.org/x/image](https://pkg.go.dev/golang.org/x/image). All standard image operations (`draw`, `font`, `math/fixed`, `tiff`, `webp`, etc.) are available.

### License

Same as the original: BSD 3-Clause License.

---

## Русский

Форк [golang.org/x/image](https://pkg.go.dev/golang.org/x/image) с SIMD-ускорением — вспомогательные Go-пакеты для изображений.

### Отличия от оригинала

Этот форк добавляет **SIMD-ускоренную отрисовку изображений** через автовекторизацию Go, конкретно в под-пакете `draw`:

| Функция | Оригинал (`golang.org/x/image`) | image-simd |
|---|---|---|
| `draw.Draw` (RGBA/NRGBA) | Посимвольная scalar-обработка | SIMD автовекторизированное накопление |
| Горизонтальное накопление | Scalar циклы float64 | Векторизовано с `archsimd.Float64x4` |
| Вертикальное накопление | Scalar циклы float64 | Векторизовано с `archsimd.Float64x4` |

#### Как это работает

Под-пакет `draw` содержит две реализации с build-тегами:

- **`draw/simd_helpers.go`** (`GOEXPERIMENT=simd && amd64`) — Использует векторы `archsimd.Float64x4` для параллельного взвешенного накопления каналов пикселей RGBA/NRGBA:
  - `accumulateHorizontalRGBA` — обрабатывает все 4 канала одновременно
  - `accumulateHorizontalNRGBA` — обрабатывает альфа-коррекцию + накопление 4 каналов
  - `accumulateVertical` — векторизованная вертикальная взвешенная сумма

- **`draw/simd_fallback.go`** (`!GOEXPERIMENT=simd || !amd64`) — Стандартные scalar-циклы для платформ без поддержки SIMD.

### Использование

Замена `golang.org/x/image` без изменений кода:

```go
import "github.com/Anykey-Nomad/image-simd/draw"

// API идентична golang.org/x/image/draw
draw.Draw(dst, src.Bounds(), src, src.Min, draw.Over)
```

Для включения SIMD-ускорения:

```bash
GOEXPERIMENT=simd go build
```

### API

API идентичен [golang.org/x/image](https://pkg.go.dev/golang.org/x/image). Доступны все стандартные операции (`draw`, `font`, `math/fixed`, `tiff`, `webp` и др.).

### Лицензия

BSD 3-Clause License — идентична оригиналу.
