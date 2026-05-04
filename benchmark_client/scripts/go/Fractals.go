package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

const (
	width    = 4000
	height   = 4000
	maxIter  = 1000
	rowBatch = 100
)

func mandelbrotRows(y0, y1 int, buf []int32) {
	const (
		xmin, xmax = -2.0, 1.0
		ymin, ymax = -1.5, 1.5
	)
	for py := y0; py < y1; py++ {
		cim := ymin + float64(py)/float64(height)*(ymax-ymin)
		rowOff := py * width
		for px := 0; px < width; px++ {
			cre := xmin + float64(px)/float64(width)*(xmax-xmin)
			zre, zim := 0.0, 0.0
			iter := 0
			for iter = 0; iter < maxIter; iter++ {
				zre2 := zre * zre
				zim2 := zim * zim
				if zre2+zim2 > 4.0 {
					break
				}
				zim = 2.0*zre*zim + cim
				zre = zre2 - zim2 + cre
			}
			buf[rowOff+px] = int32(iter)
		}
	}
}

func juliaRows(y0, y1 int, buf []int32) {
	const (
		xmin, xmax = -1.6, 1.6
		ymin, ymax = -1.6, 1.6
		cRe, cIm   = -0.7, 0.27015
	)
	for py := y0; py < y1; py++ {
		zim0 := ymin + float64(py)/float64(height)*(ymax-ymin)
		rowOff := py * width
		for px := 0; px < width; px++ {
			zre := xmin + float64(px)/float64(width)*(xmax-xmin)
			zim := zim0
			iter := 0
			for iter = 0; iter < maxIter; iter++ {
				zre2 := zre * zre
				zim2 := zim * zim
				if zre2+zim2 > 4.0 {
					break
				}
				zim = 2.0*zre*zim + cIm
				zre = zre2 - zim2 + cRe
			}
			buf[rowOff+px] = int32(iter)
		}
	}
}

func burningShipRows(y0, y1 int, buf []int32) {
	const (
		xmin, xmax = -2.2, 1.2
		ymin, ymax = -2.0, 1.0
	)
	for py := y0; py < y1; py++ {
		cim := ymin + float64(py)/float64(height)*(ymax-ymin)
		rowOff := py * width
		for px := 0; px < width; px++ {
			cre := xmin + float64(px)/float64(width)*(xmax-xmin)
			zre, zim := 0.0, 0.0
			iter := 0
			for iter = 0; iter < maxIter; iter++ {
				zre2 := zre * zre
				zim2 := zim * zim
				if zre2+zim2 > 4.0 {
					break
				}
				nextIm := 2.0*math.Abs(zre)*math.Abs(zim) + cim
				nextRe := zre2 - zim2 + cre
				zre, zim = nextRe, nextIm
			}
			buf[rowOff+px] = int32(iter)
		}
	}
}

func parallelByRows(compute func(y0, y1 int, buf []int32), buf []int32) {
	var wg sync.WaitGroup
	for y0 := 0; y0 < height; y0 += rowBatch {
		y1 := y0 + rowBatch
		if y1 > height {
			y1 = height
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			compute(a, b, buf)
		}(y0, y1)
	}
	wg.Wait()
}

func main() {
	buf := make([]int32, width*height)

	start := time.Now()
	parallelByRows(mandelbrotRows, buf)
	parallelByRows(juliaRows, buf)
	parallelByRows(burningShipRows, buf)
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6

	_ = buf[0]

	out, err := json.Marshal(map[string]float64{"execution_time_ms": elapsedMs})
	if err != nil {
		os.Exit(1)
	}
	fmt.Println(string(out))
}
