package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"time"
)

func readMatrix(path string) (int, []float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, nil, err
	}
	defer file.Close()

	var n int32
	if err := binary.Read(file, binary.LittleEndian, &n); err != nil {
		return 0, nil, err
	}
	if n <= 0 {
		return 0, nil, fmt.Errorf("invalid matrix size")
	}

	total := int(n) * int(n)
	data := make([]float32, total)
	if err := binary.Read(file, binary.LittleEndian, data); err != nil {
		return 0, nil, err
	}

	return int(n), data, nil
}

func writeMatrix(path string, n int, data []float32) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if len(data) != n*n {
		return fmt.Errorf("invalid output matrix size")
	}

	n32 := int32(n)
	if err := binary.Write(file, binary.LittleEndian, n32); err != nil {
		return err
	}
	if err := binary.Write(file, binary.LittleEndian, data); err != nil {
		return err
	}
	return nil
}

func invertGaussJordan(input []float32, n int) ([]float32, bool) {
	cols := 2 * n
	aug := make([]float32, n*cols)

	for i := 0; i < n; i++ {
		iOff := i * cols
		inOff := i * n
		for j := 0; j < n; j++ {
			aug[iOff+j] = input[inOff+j]
		}
		aug[iOff+n+i] = 1.0
	}

	for i := 0; i < n; i++ {
		pivotRow := i
		pivotAbs := float32(math.Abs(float64(aug[i*cols+i])))
		for r := i + 1; r < n; r++ {
			v := float32(math.Abs(float64(aug[r*cols+i])))
			if v > pivotAbs {
				pivotAbs = v
				pivotRow = r
			}
		}
		if pivotAbs <= 1e-12 {
			return nil, false
		}

		if pivotRow != i {
			iOff := i * cols
			pOff := pivotRow * cols
			for c := 0; c < cols; c++ {
				aug[iOff+c], aug[pOff+c] = aug[pOff+c], aug[iOff+c]
			}
		}

		iOff := i * cols
		pivot := aug[iOff+i]
		for c := 0; c < cols; c++ {
			aug[iOff+c] /= pivot
		}

		for r := 0; r < n; r++ {
			if r == i {
				continue
			}
			rOff := r * cols
			factor := aug[rOff+i]
			if factor == 0 {
				continue
			}
			for c := 0; c < cols; c++ {
				aug[rOff+c] -= factor * aug[iOff+c]
			}
		}
	}

	inv := make([]float32, n*n)
	for i := 0; i < n; i++ {
		iOff := i * cols
		outOff := i * n
		for j := 0; j < n; j++ {
			inv[outOff+j] = aug[iOff+n+j]
		}
	}

	return inv, true
}

func main() {
	n, t, err := readMatrix("T.dat")
	if err != nil {
		os.Exit(1)
	}

	start := time.Now()
	inv, ok := invertGaussJordan(t, n)
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6
	if !ok {
		os.Exit(1)
	}

	if err := writeMatrix("M3.dat", n, inv); err != nil {
		os.Exit(1)
	}
	fmt.Printf("{\"execution_time_ms\": %v}\n", elapsedMs)
}
