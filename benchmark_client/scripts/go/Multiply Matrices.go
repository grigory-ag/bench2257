package main

import (
	"encoding/binary"
	"fmt"
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

func main() {
	n1, a, err := readMatrix("M1.dat")
	if err != nil {
		os.Exit(1)
	}
	n2, b, err := readMatrix("M2.dat")
	if err != nil || n1 != n2 {
		os.Exit(1)
	}

	n := n1
	c := make([]float32, n*n)

	start := time.Now()
	for i := 0; i < n; i++ {
		iBase := i * n
		for k := 0; k < n; k++ {
			aik := a[iBase+k]
			kBase := k * n
			for j := 0; j < n; j++ {
				c[iBase+j] += aik * b[kBase+j]
			}
		}
	}
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6

	if err := writeMatrix("M3.dat", n, c); err != nil {
		os.Exit(1)
	}
	fmt.Printf("{\"execution_time_ms\": %v}\n", elapsedMs)
}
