package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"
)

const (
	nWalkers = 1_000_000
	nSteps   = 1000
	nBins2D  = 100
)

type partial1D struct {
	sumPos    []float64
	sumPosSq  []float64
	histogram []int64
}

func walk1DBatch(workerID int64, walkers int, startWalker int64) partial1D {
	src := rand.New(rand.NewSource(time.Now().UnixNano() ^ (workerID << 32) ^ int64(startWalker)))
	out := partial1D{
		sumPos:    make([]float64, nSteps),
		sumPosSq:  make([]float64, nSteps),
		histogram: make([]int64, 2*nSteps+1),
	}
	for w := 0; w < walkers; w++ {
		pos := 0
		for s := 0; s < nSteps; s++ {
			if src.Intn(2) == 0 {
				pos--
			} else {
				pos++
			}
			fp := float64(pos)
			out.sumPos[s] += fp
			out.sumPosSq[s] += fp * fp
		}
		out.histogram[pos+nSteps]++
	}
	return out
}

func walk2DBatch(workerID int64, walkers int, startWalker int64) (sumR2 []float64, hist []uint64) {
	src := rand.New(rand.NewSource(time.Now().UnixNano() ^ (workerID << 16) ^ int64(startWalker<<1)))
	sumR2 = make([]float64, nSteps)
	hist = make([]uint64, nBins2D)
	maxR := math.Sqrt(2.0 * nSteps)

	for w := 0; w < walkers; w++ {
		x, y := 0, 0
		for s := 0; s < nSteps; s++ {
			if src.Intn(2) == 0 {
				x--
			} else {
				x++
			}
			if src.Intn(2) == 0 {
				y--
			} else {
				y++
			}
			xf := float64(x)
			yf := float64(y)
			sumR2[s] += xf*xf + yf*yf
		}
		r := math.Hypot(float64(x), float64(y))
		bin := int(r / maxR * float64(nBins2D))
		if bin < 0 {
			bin = 0
		}
		if bin >= nBins2D {
			bin = nBins2D - 1
		}
		hist[bin]++
	}
	return sumR2, hist
}

func main() {
	numCPU := runtime.NumCPU()
	if numCPU < 1 {
		numCPU = 1
	}

	batchSizes := make([]int, numCPU)
	base := nWalkers / numCPU
	rem := nWalkers % numCPU
	offsets := make([]int, numCPU)
	o := 0
	for i := 0; i < numCPU; i++ {
		batchSizes[i] = base
		if i < rem {
			batchSizes[i]++
		}
		offsets[i] = o
		o += batchSizes[i]
	}

	t0 := time.Now()

	var wg sync.WaitGroup
	partials := make([]partial1D, numCPU)

	for i := 0; i < numCPU; i++ {
		if batchSizes[i] == 0 {
			continue
		}
		wg.Add(1)
		go func(idx int, wk int64, walkers int, start int64) {
			defer wg.Done()
			partials[idx] = walk1DBatch(wk, walkers, start)
		}(i, int64(i), batchSizes[i], int64(offsets[i]))
	}
	wg.Wait()

	sumPos := make([]float64, nSteps)
	sumPosSq := make([]float64, nSteps)
	hist1 := make([]int64, 2*nSteps+1)
	for i := 0; i < numCPU; i++ {
		if batchSizes[i] == 0 {
			continue
		}
		p := partials[i]
		for s := 0; s < nSteps; s++ {
			sumPos[s] += p.sumPos[s]
			sumPosSq[s] += p.sumPosSq[s]
		}
		for k := range hist1 {
			hist1[k] += p.histogram[k]
		}
	}

	wg = sync.WaitGroup{}
	sumR2Parts := make([][]float64, numCPU)
	hist2Parts := make([][]uint64, numCPU)

	for i := 0; i < numCPU; i++ {
		if batchSizes[i] == 0 {
			continue
		}
		wg.Add(1)
		go func(idx int, wk int64, walkers int, start int64) {
			defer wg.Done()
			sr2, hh := walk2DBatch(wk, walkers, start)
			sumR2Parts[idx] = sr2
			hist2Parts[idx] = hh
		}(i, int64(i), batchSizes[i], int64(offsets[i]))
	}
	wg.Wait()

	sumR2 := make([]float64, nSteps)
	hist2 := make([]uint64, nBins2D)
	for i := 0; i < numCPU; i++ {
		if batchSizes[i] == 0 {
			continue
		}
		for s := 0; s < nSteps; s++ {
			sumR2[s] += sumR2Parts[i][s]
		}
		for b := 0; b < nBins2D; b++ {
			hist2[b] += hist2Parts[i][b]
		}
	}

	elapsedMs := float64(time.Since(t0).Nanoseconds()) / 1e6

	outJSON, err := json.Marshal(map[string]float64{"execution_time_ms": elapsedMs})
	if err != nil {
		os.Exit(1)
	}
	fmt.Println(string(outJSON))

	maxR := math.Sqrt(2.0 * nSteps)

	f, err := os.Create("sigma_1d.csv")
	if err != nil {
		os.Exit(1)
	}
	_, _ = fmt.Fprintf(f, "step,sigma_empirical,sigma_theory\n")
	for s := 0; s < nSteps; s++ {
		mean := sumPos[s] / float64(nWalkers)
		meanSq := sumPosSq[s] / float64(nWalkers)
		varSigma := meanSq - mean*mean
		if varSigma < 0 && varSigma > -1e-12 {
			varSigma = 0
		}
		sig := math.Sqrt(varSigma)
		_, _ = fmt.Fprintf(f, "%d,%.10g,%.10g\n", s+1, sig, math.Sqrt(float64(s+1)))
	}
	f.Close()

	f, err = os.Create("histogram_1d.csv")
	if err != nil {
		os.Exit(1)
	}
	_, _ = fmt.Fprintf(f, "position,count\n")
	for i := 0; i < 2*nSteps+1; i++ {
		_, _ = fmt.Fprintf(f, "%d,%d\n", i-nSteps, hist1[i])
	}
	f.Close()

	f, err = os.Create("sigma_2d.csv")
	if err != nil {
		os.Exit(1)
	}
	_, _ = fmt.Fprintf(f, "step,sigma_empirical,sigma_theory\n")
	for s := 0; s < nSteps; s++ {
		meanR2 := sumR2[s] / float64(nWalkers)
		_, _ = fmt.Fprintf(f, "%d,%.10g,%.10g\n", s+1, math.Sqrt(meanR2),
			math.Sqrt(2.0*float64(s+1)))
	}
	f.Close()

	f, err = os.Create("histogram_2d.csv")
	if err != nil {
		os.Exit(1)
	}
	_, _ = fmt.Fprintf(f, "bin,r_min,r_max,count\n")
	for b := 0; b < nBins2D; b++ {
		rmin := float64(b) * maxR / float64(nBins2D)
		rmax := float64(b+1) * maxR / float64(nBins2D)
		_, _ = fmt.Fprintf(f, "%d,%.5f,%.5f,%d\n", b, rmin, rmax, hist2[b])
	}
	f.Close()
}
