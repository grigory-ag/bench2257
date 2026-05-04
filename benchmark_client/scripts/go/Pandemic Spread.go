package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"
)

const (
	susceptible = 0
	infected    = 1
	recovered   = 2
	N           = 10000
	K           = 4
	beta        = 0.1
	lambdaInf   = 0.3
	gamma       = 0.1
	maxDays     = 200
)

func buildSmallWorld(src *rand.Rand) [][]int {
	adj := make([][]int, N)
	for i := range adj {
		adj[i] = nil
	}
	for i := 0; i < N; i++ {
		for j := 1; j <= K/2; j++ {
			right := (i + j) % N
			if src.Float64() < beta {
				nt := src.Intn(N)
				for nt == i {
					nt = src.Intn(N)
				}
				adj[i] = append(adj[i], nt)
				adj[nt] = append(adj[nt], i)
			} else {
				adj[i] = append(adj[i], right)
				adj[right] = append(adj[right], i)
			}
		}
	}
	return adj
}

func epidemicStep(stIn, stOut []int, adj [][]int, src *rand.Rand) {
	for idx := 0; idx < N; idx++ {
		my := stIn[idx]
		if my == recovered {
			stOut[idx] = recovered
		} else if my == infected {
			if src.Float64() < gamma {
				stOut[idx] = recovered
			} else {
				stOut[idx] = infected
			}
		} else {
			gets := false
			for _, nb := range adj[idx] {
				if stIn[nb] != infected {
					continue
				}
				if src.Float64() < lambdaInf {
					gets = true
					break
				}
			}
			if gets {
				stOut[idx] = infected
			} else {
				stOut[idx] = susceptible
			}
		}
	}
}

func main() {
	src := rand.New(rand.NewSource(42))
	adj := buildSmallWorld(src)

	state := make([]int, N)
	nxt := make([]int, N)
	for i := 0; i < 5; i++ {
		state[src.Intn(N)] = infected
	}

	start := time.Now()
	for d := 0; d < maxDays; d++ {
		epidemicStep(state, nxt, adj, src)
		state, nxt = nxt, state
	}
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6

	_ = state[0]
	out, err := json.Marshal(map[string]float64{"execution_time_ms": elapsedMs})
	if err != nil {
		os.Exit(1)
	}
	fmt.Println(string(out))
}
