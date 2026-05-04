package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"time"
)

const (
	dead      = 0
	human     = 1
	zombie    = 2
	nx        = 150
	ny        = 150
	nHumans   = 1500
	nZombies  = 50
	steps     = 500
	pWinHuman = 0.1
)

type agent struct {
	typ int
	x   int
	y   int
}

func cas(grid []int32, addr, cmp, val int) int32 {
	old := grid[addr]
	if old == int32(cmp) {
		grid[addr] = int32(val)
	}
	return old
}

func simulateStep(agents []agent, grid []int32, src *rand.Rand) {
	na := len(agents)
	order := make([]int, na)
	for i := range order {
		order[i] = i
	}
	src.Shuffle(na, func(i, j int) { order[i], order[j] = order[j], order[i] })

	for _, ii := range order {
		idx := ii
		if agents[idx].typ == dead {
			continue
		}
		ax := agents[idx].x
		ay := agents[idx].y
		nxp := ax
		nyp := ay
		d := src.Intn(5)
		switch d {
		case 1:
			nyp = (ay - 1 + ny) % ny
		case 2:
			nyp = (ay + 1) % ny
		case 3:
			nxp = (ax - 1 + nx) % nx
		case 4:
			nxp = (ax + 1) % nx
		}
		target := nyp*nx + nxp
		oldCell := ay*nx + ax
		old := cas(grid, target, 0, idx+1)
		if old == 0 || old == int32(idx+1) {
			if d != 0 && old == 0 {
				cas(grid, oldCell, idx+1, 0)
			}
			agents[idx].x = nxp
			agents[idx].y = nyp
		} else {
			cas(grid, oldCell, 0, idx+1)
		}
	}

	src.Shuffle(na, func(i, j int) { order[i], order[j] = order[j], order[i] })
	for _, ii := range order {
		idx := ii
		if agents[idx].typ != human {
			continue
		}
		mx := agents[idx].x
		my := agents[idx].y
		nbr := []int{
			((my-1+ny)%ny)*nx + mx,
			((my+1)%ny)*nx + mx,
			my*nx + ((mx-1+nx)%nx),
			my*nx + ((mx+1)%nx),
		}
		for _, gid := range nbr {
			nid := int(grid[gid]) - 1
			if nid >= 0 && agents[nid].typ == zombie {
				if src.Float64() < pWinHuman {
					agents[nid].typ = dead
					grid[gid] = 0
				} else {
					agents[idx].typ = zombie
				}
				break
			}
		}
	}
}

func main() {
	total := nHumans + nZombies
	agents := make([]agent, total)
	grid := make([]int32, nx*ny)
	src := rand.New(rand.NewSource(12345))
	pos := src.Perm(nx * ny)

	for i := 0; i < total; i++ {
		agents[i].typ = human
		if i >= nHumans {
			agents[i].typ = zombie
		}
		p := pos[i]
		agents[i].x = p % nx
		agents[i].y = p / nx
		grid[agents[i].y*nx+agents[i].x] = int32(i + 1)
	}

	start := time.Now()
	for t := 0; t < steps; t++ {
		simulateStep(agents, grid, src)
	}
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6

	_ = agents[0].typ
	out, err := json.Marshal(map[string]float64{"execution_time_ms": elapsedMs})
	if err != nil {
		os.Exit(1)
	}
	fmt.Println(string(out))
}
