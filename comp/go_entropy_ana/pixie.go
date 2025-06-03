/**
 * @ Create Time: 2024-12-26 13:52:37
 * @ Modified time: 2025-06-02 14:50:51
 * @ Description: implement Pixie Random Walk with Golang
 */

package comp

import (
	"math/rand"
	"sync"
	"time"
)

// Graph representation with edge weight
type Graph struct {
	Nodes map[string]Node
	Edges map[string]map[string]Edge
}

type Node struct {
	Value string `json:"Value"`
	Type  string `json:"Type"`
	Eco   string `json:"Eco"`
}

type Edge struct {
	Source string      `json:"Source"`
	Target string      `json:"Target"`
	Value  interface{} `json:"Value"`
	Type   string      `json:"Type"`
}


// AddNode adds a node to the graph
func (g *Graph) AddNode(key string, node Node) {
	if g.Nodes == nil {
		g.Nodes = make(map[string]Node)
	}

	g.Nodes[key] = node
}

// AddEdge adds an edge to the graph
func (g *Graph) AddEdge(source, target string, edge Edge) {
	if g.Edges == nil {
		g.Edges = make(map[string]map[string]Edge)
	}

	if g.Edges[source] == nil {
		g.Edges[source] = make(map[string]Edge)
	}

	g.Edges[source][target] = edge
}


// Worker function for random walks
func randomWalkWorker(g *Graph, startNode string, numWalks int, maxSteps int, 
	results chan map[string]int, wg *sync.WaitGroup) {
	/*
	:param startNode: starting node from random walk
	:param numWalks: number of random walks from starting node
	:param maxSteps: maximum number of steps allowed in each random walk
	:param results: channel to save result from every process
	*/
	
	// done decrements the wait group counter by one
	defer wg.Done()

	proximity := make(map[string]int)
	src := rand.NewSource(time.Now().UnixNano())
	rng := rand.New(src)

	// iter from number of walks
	for i:=0; i<numWalks; i++ {
		currentNode := startNode
		// iter from maxmium steps
		for j:=0; j < maxSteps; j++ {
			// calculate proximity of node

			proximity[currentNode]++

			// get neighbors
			neighbors := g.Edges[currentNode]
			if len(neighbors) == 0 {
				break
			}

			// choose random ngb to walk
			var nextNode string
			n := rng.Intn(len(neighbors))

			for neighbor := range neighbors {
				if n==0 {
					nextNode = neighbor
					break
				}
				n--
			}
			currentNode = nextNode

		}
	}
	// Send local proximity results back to the main thread
	results <- proximity

}



// PixieRandomWalk performs parallel random walks using goroutines
func PixieRandomWalk(g *Graph, startNode string, totalWalks int, maxSteps int, 
	numWorkers int) map[string]int {

	results := make(chan map[string]int, numWorkers)
	proximity := make(map[string]int)

	var wg sync.WaitGroup

	// calculate number of walks per worker
	walksPerWorker := totalWalks / numWorkers
	remainWalkers := totalWalks % numWorkers

	for i:=0 ;i < numWorkers; i++ {
		numWalks := walksPerWorker
		if i < remainWalkers {
			// distribute remainder among workers
			numWalks ++
		}

		wg.Add(1)
		randomWalkWorker(g, startNode, numWalks, maxSteps, results, &wg)

	}

	// close the results channel when all goroutines complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// aggregate results from all workers
	for res := range results {
		for node, count := range res {
			proximity[node] += count
		}
	}

	return proximity
}