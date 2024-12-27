/**
 * @ Create Time: 2024-12-26 13:52:37
 * @ Modified time: 2024-12-27 12:09:28
 * @ Description: implement personalized pagerank with golang
 */

package model

import "math"

// Graph representation with edge weight
type Graph struct {
	Nodes map[string]Node
	Edges map[[2]string]Edge
	Weights map[[2]string]float64 
}

// Node structure with value, type, and eco attributes
type Node struct {
	Type string // Node type (e.g. Path, Package_Name)
	Eco string // Ecosystem type
}

// Edge Structure with source, target, value, and type
type Edge struct {
	Value interface{} // edge value (string or other types)
	Type string
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
		g.Edges = make(map[[2]string]Edge)
	}

	if g.Weights == nil {
		g.Weights = make(map[[2]string]float64)
	}

	g.Edges[[2]string{source, target}] = edge
	// initialize the weight with zero
	g.Weights[[2]string{source, target}] = 0.0
}

// normalize the edge weights for each node
func (g *Graph) NormalizeWeights() {
	// compute total weight for each source node
	nodeTotal := make(map[string]float64)

	for edge, weight := range g.Weights{
		source := edge[0]
		nodeTotal[source] += weight
	}

	// normalize weight
	for edge := range g.Weights {
		source := edge[0]
		if nodeTotal[source] > 0 {
			g.Weights[edge] /= nodeTotal[source]
		}
	}
}

// calculate cosine similarity between two node attributes with semantic embeddings

func CosineSimiliarity(vecA, vecB []float64) float64 {
	if len(vecA) != len(vecB) {
		panic("vectors must be the same length")
	}

	var dotProduct, normA, normB float64

	for i:= 0; i < len(vecA); i++ {
		dotProduct += vecA[i] * vecB[i]
		normA += vecA[i] * vecA[i]
		normB += vecB[i] * vecB[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))

}


// compute the PPR using semantic weights
func PersonalizedPageRank(g *Graph, startNode string, alpha float64, maxIter int, tol float64) map[string]float64{
	/*
	:param startNode: the node starting to perform random walk (malicious node)
	:param alpha: teleport possibility --- avoid leaf node without connection and self-loop node
	:param maxIter: iteration times
	:param tol: convergence condition
	*/

	// initialize ppr values

	// initialize all nodes to 0

	// start node gets all the initial probability

	// perform iterations

		// copy current PPR to prevPPR

		// calculate new PPR values

	
	// wait for all Goroutines to finish

	// check convergence


}
