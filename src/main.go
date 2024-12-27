/**
 * @ Create Time: 2024-12-26 16:00:03
 * @ Modified time: 2024-12-27 12:08:42
 * @ Description: find malicious indicator in small-scale subgraphs
 */

package main

import (
	"DDGRL/model"
	"fmt"
)

// Main function to test the implementation
func main() {
	// Create the graph
	g := &model.Graph{}

	// Add nodes
	g.AddNode("A", model.Node{Type: "Package_Name", Eco: "Eco1"})
	g.AddNode("B", model.Node{Type: "Command", Eco: "Eco2"})
	g.AddNode("C", model.Node{Type: "IP", Eco: "Eco3"})

	// Add edges
	g.AddEdge("A", "B", model.Edge{Value: "install", Type: "action"})
	g.AddEdge("B", "C", model.Edge{Value: "resolve", Type: "DNS"})
	g.AddEdge("C", "A", model.Edge{Value: "query", Type: "DNS"})

	// Assign initial edge weights (e.g., based on semantic similarity)
	g.Weights[[2]string{"A", "B"}] = 1.0
	g.Weights[[2]string{"B", "C"}] = 1.0
	g.Weights[[2]string{"C", "A"}] = 1.0

	// Normalize weights
	g.NormalizeWeights()

	// Run Personalized PageRank
	alpha := 0.15
	maxIter := 100
	tol := 1e-6
	startNode := "A"

	ppr := model.PersonalizedPageRank(g, startNode, alpha, maxIter, tol)

	// Print the results
	fmt.Println("Personalized PageRank scores:")
	for node, score := range ppr {
		fmt.Printf("Node %s: %.4f\n", node, score)
	}
}