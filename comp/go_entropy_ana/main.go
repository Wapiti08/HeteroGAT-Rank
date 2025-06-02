/**
 * @ Create Time: 2024-12-26 16:00:03
 * @ Modified time: 2025-06-02 14:50:11
 * @ Description: find malicious indicator in small-scale subgraphs
 */

package main

import (
	"DDGRL/comp"
	"fmt"
)

// Main function to test the implementation
func main() {
	// Create the graph
	g := &comp.Graph{}

	// Add nodes
	g.AddNode("A", comp.Node{Type: "Package_Name", Eco: "npm"})
	g.AddNode("B", comp.Node{Type: "Command", Eco: "pypi"})
	g.AddNode("C", model.Node{Type: "IP", Eco: "ruby"})

	// Add edges
	g.AddEdge("A", "B", model.Edge{Value: "install", Type: "action"})
	g.AddEdge("B", "C", model.Edge{Value: "resolve", Type: "DNS"})
	g.AddEdge("C", "A", model.Edge{Value: "query", Type: "DNS"})

	// Run Personalized PageRank
	totalWalks := 10000
	maxSteps := 5
	numWorkers := 4
	startNode := "A"

	proximity := model.PixieRandomWalk(g, startNode, totalWalks, maxSteps, numWorkers)

	// Print results
	fmt.Println("Node Proximity Scores:")
	for node, score := range proximity {
		fmt.Printf("Node %s: %d\n", node, score)
	}
}