/**
 * @ Create Time: 2024-12-26 16:00:03
 * @ Modified time: 2025-06-04 15:13:22
 * @ Description: find malicious indicator in small-scale subgraphs
 */

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Main function to test the implementation
func main() {
	var graphs []*LabeledGraph
	
	files, err := os.ReadDir("sample")

	if err != nil {
		fmt.Println("Failed to read sample directory", err)
		return
	}

	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		path := filepath.Join("sample", file.Name())
		// load graph in json format
		lg, err := LoadLabeledGraph(path)
		if err != nil {
			fmt.Println("Error loading:", path, err)
			continue
		}
		graphs = append(graphs, lg)
	}

	os.MkdirAll("result", os.ModePerm)

	nodeEnt := CountNodeValueEntropy(graphs)
	SaveTopEntropy(nodeEnt, "result/top_node_entropy.csv", 50)

	edgeEnt := CountEdgeValueEntropy(graphs)
	SaveTopEntropy(edgeEnt, "result/top_edge_entropy.csv", 50)

	fmt.Println("✅ Entropy analysis completed. Results in ./result")
}