/**
 * @ Create Time: 2024-12-26 16:00:03
 * @ Modified time: 2025-06-13 11:45:32
 * @ Description: find malicious indicator in small-scale subgraphs
 */

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
)

// Main function to test the implementation
func main() {
	start := time.Now()

	// Measure CPU usage in a separate goroutine
	done := make(chan struct{})
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				percent, _ := cpu.Percent(time.Second, false)
				if len(percent) > 0 {
					fmt.Printf("🧠 CPU Usage: %.2f%%\n", percent[0])
				}
			}
		}
	}()

	var graphs []*LabeledGraph

	files, err := os.ReadDir("sample")
	if err != nil {
		fmt.Println("Failed to read sample directory", err)
		close(done)
		return
	}

	for _, file := range files {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		path := filepath.Join("sample", file.Name())
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

	nodeEntByEco := CountNodeValueEntropyByEco(graphs)
	for eco, entMap := range nodeEntByEco {
		outPath := fmt.Sprintf("result/node_entropy_%s.csv", eco)
		SaveTopEntropy(entMap, outPath, 50)
	}

	close(done)

	fmt.Println("✅ Entropy analysis completed. Results in ./result")
	fmt.Printf("⏱ Time elapsed: %v\n", time.Since(start))
}










