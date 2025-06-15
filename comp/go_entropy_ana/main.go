/**
 * @ Create Time: 2024-12-26 16:00:03
 * @ Modified time: 2025-06-15 09:19:58
 * @ Description: find malicious indicator in subgraphs with entropy analysis
 */

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/mem"
)

  
  func main() {
		var wg sync.WaitGroup
		done := make(chan struct{})
	  start := time.Now()
  
	  wg.Add(1)
	  go func() {
		  defer wg.Done()
		  var peakUsed uint64 = 0
	  
		  ticker := time.NewTicker(1 * time.Second)
		  defer ticker.Stop()
	  
		  for {
			  select {
			  case <-done:
				  fmt.Printf("📦 Peak Memory Usage: %.2f GB\n", float64(peakUsed)/(1024*1024*1024))
				  return
			  case <-ticker.C:
				  v, err := mem.VirtualMemory()
				  if err == nil && v.Used > peakUsed {
					  peakUsed = v.Used
				  }
			  }
		  }
	  }()
  
	  var graphs []*LabeledGraph
  
	  files, err := os.ReadDir("sample")
	  if err != nil {
		  fmt.Println("Failed to read sample directory", err)
		  close(done)
		  wg.Wait() // wait for monitor to finish
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
  
	  nodeEntEco := CountNodeEntropyByEco(graphs)
	  nodeVariance := ComputeSimpleEntropyVariance(nodeEntEco)
	  SaveTopVariance(nodeVariance, "result/node_entropy_variance.csv", 50)
	  fmt.Printf("[Overall Node Entropy Variance] = %.6f\n", ComputeOverallEntropyVariance(nodeEntEco))
  
	  edgeEntEco := CountEdgeEntropyByEco(graphs)
	  edgeVariance := ComputeSimpleEntropyVariance(edgeEntEco)
	  SaveTopVariance(edgeVariance, "result/edge_entropy_variance.csv", 50)
	  fmt.Printf("[Overall Edge Entropy Variance] = %.6f\n", ComputeOverallEntropyVariance(edgeEntEco))
 
	  ecoVariance := ComputeEntropyVariancePerEco(nodeEntEco)
	  SaveTopVariance(ecoVariance, "result/eco_entropy_variance.csv", 50)
	  
	  // Print top 10 nodes with highest entropy
		fmt.Println("\n🔝 Top 10 Nodes by Entropy Score:")
		topNodeEnt := ExtractTopEntropy(nodeEntEco, 10)
		for _, item := range topNodeEnt {
			fmt.Printf("Node: %s | Entropy: %.6f\n", item.Key, item.Val)
		}

		// Print top 10 edges with highest entropy
		fmt.Println("\n🔝 Top 10 Edges by Entropy Score:")
		topEdgeEnt := ExtractTopEntropy(edgeEntEco, 10)
		for _, item := range topEdgeEnt {
			fmt.Printf("Edge: %s | Entropy: %.6f\n", item.Key, item.Val)
		}

	  // ✅ Gracefully stop monitor and wait
	  close(done)
	  wg.Wait()
  
	  fmt.Println("✅ Entropy analysis completed. Results in ./result")
	  fmt.Printf("⏱ Time elapsed: %v\n", time.Since(start))
  }
  