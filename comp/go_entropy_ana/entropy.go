/**
 * @ Create Time: 2024-12-26 13:52:37
 * @ Modified time: 2025-06-14 16:30:33
 * @ Description: implement Pixie Random Walk with Golang
 */

package main

import (
	"fmt"
	"math"
	"os"
	"sort"
)

// Graph representation with edge weight
  func computeEntropy(counts map[int]int) float64 {
	  // implement the entropy calculation based on formula
	  total := 0
	  for _, v := range counts {
		  total += v
	  }
  
	  if total == 0 {
		  return 0.0
	  }
  
	  ent := 0.0
	  for _, v := range counts {
		  p := float64(v) / float64(total)
		  ent -= p * math.Log2(p)
	  }
  
	  return ent
  }
  
  func computeVariance(values []float64) float64 {
	  n := float64(len(values))
	  if n == 0 {
		  return 0.0
	  }
	  sum := 0.0
	  sumSq := 0.0
	  for _, v := range values {
		  sum += v
		  sumSq += v * v
	  }
	  mean := sum / n
	  return (sumSq / n) - (mean * mean)
  }
  
  func CountNodeValueEntropy(graphs []*LabeledGraph) map[string]float64 {
	  // create counts map to save results
	  counts := make(map[string]map[int]int)
	  for _, lg := range graphs {
		  for _, node := range lg.Graph.Nodes {
			  key := fmt.Sprintf("%s|%s|%s", node.Type, node.Eco, node.Value)
			  // check whether key exists before
			  if _, ok := counts[key];!ok {
				  counts[key] = make(map[int]int)
			  }
			  counts[key][lg.Label]++
  
		  }
	  }
  
	  result := make(map[string]float64)
	  for val, labelCounts := range counts {
		  result[val] = computeEntropy(labelCounts)
	  }
  
	  return result
  }
  
  
  func CountNodeEntropyByEco(graphs []*LabeledGraph) map[string]map[string]float64 {
	  counts := make(map[string]map[string]map[int]int)
  
	  for _, lg := range graphs {
		  for _, node := range lg.Graph.Nodes {
			  nodeKey := fmt.Sprintf("%s|%s", node.Type, node.Value)
			  eco := node.Eco
  
			  if _, ok := counts[nodeKey]; !ok {
				  counts[nodeKey] = make(map[string]map[int]int)
			  }
			  if _, ok := counts[nodeKey][eco]; !ok {
				  counts[nodeKey][eco] = make(map[int]int)
			  }
			  counts[nodeKey][eco][lg.Label]++
		  }
	  }
  
	  result := make(map[string]map[string]float64)
	  for nodeKey, ecoCounts := range counts {
		  result[nodeKey] = make(map[string]float64)
		  for eco, labelCounts := range ecoCounts {
			  result[nodeKey][eco] = computeEntropy(labelCounts)
		  }
	  }
	  return result
  }
  
  
  
  func CountEdgeEntropyByEco(graphs []*LabeledGraph) map[string]map[string]float64 {
	  counts := make(map[string]map[string]map[int]int)
  
	  for _, lg := range graphs {
		  for _, targets := range lg.Graph.Edges {
			  for _, edge := range targets {
				  edgeKey := fmt.Sprintf("%s|%v", edge.Type, edge.Value)
				  sourceNode, ok := lg.Graph.Nodes[edge.Source]
				  if !ok {
					  continue
				  }
				  eco := sourceNode.Eco
  
				  if _, ok := counts[edgeKey]; !ok {
					  counts[edgeKey] = make(map[string]map[int]int)
				  }
				  if _, ok := counts[edgeKey][eco]; !ok {
					  counts[edgeKey][eco] = make(map[int]int)
				  }
				  counts[edgeKey][eco][lg.Label]++
			  }
		  }
	  }
	  result := make(map[string]map[string]float64)
	  for edgeKey, ecoCounts := range counts {
		  result[edgeKey] = make(map[string]float64)
		  for eco, labelCounts := range ecoCounts {
			  result[edgeKey][eco] = computeEntropy(labelCounts)
		  }
	  }
	  return result
  }
  
  func ComputeSimpleEntropyVariance(entropyByEco map[string]map[string]float64) map[string]float64 {
	  result := make(map[string]float64)
	  for key, ecoMap := range entropyByEco {
		  var values []float64
		  for _, v := range ecoMap {
			  values = append(values, v)
		  }
		  result[key] = computeVariance(values)
	  }
	  return result
  }
 
 func ComputeEntropyVariancePerEco(entropyByEco map[string]map[string]float64) map[string]float64 {
	 ecoEntropyValues := make(map[string][]float64)
 
	 for _, ecoMap := range entropyByEco {
		 for eco, val := range ecoMap {
			 ecoEntropyValues[eco] = append(ecoEntropyValues[eco], val)
		 }
	 }
 
	 ecoVariance := make(map[string]float64)
	 for eco, values := range ecoEntropyValues {
		 ecoVariance[eco] = computeVariance(values)
	 }
 
	 return ecoVariance
 }
 
  
  func ComputeOverallEntropyVariance(entropyByEco map[string]map[string]float64) float64 {
	  var all []float64
	  for _, ecoMap := range entropyByEco {
		  for _, v := range ecoMap {
			  all = append(all, v)
		  }
	  }
	  return computeVariance(all)
 }
  
  func SaveTopVariance(variances map[string]float64, filePath string, topN int) error {
	  type pair struct {
		  Key string
		  Val float64
	  }
	  var sorted []pair
	  for k, v := range variances {
		  sorted = append(sorted, pair{k, v})
	  }
	  sort.Slice(sorted, func(i, j int) bool {
		  return sorted[i].Val > sorted[j].Val
	  })
  
	  f, err := os.Create(filePath)
	  if err != nil {
		  return err
	  }
	  defer f.Close()
  
	  fmt.Fprintf(f, "Key,Variance\n")
	  for i := 0; i < topN && i < len(sorted); i++ {
		  item := sorted[i]
		  fmt.Fprintf(f, "%s,%.6f\n", item.Key, item.Val)
	  }
	  return nil
 }