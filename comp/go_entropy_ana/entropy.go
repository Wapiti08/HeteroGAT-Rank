/**
 * @ Create Time: 2024-12-26 13:52:37
 * @ Modified time: 2025-06-04 15:13:07
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

func CountEdgeValueEntropy(graphs []*LabeledGraph) map[string]float64 {
	counts := make(map[string]map[int]int)
	for _, lg := range graphs {
		for _, targetMap := range lg.Graph.Edges {
			for _, edge := range targetMap {
				key := fmt.Sprintf("%s|%v", edge.Type, edge.Value)
				if _, ok := counts[key]; !ok {
					counts[key] = make(map[int]int)
				}
				counts[key][lg.Label]++
			}
		}
	}
	result := make(map[string]float64)
	for val, labelCounts := range counts {
		result[val] = computeEntropy(labelCounts)
	}
	return result
}

func SaveTopEntropy(entropies map[string]float64, filePath string, topN int) error {
	type pair struct {
		Key string
		Val float64
	}

	var sorted []pair
	for k, v := range entropies {
		sorted = append(sorted, pair{k,v})
	}
	// sort with a custom closure
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Val < sorted[j].Val
	})

	// save to a file
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}

	defer f.Close()

	fmt.Fprintf(f, "Item, Entropy \n")

	for i:=0; i<topN && i <len(sorted); i++ {
		fmt.Fprintf(f, "%s,%.4f\n", sorted[i].Key, sorted[i].Val)
	}
	return nil

}