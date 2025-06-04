/*
defines Graph, Node, Edge structures and JSON loading

*/

package main

import (
	"encoding/json"
	"os"
)

type Node struct {
	Value string `json:"Value"`
	Type  string `json:"Type"`
	Eco   string `json:"Eco"`
}

type Edge struct {
	Source string `json:"Source"`
	Target string `json:"Target"`
	Value interface{} `json:"Value"`
	Type string `json:"Type"`
}

type Graph struct {
	Nodes map[string]Node			`json:"Nodes"`
	Edges map[string]map[string]Edge	`json:"Edges"`
}

type LabeledGraph struct {
	Graph *Graph
	Label int
}

type GraphFile struct {
	Label int             `json:"Label"`
	Nodes map[string]Node `json:"Nodes"`
	Edges map[string]map[string]Edge `json:"Edges"`
}


func LoadLabeledGraph(filePath string, label int) (*LabeledGraph, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var gf GraphFile
	// read graph in json format
	err = json.Unmarshal(data, &g)
	if err != nil {
		return nil, err
	}
	
	return &LabeledGraph{
		Graph: &Graph{Nodes: gf.Nodes, Edges: gf.Edges},
		Label: gf.Label,
	}, nil}