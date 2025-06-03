/*
defines Graph, Node, Edge structures and JSON loading

*/

package main

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

