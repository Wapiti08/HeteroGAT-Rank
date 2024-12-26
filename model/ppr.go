/**
 * @ Create Time: 2024-12-26 13:52:37
 * @ Modified time: 2024-12-26 16:31:47
 * @ Description: implement personalized pagerank with golang
 */

package model

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



// calculate similarity between two node attributes

// helper function to calculate common characters in two strings

// helper function to calculate the intersection of two lists

// compute the PPR using semantic weights

