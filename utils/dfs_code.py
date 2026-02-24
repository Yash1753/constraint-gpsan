"""
DFS Code Implementation for GSPAN
This is the core canonical representation used in GSPAN algorithm
"""

from typing import List, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import copy

@dataclass(frozen=True)
class DFSEdge:
    """
    Single edge in DFS code
    Format: (i, j, label_i, edge_label, label_j)
    where i < j for forward edges, i > j for backward edges
    """
    frm: int          # From vertex ID in DFS tree
    to: int           # To vertex ID in DFS tree
    from_label: int   # Label of from vertex
    edge_label: int   # Label of edge
    to_label: int     # Label of to vertex
    
    def __lt__(self, other):
        """
        Define ordering for DFS edges (crucial for canonical form)
        Forward edges: (i < j)
        Backward edges: (i > j)
        """
        # Compare forward vs backward
        self_forward = self.frm < self.to
        other_forward = other.frm < other.to
        
        if self_forward and not other_forward:
            return True  # Forward edges come first
        if not self_forward and other_forward:
            return False
        
        # Both forward or both backward
        if self_forward:  # Both forward
            # Order by: (i, j, label_i, edge_label, label_j)
            return (self.frm, self.to, self.from_label, self.edge_label, self.to_label) < \
                   (other.frm, other.to, other.from_label, other.edge_label, other.to_label)
        else:  # Both backward
            # Order by: (i, j, edge_label)
            return (self.frm, self.to, self.edge_label) < \
                   (other.frm, other.to, other.edge_label)
    
    def __repr__(self):
        direction = "→" if self.frm < self.to else "←"
        return f"({self.frm}{direction}{self.to}, {self.from_label}-{self.edge_label}-{self.to_label})"

class DFSCode:
    """
    DFS Code: Canonical representation of a graph
    A sequence of edges in DFS traversal order
    """
    
    def __init__(self):
        self.edges: List[DFSEdge] = []
        self.support = 0
        self.graph_ids: Set[int] = set()
    
    def append(self, edge: DFSEdge):
        """Add edge to DFS code"""
        self.edges.append(edge)
    
    def __len__(self):
        return len(self.edges)
    
    def __getitem__(self, idx):
        return self.edges[idx]
    
    def __repr__(self):
        return f"DFSCode({self.edges})"
    
    def __eq__(self, other):
        if not isinstance(other, DFSCode):
            return False
        return self.edges == other.edges
    
    def __lt__(self, other):
        """
        Compare two DFS codes lexicographically
        Used to determine canonical (minimum) DFS code
        """
        for i in range(min(len(self.edges), len(other.edges))):
            if self.edges[i] < other.edges[i]:
                return True
            elif self.edges[i] > other.edges[i]:
                return False
        # If all edges equal so far, shorter code is smaller
        return len(self.edges) < len(other.edges)
    
    def __hash__(self):
        return hash(tuple(self.edges))
    
    def copy(self):
        """Deep copy of DFS code"""
        new_code = DFSCode()
        new_code.edges = self.edges.copy()
        new_code.support = self.support
        new_code.graph_ids = self.graph_ids.copy()
        return new_code
    
    def get_num_vertices(self) -> int:
        """Get number of vertices in the pattern"""
        vertices = set()
        for edge in self.edges:
            vertices.add(edge.frm)
            vertices.add(edge.to)
        return len(vertices)
    
    def get_vertex_labels(self) -> List[int]:
        """Get all vertex labels in order"""
        vertex_labels = {}
        for edge in self.edges:
            if edge.frm not in vertex_labels:
                vertex_labels[edge.frm] = edge.from_label
            if edge.to not in vertex_labels:
                vertex_labels[edge.to] = edge.to_label
        
        max_id = max(vertex_labels.keys()) if vertex_labels else -1
        labels = [vertex_labels.get(i, -1) for i in range(max_id + 1)]
        return labels
    
    def get_rightmost_path(self) -> List[int]:
        """
        Get rightmost path in DFS tree
        Critical for GSPAN's rightmost extension
        """
        if not self.edges:
            return []
        
        # Build DFS tree
        rightmost_vertex = max(max(e.frm, e.to) for e in self.edges)
        
        # Trace back to root
        path = [rightmost_vertex]
        current = rightmost_vertex
        
        # Find parent of each vertex
        parent = {}
        for edge in self.edges:
            if edge.frm < edge.to:  # Forward edge
                parent[edge.to] = edge.frm
        
        while current in parent:
            current = parent[current]
            path.append(current)
        
        return path
    
    def to_graph(self):
        """Convert DFS code to Graph object"""
        from .graph_loader import Graph
        
        g = Graph()
        
        # Add vertices
        vertex_labels = self.get_vertex_labels()
        for vid, vlabel in enumerate(vertex_labels):
            if vlabel != -1:
                g.add_vertex(vid, vlabel)
        
        # Add edges (avoid duplicates)
        added_edges = set()
        for edge in self.edges:
            edge_key = (min(edge.frm, edge.to), max(edge.frm, edge.to), edge.edge_label)
            if edge_key not in added_edges:
                g.add_edge(edge.frm, edge.to, edge.edge_label)
                added_edges.add(edge_key)
        
        return g

class DFSCodeBuilder:
    """Helper class to build DFS codes from graphs"""
    
    @staticmethod
    def from_graph(graph, start_vertex=None):
        """
        Generate all possible DFS codes from a graph
        (For finding minimum DFS code)
        """
        from .graph_loader import Graph
        
        if not graph.vertices:
            return []
        
        dfs_codes = []
        
        # Try starting from each vertex
        start_vertices = [start_vertex] if start_vertex is not None else range(len(graph.vertices))
        
        for start in start_vertices:
            if start >= len(graph.vertices):
                continue
            
            # Generate DFS code starting from this vertex
            code = DFSCode()
            visited = set()
            vertex_to_dfs_id = {}
            
            def dfs(v, dfs_id):
                visited.add(v)
                vertex_to_dfs_id[v] = dfs_id
                current_dfs_id = dfs_id
                
                # Get neighbors
                neighbors = []
                for edge in graph.edges:
                    if edge.frm == v:
                        neighbors.append((edge.to, edge.elabel))
                    elif edge.to == v:
                        neighbors.append((edge.frm, edge.elabel))
                
                # Sort neighbors for consistent traversal
                neighbors.sort()
                
                for neighbor, edge_label in neighbors:
                    if neighbor not in visited:
                        # Forward edge
                        current_dfs_id += 1
                        dfs_edge = DFSEdge(
                            frm=dfs_id,
                            to=current_dfs_id,
                            from_label=graph.vertices[v].label,
                            edge_label=edge_label,
                            to_label=graph.vertices[neighbor].label
                        )
                        code.append(dfs_edge)
                        dfs(neighbor, current_dfs_id)
                    elif vertex_to_dfs_id[neighbor] < dfs_id:
                        # Backward edge
                        dfs_edge = DFSEdge(
                            frm=dfs_id,
                            to=vertex_to_dfs_id[neighbor],
                            from_label=graph.vertices[v].label,
                            edge_label=edge_label,
                            to_label=graph.vertices[neighbor].label
                        )
                        code.append(dfs_edge)
            
            dfs(start, 0)
            
            if len(visited) == len(graph.vertices):  # Connected graph
                dfs_codes.append(code)
        
        return dfs_codes
    
    @staticmethod
    def get_min_dfs_code(graph):
        """
        Get minimum (canonical) DFS code for a graph
        This is THE key operation in GSPAN
        """
        dfs_codes = DFSCodeBuilder.from_graph(graph)
        if not dfs_codes:
            return DFSCode()
        
        # Return lexicographically smallest DFS code
        return min(dfs_codes)