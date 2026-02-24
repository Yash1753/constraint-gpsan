"""
Proper GSPAN Algorithm Implementation
Based on: "gSpan: Graph-Based Substructure Pattern Mining" (Yan & Han, 2002)
"""

import time
from collections import defaultdict
from typing import List, Set, Dict, Tuple
from .graph_loader import Graph, Pattern
from .dfs_code import DFSCode, DFSEdge, DFSCodeBuilder
from .constraints import Constraint, ConstraintManager

class ProperGSPAN:
    """
    Real GSPAN algorithm with DFS codes and canonical form checking
    """
    
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 constraints: List[Constraint] = None,
                 max_pattern_size: int = None,
                 verbose: bool = True):
        
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.constraint_manager = ConstraintManager(constraints or [])
        self.max_pattern_size = max_pattern_size or 10  # Prevent explosion
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            'patterns_found': 0,
            'candidates_generated': 0,
            'non_minimal_pruned': 0,
            'constraint_pruned': 0,
            'support_pruned': 0,
            'runtime': 0
        }
        
        # Frequent patterns (stored as DFS codes)
        self.frequent_patterns: List[DFSCode] = []
        
        if self.verbose:
            self._print_init()
    
    def _print_init(self):
        """Print initialization"""
        print(f"\n{'='*70}")
        print(f"{'PROPER GSPAN ALGORITHM':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {self.min_support_count} ({self.min_support_count/len(self.database)*100:.1f}%)")
        print(f"  Max pattern size: {self.max_pattern_size}")
        print(f"  Constraints: {len(self.constraint_manager.constraints)}")
        print(f"{'='*70}\n")
    
    def mine(self) -> List[Tuple[Pattern, int]]:
        """
        Main GSPAN mining algorithm
        Returns list of (Pattern, support) tuples
        """
        
        if self.verbose:
            print("Starting GSPAN mining...")
        
        start_time = time.time()
        
        # Step 1: Find frequent 1-edge patterns
        frequent_1edge = self._find_frequent_1edge_patterns()
        
        if self.verbose:
            print(f"Found {len(frequent_1edge)} frequent 1-edge patterns\n")
        
        # Step 2: Recursively grow patterns using DFS
        for edge_pattern in frequent_1edge:
            self._gspan_recursive(edge_pattern)
        
        self.stats['runtime'] = time.time() - start_time
        
        # Convert DFS codes to Pattern objects
        result_patterns = []
        for dfs_code in self.frequent_patterns:
            pattern = self._dfs_code_to_pattern(dfs_code)
            result_patterns.append((pattern, dfs_code.support))
        
        if self.verbose:
            self._print_results(result_patterns)
        
        return result_patterns
    
    def _find_frequent_1edge_patterns(self) -> List[DFSCode]:
        """
        Find all frequent single-edge patterns
        This is the base case for GSPAN
        """
        
        edge_counts = defaultdict(set)  # (from_label, edge_label, to_label) -> set of graph IDs
        
        # Count edge patterns across all graphs
        for graph in self.database:
            seen_in_graph = set()
            
            for edge in graph.edges:
                v_from = graph.vertex_map[edge.frm]
                v_to = graph.vertex_map[edge.to]
                
                # Create edge signature (undirected, so normalize)
                if v_from.label <= v_to.label:
                    edge_sig = (v_from.label, edge.elabel, v_to.label)
                else:
                    edge_sig = (v_to.label, edge.elabel, v_from.label)
                
                if edge_sig not in seen_in_graph:
                    edge_counts[edge_sig].add(graph.gid)
                    seen_in_graph.add(edge_sig)
        
        # Create DFS codes for frequent edges
        frequent_edges = []
        
        for (from_label, edge_label, to_label), graph_ids in edge_counts.items():
            if len(graph_ids) >= self.min_support_count:
                # Create minimal DFS code for single edge
                dfs_code = DFSCode()
                dfs_edge = DFSEdge(
                    frm=0,
                    to=1,
                    from_label=from_label,
                    edge_label=edge_label,
                    to_label=to_label
                )
                dfs_code.append(dfs_edge)
                dfs_code.support = len(graph_ids)
                dfs_code.graph_ids = graph_ids
                
                # Check constraints
                pattern = self._dfs_code_to_pattern(dfs_code)
                if self.constraint_manager.can_satisfy_all(pattern):
                    frequent_edges.append(dfs_code)
        
        return sorted(frequent_edges)  # Sort for consistent ordering
    
    def _gspan_recursive(self, dfs_code: DFSCode):
        """
        Recursive GSPAN mining from a DFS code
        Core of the algorithm
        """
        
        self.stats['candidates_generated'] += 1
        
        # Check if this is a minimal (canonical) DFS code
        if not self._is_min_dfs_code(dfs_code):
            self.stats['non_minimal_pruned'] += 1
            return
        
        # Check constraints
        pattern = self._dfs_code_to_pattern(dfs_code)
        if not self.constraint_manager.check_antimonotone(pattern):
            self.stats['constraint_pruned'] += 1
            return
        
        # Check if frequent
        support = self._compute_support(dfs_code)
        dfs_code.support = support
        
        if support < self.min_support_count:
            self.stats['support_pruned'] += 1
            return
        
        # This is a frequent pattern!
        if self.constraint_manager.check_monotone(pattern):
            self.frequent_patterns.append(dfs_code.copy())
            self.stats['patterns_found'] += 1
        
        # Stop if max size reached
        if dfs_code.get_num_vertices() >= self.max_pattern_size:
            return
        
        # Generate all possible extensions (rightmost path extension)
        extensions = self._generate_extensions(dfs_code)
        
        # Recursively mine each extension
        for extended_code in extensions:
            self._gspan_recursive(extended_code)
    
    def _is_min_dfs_code(self, dfs_code: DFSCode) -> bool:
        """
        Check if DFS code is minimal (canonical)
        This is THE key operation that makes GSPAN efficient
        """
        
        # Convert to graph
        graph = dfs_code.to_graph()
        
        # Get minimum DFS code for this graph
        min_code = DFSCodeBuilder.get_min_dfs_code(graph)
        
        # Check if current code equals minimum code
        return dfs_code == min_code
    
    def _compute_support(self, dfs_code: DFSCode) -> int:
        """
        Compute support by checking subgraph isomorphism
        Uses proper subgraph matching
        """
        
        pattern_graph = dfs_code.to_graph()
        support = 0
        graph_ids = set()
        
        for graph in self.database:
            if self._is_subgraph(pattern_graph, graph):
                support += 1
                graph_ids.add(graph.gid)
        
        dfs_code.graph_ids = graph_ids
        return support
    
    def _is_subgraph(self, pattern: Graph, graph: Graph) -> bool:
        """
        Check if pattern is a subgraph of graph
        Simplified VF2-like algorithm
        """
        
        # Quick label check first
        pattern_vertex_labels = defaultdict(int)
        for v in pattern.vertices:
            pattern_vertex_labels[v.label] += 1
        
        graph_vertex_labels = defaultdict(int)
        for v in graph.vertices:
            graph_vertex_labels[v.label] += 1
        
        # Pattern labels must be subset
        for label, count in pattern_vertex_labels.items():
            if graph_vertex_labels[label] < count:
                return False
        
        # If pattern is larger than graph, can't be subgraph
        if len(pattern.vertices) > len(graph.vertices):
            return False
        
        # For small patterns, use backtracking search
        return self._subgraph_search(pattern, graph)
    
    def _subgraph_search(self, pattern: Graph, graph: Graph) -> bool:
        """
        Backtracking search for subgraph isomorphism
        Simplified but correct version
        """
        
        if len(pattern.vertices) == 0:
            return True
        
        # Try to map pattern vertices to graph vertices
        mapping = {}
        
        def is_valid_mapping():
            # Check if current mapping preserves edges
            for edge in pattern.edges:
                if edge.frm in mapping and edge.to in mapping:
                    # Check if corresponding edge exists in graph
                    mapped_from = mapping[edge.frm]
                    mapped_to = mapping[edge.to]
                    
                    # Look for edge in graph
                    found = False
                    for g_edge in graph.edges:
                        if ((g_edge.frm == mapped_from and g_edge.to == mapped_to) or
                            (g_edge.frm == mapped_to and g_edge.to == mapped_from)):
                            if g_edge.elabel == edge.elabel:
                                found = True
                                break
                    
                    if not found:
                        return False
            return True
        
        def backtrack(pattern_v_idx):
            if pattern_v_idx >= len(pattern.vertices):
                return is_valid_mapping()
            
            pattern_v = pattern.vertices[pattern_v_idx]
            
            for graph_v in graph.vertices:
                # Check if labels match and not already mapped
                if graph_v.label == pattern_v.label and graph_v.vid not in mapping.values():
                    mapping[pattern_v.vid] = graph_v.vid
                    
                    if is_valid_mapping():
                        if backtrack(pattern_v_idx + 1):
                            return True
                    
                    del mapping[pattern_v.vid]
            
            return False
        
        return backtrack(0)
    
    def _generate_extensions(self, dfs_code: DFSCode) -> List[DFSCode]:
        """
        Generate all valid rightmost path extensions
        Core of GSPAN's search strategy
        """
        
        extensions = []
        rightmost_path = dfs_code.get_rightmost_path()
        
        if not rightmost_path:
            return extensions
        
        rightmost_vertex = rightmost_path[0]
        
        # Get all possible labels from database
        vertex_labels = self._get_frequent_vertex_labels()
        edge_labels = self._get_frequent_edge_labels()
        
        # Extension 1: Forward extension from rightmost vertex
        for v_label in vertex_labels:
            for e_label in edge_labels:
                new_code = dfs_code.copy()
                new_vertex_id = dfs_code.get_num_vertices()
                
                # Get label of rightmost vertex
                rightmost_label = None
                for edge in dfs_code.edges:
                    if edge.to == rightmost_vertex:
                        rightmost_label = edge.to_label
                        break
                    if edge.frm == rightmost_vertex:
                        rightmost_label = edge.from_label
                        break
                
                if rightmost_label is None and len(dfs_code.edges) > 0:
                    rightmost_label = dfs_code.edges[-1].to_label
                
                if rightmost_label is not None:
                    new_edge = DFSEdge(
                        frm=rightmost_vertex,
                        to=new_vertex_id,
                        from_label=rightmost_label,
                        edge_label=e_label,
                        to_label=v_label
                    )
                    new_code.append(new_edge)
                    extensions.append(new_code)
        
        # Extension 2: Backward extension to vertices on rightmost path
        for path_vertex in rightmost_path[1:]:  # Exclude rightmost itself
            for e_label in edge_labels:
                new_code = dfs_code.copy()
                
                # Get labels
                rightmost_label = None
                path_label = None
                
                for edge in dfs_code.edges:
                    if edge.to == rightmost_vertex or edge.frm == rightmost_vertex:
                        rightmost_label = edge.to_label if edge.to == rightmost_vertex else edge.from_label
                    if edge.to == path_vertex or edge.frm == path_vertex:
                        path_label = edge.to_label if edge.to == path_vertex else edge.from_label
                
                if rightmost_label and path_label:
                    new_edge = DFSEdge(
                        frm=rightmost_vertex,
                        to=path_vertex,
                        from_label=rightmost_label,
                        edge_label=e_label,
                        to_label=path_label
                    )
                    new_code.append(new_edge)
                    extensions.append(new_code)
        
        return extensions
    
    def _get_frequent_vertex_labels(self) -> Set[int]:
        """Get all vertex labels that appear frequently enough"""
        label_counts = defaultdict(set)
        
        for graph in self.database:
            for vertex in graph.vertices:
                label_counts[vertex.label].add(graph.gid)
        
        return {label for label, gids in label_counts.items() 
                if len(gids) >= self.min_support_count}
    
    def _get_frequent_edge_labels(self) -> Set[int]:
        """Get all edge labels that appear frequently enough"""
        label_counts = defaultdict(set)
        
        for graph in self.database:
            for edge in graph.edges:
                label_counts[edge.elabel].add(graph.gid)
        
        # Default to {0} if no edge labels
        if not label_counts:
            return {0}
        
        frequent = {label for label, gids in label_counts.items() 
                   if len(gids) >= self.min_support_count}
        
        return frequent if frequent else {0}
    
    def _dfs_code_to_pattern(self, dfs_code: DFSCode) -> Pattern:
        """Convert DFS code to Pattern object"""
        from .graph_loader import Pattern
        
        pattern = Pattern()
        
        # Get vertex labels
        vertex_labels = dfs_code.get_vertex_labels()
        
        # Add vertices
        for vlabel in vertex_labels:
            if vlabel != -1:
                pattern.add_vertex(vlabel)
        
        # Add edges (avoid duplicates)
        added_edges = set()
        for dfs_edge in dfs_code.edges:
            edge_key = (min(dfs_edge.frm, dfs_edge.to), 
                       max(dfs_edge.frm, dfs_edge.to), 
                       dfs_edge.edge_label)
            if edge_key not in added_edges:
                pattern.add_edge(dfs_edge.frm, dfs_edge.to, dfs_edge.edge_label)
                added_edges.add(edge_key)
        
        pattern.support = dfs_code.support
        pattern.graph_ids = dfs_code.graph_ids
        
        return pattern
    
    def _print_results(self, patterns: List[Tuple[Pattern, int]]):
        """Print mining results"""
        print(f"\n{'='*70}")
        print(f"{'GSPAN MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results:")
        print(f"  Frequent patterns: {len(patterns)}")
        print(f"  Runtime: {self.stats['runtime']:.2f}s")
        
        print(f"\n📈 Statistics:")
        print(f"  Candidates generated: {self.stats['candidates_generated']:,}")
        print(f"  Non-minimal pruned: {self.stats['non_minimal_pruned']:,}")
        print(f"  Constraint pruned: {self.stats['constraint_pruned']:,}")
        print(f"  Support pruned: {self.stats['support_pruned']:,}")
        
        if self.stats['candidates_generated'] > 0:
            prune_rate = ((self.stats['non_minimal_pruned'] + 
                          self.stats['constraint_pruned'] + 
                          self.stats['support_pruned']) / 
                         self.stats['candidates_generated'] * 100)
            print(f"  Overall pruning: {prune_rate:.1f}%")
        
        print(f"\n🏆 Top 10 Patterns:")
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns[:10], 1):
            labels = [v.label for v in pattern.vertices]
            print(f"  {i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                  f"Edges={len(pattern.edges)}, Support={support}")
        
        print(f"\n{'='*70}\n")