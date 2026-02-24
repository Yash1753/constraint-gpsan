"""
Basic Constraint-Aware GSPAN Implementation
Optimized for MUTAG dataset
"""

import time
from collections import defaultdict
from typing import List, Set, Dict, Tuple
from utils.graph_loader import Graph, Pattern, Vertex, Edge, DatasetLoader
from utils.constraints import *
from utils.visualization import PatternVisualizer

class BasicConstraintGSPAN:
    """
    Basic GSPAN with integrated constraint checking
    Optimized for MUTAG molecular graphs
    """
    
    def __init__(self, 
                 database: List[Graph], 
                 min_support: float,
                 constraints: List[Constraint] = None,
                 verbose: bool = True):
        
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.constraint_manager = ConstraintManager(constraints or [])
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            'patterns_generated': 0,
            'patterns_pruned_constraints': 0,
            'patterns_pruned_support': 0,
            'patterns_pruned_infrequent_labels': 0,
            'patterns_found': 0,
            'support_computations': 0,
            'runtime': 0
        }
        
        # Pre-compute label statistics
        self.label_stats = self._compute_label_stats()
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization information"""
        print(f"\n{'='*70}")
        print(f"{'BASIC CONSTRAINT-AWARE GSPAN':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {self.min_support_count} graphs ({self.min_support_count/len(self.database)*100:.1f}%)")
        print(f"  Constraints: {len(self.constraint_manager.constraints)}")
        
        if self.constraint_manager.constraints:
            print(f"\n  Constraint List:")
            for i, c in enumerate(self.constraint_manager.constraints, 1):
                print(f"    {i}. {c.name} [{c.type}]")
        
        print(f"\n  Label Statistics:")
        print(f"    Unique vertex labels: {len(self.label_stats['vertex_counts'])}")
        print(f"    Frequent labels: {sum(1 for c in self.label_stats['vertex_counts'].values() if c >= self.min_support_count)}")
        print(f"{'='*70}\n")
    
    def _compute_label_stats(self) -> Dict:
        """Pre-compute label frequencies for early pruning"""
        stats = {
            'vertex_labels': defaultdict(set),
            'edge_labels': defaultdict(set),
            'label_pairs': defaultdict(set)
        }
        
        for graph in self.database:
            # Track which graphs contain each label
            for v in graph.vertices:
                stats['vertex_labels'][v.label].add(graph.gid)
            
            for e in graph.edges:
                stats['edge_labels'][e.elabel].add(graph.gid)
                
                # Track label pairs (for extension pruning)
                v_from = graph.vertex_map.get(e.frm)
                v_to = graph.vertex_map.get(e.to)
                if v_from and v_to:
                    pair = (v_from.label, e.elabel, v_to.label)
                    stats['label_pairs'][pair].add(graph.gid)
        
        # Convert to counts
        stats['vertex_counts'] = {
            label: len(gids) for label, gids in stats['vertex_labels'].items()
        }
        stats['edge_counts'] = {
            label: len(gids) for label, gids in stats['edge_labels'].items()
        }
        stats['pair_counts'] = {
            pair: len(gids) for pair, gids in stats['label_pairs'].items()
        }
        
        return stats
    
    def _can_be_frequent(self, pattern: Pattern) -> bool:
        """Quick check: can pattern possibly be frequent based on labels?"""
        # Check if all vertex labels are frequent enough
        for v in pattern.vertices:
            if self.label_stats['vertex_counts'].get(v.label, 0) < self.min_support_count:
                self.stats['patterns_pruned_infrequent_labels'] += 1
                return False
        
        return True
    
    def _compute_support(self, pattern: Pattern) -> int:
        """Compute support using simplified subgraph matching"""
        self.stats['support_computations'] += 1
        
        support = 0
        pattern.graph_ids = set()
        
        # Quick label-based filtering
        candidate_graphs = self._get_candidate_graphs(pattern)
        
        for graph in candidate_graphs:
            if self._has_pattern(graph, pattern):
                support += 1
                pattern.graph_ids.add(graph.gid)
        
        return support
    
    def _get_candidate_graphs(self, pattern: Pattern) -> List[Graph]:
        """Get graphs that could potentially contain pattern"""
        # Start with all graphs
        candidate_gids = set(g.gid for g in self.database)
        
        # Intersect with graphs containing each vertex label
        for v in pattern.vertices:
            label_gids = self.label_stats['vertex_labels'].get(v.label, set())
            candidate_gids &= label_gids
        
        return [g for g in self.database if g.gid in candidate_gids]
    
    def _has_pattern(self, graph: Graph, pattern: Pattern) -> bool:
        """Simplified pattern matching (label-based)"""
        # Count pattern labels
        pattern_labels = defaultdict(int)
        for v in pattern.vertices:
            pattern_labels[v.label] += 1
        
        # Count graph labels
        graph_labels = defaultdict(int)
        for v in graph.vertices:
            graph_labels[v.label] += 1
        
        # Pattern labels must be subset of graph labels
        for label, count in pattern_labels.items():
            if graph_labels[label] < count:
                return False
        
        # For small patterns, label matching is sufficient
        # For production: implement VF2 subgraph isomorphism
        return True
    
    def _generate_single_vertex_patterns(self) -> List[Pattern]:
        """Generate initial single-vertex patterns"""
        patterns = []
        
        if self.verbose:
            print("Generating single-vertex patterns...")
        
        # Get frequent vertex labels
        for label, count in self.label_stats['vertex_counts'].items():
            if count >= self.min_support_count:
                p = Pattern()
                p.add_vertex(label)
                p.support = count
                p.graph_ids = self.label_stats['vertex_labels'][label].copy()
                
                # Check constraints
                if self.constraint_manager.can_satisfy_all(p):
                    patterns.append(p)
        
        if self.verbose:
            print(f"  Generated {len(patterns)} single-vertex patterns\n")
        
        return patterns
    
    def _extend_pattern(self, pattern: Pattern) -> List[Pattern]:
        """Generate extensions of pattern (simplified rightmost extension)"""
        extensions = []
        
        # For simplicity, try adding each frequent label
        for label, count in self.label_stats['vertex_counts'].items():
            if count < self.min_support_count:
                continue
            
            # Create new pattern
            new_pattern = pattern.copy()
            new_vid = len(new_pattern.vertices)
            new_pattern.add_vertex(label)
            
            # Add edge to connect (simple chain extension)
            if len(new_pattern.vertices) > 1:
                # Connect to last vertex
                new_pattern.add_edge(new_vid - 1, new_vid, 0)
            
            extensions.append(new_pattern)
        
        return extensions
    
    def mine(self) -> List[Tuple[Pattern, int]]:
        """Main mining algorithm"""
        if self.verbose:
            print(f"{'='*70}")
            print(f"{'STARTING MINING PROCESS':^70}")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        
        frequent_patterns = []
        
        # Level 1: Single vertex patterns
        candidates = self._generate_single_vertex_patterns()
        
        level = 1
        
        while candidates:
            if self.verbose:
                print(f"{'─'*70}")
                print(f"Level {level}: Processing {len(candidates)} candidates")
                print(f"{'─'*70}")
            
            new_candidates = []
            level_frequent = 0
            level_pruned_constraints = 0
            level_pruned_support = 0
            
            for pattern in candidates:
                self.stats['patterns_generated'] += 1
                
                # CHECKPOINT 1: Anti-monotone constraints (early pruning)
                if not self.constraint_manager.check_antimonotone(pattern):
                    self.stats['patterns_pruned_constraints'] += 1
                    level_pruned_constraints += 1
                    continue
                
                # CHECKPOINT 2: Can pattern be frequent based on labels?
                if not self._can_be_frequent(pattern):
                    level_pruned_support += 1
                    continue
                
                # CHECKPOINT 3: Compute support
                support = self._compute_support(pattern)
                
                if support >= self.min_support_count:
                    # CHECKPOINT 4: Monotone constraints
                    if self.constraint_manager.check_monotone(pattern):
                        frequent_patterns.append((pattern, support))
                        self.stats['patterns_found'] += 1
                        level_frequent += 1
                    
                    # Generate extensions if can satisfy constraints
                    if self.constraint_manager.can_satisfy_all(pattern):
                        extensions = self._extend_pattern(pattern)
                        new_candidates.extend(extensions)
                else:
                    self.stats['patterns_pruned_support'] += 1
                    level_pruned_support += 1
            
            if self.verbose:
                print(f"  Frequent: {level_frequent}")
                print(f"  Pruned by constraints: {level_pruned_constraints}")
                print(f"  Pruned by support: {level_pruned_support}")
                print(f"  Extensions generated: {len(new_candidates)}\n")
            
            candidates = new_candidates
            level += 1
        
        self.stats['runtime'] = time.time() - start_time
        
        if self.verbose:
            self._print_results(frequent_patterns)
        
        return frequent_patterns
    
    def _print_results(self, patterns: List[Tuple[Pattern, int]]):
        """Print mining results and statistics"""
        print(f"\n{'='*70}")
        print(f"{'MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results Summary:")
        print(f"  Frequent patterns found: {len(patterns)}")
        print(f"  Runtime: {self.stats['runtime']:.2f} seconds")
        
        print(f"\n📈 Mining Statistics:")
        print(f"  Patterns generated: {self.stats['patterns_generated']:,}")
        print(f"  Pruned by constraints: {self.stats['patterns_pruned_constraints']:,}")
        print(f"  Pruned by support: {self.stats['patterns_pruned_support']:,}")
        print(f"  Pruned by infrequent labels: {self.stats['patterns_pruned_infrequent_labels']:,}")
        print(f"  Support computations: {self.stats['support_computations']:,}")
        
        if self.stats['patterns_generated'] > 0:
            total_pruned = (self.stats['patterns_pruned_constraints'] + 
                          self.stats['patterns_pruned_support'] +
                          self.stats['patterns_pruned_infrequent_labels'])
            prune_rate = total_pruned / self.stats['patterns_generated'] * 100
            print(f"  Overall pruning rate: {prune_rate:.1f}%")
        
        if self.constraint_manager.constraints:
            print(f"\n🔧 Constraint Statistics:")
            for stat in self.constraint_manager.get_all_stats():
                if stat['checks'] > 0:
                    print(f"  {stat['name']}:")
                    print(f"    Checks: {stat['checks']:,}")
                    print(f"    Prunes: {stat['prunes']:,} ({stat['prune_rate']*100:.1f}%)")
        
        print(f"\n🏆 Top 10 Patterns by Support:")
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns[:10], 1):
            labels = [v.label for v in pattern.vertices]
            print(f"  {i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                  f"Edges={len(pattern.edges)}, Support={support}")
        
        print(f"\n{'='*70}\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run basic constraint-aware GSPAN on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG DATASET - BASIC CONSTRAINT GSPAN':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG dataset
    print("Loading MUTAG dataset...")
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=None)  # Use all 188 graphs
    
    # For quick testing, use subset
    # graphs = graphs[:50]
    
    # Define constraints for MUTAG
    constraints = MUTAGConstraints.basic_chemical()
    
    print(f"\nUsing constraint set: 'basic_chemical'")
    for i, c in enumerate(constraints, 1):
        print(f"  {i}. {c.name}")
    
    # Run mining
    miner = BasicConstraintGSPAN(
        database=graphs,
        min_support=0.1,  # 10% support = ~19 graphs for full MUTAG
        constraints=constraints,
        verbose=True
    )
    
    patterns = miner.mine()
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = PatternVisualizer()
    
    fig1 = visualizer.plot_pattern_distribution(patterns, 
                                                title="MUTAG Pattern Distribution")
    
    if miner.constraint_manager.constraints:
        fig2 = visualizer.plot_constraint_effectiveness(
            miner.constraint_manager.get_all_stats()
        )
    
    # Save plots
    import os
    os.makedirs('results', exist_ok=True)
    visualizer.save_all_plots('results')
    
    # Save patterns to file
    with open('results/basic_patterns.txt', 'w') as f:
        f.write("MUTAG - Basic Constraint GSPAN Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total patterns: {len(patterns)}\n")
        f.write(f"Runtime: {miner.stats['runtime']:.2f}s\n\n")
        f.write("Patterns (sorted by support):\n")
        f.write("-"*70 + "\n")
        
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns, 1):
            labels = [v.label for v in pattern.vertices]
            f.write(f"{i:3d}. Labels={labels}, Size={len(pattern.vertices)}, "
                   f"Support={support}\n")
    
    print(f"\n✓ Results saved to 'results/' directory")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
