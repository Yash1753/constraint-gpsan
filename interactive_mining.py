"""
NOVEL CONTRIBUTION 2: Interactive Constraint Refinement with Proper GSPAN
Optimized for MUTAG dataset

User iteratively explores patterns and refines constraints based on results
Integrates with real GSPAN algorithm for correct pattern mining
"""

import time
from typing import List, Tuple, Dict
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import *
from utils.gspan_algorithm import ProperGSPAN  # ← Use proper GSPAN!
from utils.visualization import PatternVisualizer

class InteractiveConstraintProperGSPAN:
    """
    Interactive mining with proper GSPAN
    User refines constraints based on actual subgraph patterns
    
    APPROACH:
    1. Mine with current constraints using proper GSPAN
    2. Show results to user
    3. User adds/removes constraints
    4. Repeat until satisfied
    """
    
    def __init__(self, 
                 database: List[Graph], 
                 min_support: float,
                 max_pattern_size: int = 6,
                 verbose: bool = True):
        
        self.database = database
        self.min_support = min_support
        self.min_support_count = max(1, int(min_support * len(database)))
        self.max_pattern_size = max_pattern_size
        self.constraints = []
        self.session_history = []
        self.verbose = verbose
        
        # Statistics
        self.stats = {
            'total_mining_time': 0,
            'total_rounds': 0,
            'constraints_added': 0,
            'constraints_removed': 0
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"{'INTERACTIVE CONSTRAINT MINING + PROPER GSPAN':^70}")
            print(f"{'='*70}")
            print(f"  Database: {len(database)} graphs")
            print(f"  Min support: {self.min_support_count} ({self.min_support*100:.0f}%)")
            print(f"  Max pattern size: {max_pattern_size}")
            print(f"{'='*70}\n")
    
    def _mine_with_current_constraints(self) -> List[Tuple[Pattern, int]]:
        """
        Mine using proper GSPAN with current constraints
        This is the KEY difference from the original version
        """
        
        if self.verbose:
            print(f"  Mining with Proper GSPAN...")
        
        # Use PROPER GSPAN algorithm
        gspan = ProperGSPAN(
            database=self.database,
            min_support=self.min_support,
            constraints=self.constraints,
            max_pattern_size=self.max_pattern_size,
            verbose=False  # Suppress GSPAN output during session
        )
        
        patterns = gspan.mine()
        
        return patterns
    
    def _display_patterns(self, patterns: List[Tuple[Pattern, int]], limit: int = 10):
        """Display sample patterns to user with detailed information"""
        print(f"\n  Showing {min(limit, len(patterns))} sample patterns:")
        print(f"  {'─'*66}")
        print(f"  {'#':<4} {'Labels':<20} {'V':<4} {'E':<4} {'Diam':<6} {'Support':<8}")
        print(f"  {'─'*66}")
        
        for i, (pattern, support) in enumerate(patterns[:limit], 1):
            labels = [v.label for v in pattern.vertices]
            labels_str = str(labels)[:18] + '..' if len(str(labels)) > 20 else str(labels)
            
            # Calculate diameter
            diameter = self._calculate_diameter(pattern)
            
            print(f"  {i:<4} {labels_str:<20} {len(pattern.vertices):<4} "
                  f"{len(pattern.edges):<4} {diameter:<6} {support:<8}")
        
        if len(patterns) > limit:
            print(f"  {'─'*66}")
            print(f"  ... and {len(patterns) - limit} more patterns")
    
    def _calculate_diameter(self, pattern: Pattern) -> int:
        """Calculate diameter of pattern (longest shortest path)"""
        if len(pattern.vertices) <= 1:
            return 0
        
        from collections import deque
        
        # Build adjacency list
        adj = defaultdict(list)
        for e in pattern.edges:
            adj[e.frm].append(e.to)
            adj[e.to].append(e.frm)
        
        max_dist = 0
        
        # BFS from each vertex
        for start_vid in range(len(pattern.vertices)):
            dist = {start_vid: 0}
            queue = deque([start_vid])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        queue.append(v)
        
        return max_dist
    
    def _get_pattern_statistics(self, patterns: List[Tuple[Pattern, int]]) -> Dict:
        """Compute statistics about current pattern set"""
        if not patterns:
            return {
                'count': 0,
                'avg_size': 0,
                'avg_edges': 0,
                'avg_diameter': 0,
                'size_range': (0, 0),
                'support_range': (0, 0)
            }
        
        sizes = [len(p.vertices) for p, _ in patterns]
        edges = [len(p.edges) for p, _ in patterns]
        diameters = [self._calculate_diameter(p) for p, _ in patterns]
        supports = [s for _, s in patterns]
        
        return {
            'count': len(patterns),
            'avg_size': sum(sizes) / len(sizes),
            'avg_edges': sum(edges) / len(edges),
            'avg_diameter': sum(diameters) / len(diameters),
            'size_range': (min(sizes), max(sizes)),
            'edge_range': (min(edges), max(edges)),
            'support_range': (min(supports), max(supports))
        }
    
    def _simulate_user_action(self, round_num: int, num_patterns: int, stats: Dict) -> str:
        """
        Simulate intelligent user decision-making
        In real implementation, this would be user input
        """
        
        # Round 1: Always start unconstrained
        if round_num == 1:
            if num_patterns > 50:
                return 'add_max_size'
            else:
                return 'add_connected'
        
        # Round 2: Refine based on pattern count
        elif round_num == 2:
            if num_patterns > 30:
                return 'add_min_size'
            elif num_patterns < 5:
                return 'remove_constraint'
            else:
                return 'add_diameter'
        
        # Round 3: Look at pattern characteristics
        elif round_num == 3:
            if stats['avg_size'] > 8:
                return 'add_max_size'
            elif stats['avg_diameter'] > 6:
                return 'add_diameter'
            else:
                return 'view_more'
        
        # Round 4: View more or finish
        elif round_num == 4:
            if 10 <= num_patterns <= 30:
                return 'finish'  # Good number of patterns
            else:
                return 'view_more'
        
        # Round 5+: Finish
        else:
            return 'finish'
    
    def _add_constraint_interactive(self, constraint_type: str):
        """Add constraint based on user choice"""
        print(f"\n  💡 Adding {constraint_type} constraint...")
        
        if constraint_type == 'add_max_size':
            max_size = 6  # Reasonable for MUTAG molecules
            constraint = MaxSizeConstraint(max_size)
            self.constraints.append(constraint)
            self.stats['constraints_added'] += 1
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_min_size':
            min_size = 3
            constraint = MinSizeConstraint(min_size)
            self.constraints.append(constraint)
            self.stats['constraints_added'] += 1
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_connected':
            constraint = ConnectedConstraint()
            self.constraints.append(constraint)
            self.stats['constraints_added'] += 1
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_diameter':
            diameter = 5
            constraint = DiameterConstraint(diameter)
            self.constraints.append(constraint)
            self.stats['constraints_added'] += 1
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'remove_constraint':
            if self.constraints:
                removed = self.constraints.pop()
                self.stats['constraints_removed'] += 1
                print(f"  ✓ Removed: {removed.name}")
            else:
                print(f"  ⚠️  No constraints to remove")
        
        else:
            print(f"  ⚠️  Unknown constraint type: {constraint_type}")
    
    def interactive_session(self, max_rounds: int = 5) -> List[Tuple[Pattern, int]]:
        """
        Run interactive mining session with proper GSPAN
        """
        
        if self.verbose:
            print(f"{'─'*70}")
            print("Starting Interactive Session")
            print("User iteratively refines constraints based on mined patterns")
            print(f"{'─'*70}")
        
        round_num = 0
        final_patterns = []
        
        while round_num < max_rounds:
            round_num += 1
            
            if self.verbose:
                print(f"\n{'═'*70}")
                print(f"Round {round_num}/{max_rounds}")
                print(f"{'═'*70}")
            
            # Show current constraints
            if self.constraints:
                print(f"\n  Current Constraints ({len(self.constraints)}):")
                for i, c in enumerate(self.constraints, 1):
                    print(f"    {i}. {c.name}")
            else:
                print(f"\n  No constraints (unconstrained mining)")
            
            # Mine with current constraints using PROPER GSPAN
            start_time = time.time()
            patterns = self._mine_with_current_constraints()
            mine_time = time.time() - start_time
            
            self.stats['total_mining_time'] += mine_time
            
            if self.verbose:
                print(f"  ✓ Found {len(patterns)} patterns in {mine_time:.2f}s")
            
            # Get statistics
            stats = self._get_pattern_statistics(patterns)
            
            if stats['count'] > 0:
                print(f"\n  📊 Pattern Statistics:")
                print(f"      Count: {stats['count']}")
                print(f"      Size: {stats['avg_size']:.1f} avg (range: {stats['size_range'][0]}-{stats['size_range'][1]})")
                print(f"      Edges: {stats['avg_edges']:.1f} avg (range: {stats['edge_range'][0]}-{stats['edge_range'][1]})")
                print(f"      Diameter: {stats['avg_diameter']:.1f} avg")
                print(f"      Support: {stats['support_range'][0]}-{stats['support_range'][1]}")
            
            # Display sample patterns
            self._display_patterns(patterns, limit=10)
            
            # Save to history
            self.session_history.append({
                'round': round_num,
                'constraints': [c.name for c in self.constraints],
                'num_patterns': len(patterns),
                'time': mine_time,
                'stats': stats
            })
            
            # Simulate user decision
            print(f"\n  {'─'*66}")
            print(f"  What would you like to do?")
            print(f"    1. Add constraint (refine/narrow results)")
            print(f"    2. Remove constraint (broaden results)")
            print(f"    3. View more patterns")
            print(f"    4. Finish and accept results")
            
            action = self._simulate_user_action(round_num, len(patterns), stats)
            print(f"\n  → User decision: {action}")
            
            if action == 'finish':
                final_patterns = patterns
                if self.verbose:
                    print(f"\n  ✓ Session complete. Accepting current results.")
                break
            
            elif action == 'view_more':
                if self.verbose:
                    print(f"\n  Showing more patterns...")
                self._display_patterns(patterns, limit=20)
            
            elif action.startswith('add_') or action.startswith('remove_'):
                self._add_constraint_interactive(action)
            
            final_patterns = patterns
        
        self.stats['total_rounds'] = round_num
        
        if self.verbose:
            self._print_session_summary(final_patterns)
        
        return final_patterns
    
    def _print_session_summary(self, patterns: List[Tuple[Pattern, int]]):
        """Print comprehensive summary of interactive session"""
        
        print(f"\n{'='*70}")
        print(f"{'SESSION SUMMARY':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Final Results:")
        print(f"  Patterns found: {len(patterns)}")
        print(f"  Rounds: {self.stats['total_rounds']}")
        print(f"  Final constraints: {len(self.constraints)}")
        
        print(f"\n⏱️  Time:")
        print(f"  Total mining time: {self.stats['total_mining_time']:.2f}s")
        print(f"  Average per round: {self.stats['total_mining_time']/self.stats['total_rounds']:.2f}s")
        
        print(f"\n🔧 Constraint Evolution:")
        print(f"  Constraints added: {self.stats['constraints_added']}")
        print(f"  Constraints removed: {self.stats['constraints_removed']}")
        
        print(f"\n📝 Session History:")
        print(f"  {'Round':<7} {'Patterns':<10} {'Constraints':<15} {'Time':<8}")
        print(f"  {'-'*60}")
        
        for record in self.session_history:
            constraints_str = str(len(record['constraints']))
            print(f"  {record['round']:<7} {record['num_patterns']:<10} "
                  f"{constraints_str:<15} {record['time']:<8.2f}s")
        
        print(f"\n🔧 Final Constraint Set:")
        if self.constraints:
            for i, c in enumerate(self.constraints, 1):
                print(f"  {i}. {c.name}")
        else:
            print(f"  (no constraints)")
        
        # Pattern progression
        if len(self.session_history) > 1:
            print(f"\n📈 Pattern Count Progression:")
            for record in self.session_history:
                count = record['num_patterns']
                bar = '█' * min(count // 2, 40)
                print(f"  Round {record['round']}: {bar} ({count})")
            
            initial = self.session_history[0]['num_patterns']
            final = len(patterns)
            change = final - initial
            direction = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"\n  Pattern refinement: {initial} {direction} {final} ({abs(change)} change)")
        
        # Top patterns
        if patterns:
            print(f"\n🏆 Top 10 Patterns by Support:")
            sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
            for i, (pattern, support) in enumerate(sorted_patterns[:10], 1):
                labels = [v.label for v in pattern.vertices]
                diameter = self._calculate_diameter(pattern)
                print(f"  {i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                      f"Edges={len(pattern.edges)}, Diam={diameter}, Support={support}")
        
        print(f"\n{'='*70}\n")

def main():
    """Run interactive mining with proper GSPAN on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - INTERACTIVE MINING + PROPER GSPAN':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=50)  # Use 50 for reasonable speed
    
    print(f"Configuration:")
    print(f"  Dataset: {len(graphs)} graphs")
    print(f"  Min support: 0.25 (25%)")
    print(f"  Max pattern size: 5")
    print(f"  Max rounds: 5")
    
    # Run interactive session
    miner = InteractiveConstraintProperGSPAN(
        database=graphs,
        min_support=0.25,
        max_pattern_size=5,
        verbose=True
    )
    
    patterns = miner.interactive_session(max_rounds=5)
    
    # Visualize
    try:
        print(f"\nGenerating visualizations...")
        visualizer = PatternVisualizer()
        fig = visualizer.plot_pattern_distribution(
            patterns,
            title="MUTAG Interactive Mining Results"
        )
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    try:
        visualizer.save_all_plots('results')
    except:
        pass
    
    with open('results/interactive_proper_session.txt', 'w') as f:
        f.write("MUTAG - Interactive Mining + Proper GSPAN Session\n")
        f.write("="*70 + "\n\n")
        f.write(f"Final patterns: {len(patterns)}\n")
        f.write(f"Rounds: {miner.stats['total_rounds']}\n")
        f.write(f"Total mining time: {miner.stats['total_mining_time']:.2f}s\n")
        f.write(f"Constraints added: {miner.stats['constraints_added']}\n")
        f.write(f"Constraints removed: {miner.stats['constraints_removed']}\n\n")
        
        f.write("Session History:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Round':<7} {'Patterns':<10} {'Constraints':<40} {'Time':<8}\n")
        f.write("-"*70 + "\n")
        
        for record in miner.session_history:
            constraints_str = ', '.join(record['constraints']) if record['constraints'] else 'none'
            constraints_str = constraints_str[:38] + '..' if len(constraints_str) > 40 else constraints_str
            f.write(f"{record['round']:<7} {record['num_patterns']:<10} "
                   f"{constraints_str:<40} {record['time']:<8.2f}s\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Top 20 Final Patterns:\n")
        f.write("-"*70 + "\n")
        
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns[:20], 1):
            labels = [v.label for v in pattern.vertices]
            diameter = miner._calculate_diameter(pattern)
            f.write(f"{i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                   f"Edges={len(pattern.edges)}, Diameter={diameter}, Support={support}\n")
    
    print(f"\n✓ Results saved to 'results/interactive_proper_session.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()