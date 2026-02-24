"""
NOVEL CONTRIBUTION 2: Interactive Constraint Refinement
Optimized for MUTAG dataset

User iteratively explores patterns and refines constraints based on results
"""

import time
from typing import List, Tuple
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import *
from utils.visualization import PatternVisualizer

class InteractiveConstraintGSPAN:
    """
    Interactive mining where user refines constraints based on feedback
    Simulated for automated testing
    """
    
    def __init__(self, database: List[Graph], min_support: float, verbose: bool = True):
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.constraints = []
        self.session_history = []
        self.verbose = verbose
        
        # Label statistics
        self.label_stats = self._compute_label_stats()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"{'INTERACTIVE CONSTRAINT MINING':^70}")
            print(f"{'='*70}")
            print(f"  Database: {len(database)} graphs")
            print(f"  Min support: {self.min_support_count}")
            print(f"{'='*70}\n")
    
    def _compute_label_stats(self) -> dict:
        """Compute label statistics"""
        stats = {
            'vertex_counts': defaultdict(int),
            'vertex_labels': defaultdict(set)
        }
        
        for graph in self.database:
            for v in graph.vertices:
                if v.label not in stats['vertex_labels'][v.label]:
                    stats['vertex_counts'][v.label] += 1
                stats['vertex_labels'][v.label].add(graph.gid)
        
        return stats
    
    def _mine_current_constraints(self) -> List[Tuple[Pattern, int]]:
        """Mine with current constraint set"""
        patterns = []
        
        if not self.constraints:
            constraint_manager = None
        else:
            constraint_manager = ConstraintManager(self.constraints)
        
        # Generate single-vertex patterns
        for label, count in self.label_stats['vertex_counts'].items():
            if count >= self.min_support_count:
                p = Pattern()
                p.add_vertex(label)
                p.support = count
                
                if constraint_manager is None or constraint_manager.can_satisfy_all(p):
                    if constraint_manager is None or constraint_manager.check_antimonotone(p):
                        if constraint_manager is None or constraint_manager.check_monotone(p):
                            patterns.append((p, count))
        
        return patterns
    
    def _display_patterns(self, patterns: List[Tuple[Pattern, int]], limit: int = 10):
        """Display sample patterns to user"""
        print(f"\n  Showing {min(limit, len(patterns))} sample patterns:")
        print(f"  {'─'*66}")
        
        for i, (pattern, support) in enumerate(patterns[:limit], 1):
            labels = [v.label for v in pattern.vertices]
            print(f"  {i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                  f"Edges={len(pattern.edges)}, Support={support}")
        
        if len(patterns) > limit:
            print(f"  ... and {len(patterns) - limit} more patterns")
    
    def _simulate_user_action(self, round_num: int, num_patterns: int) -> str:
        """
        Simulate user decision-making
        In real implementation, this would be user input
        """
        # Simulated decision logic based on round and results
        if round_num == 1:
            if num_patterns > 100:
                return 'add_max_size'
            else:
                return 'add_min_size'
        
        elif round_num == 2:
            if num_patterns > 50:
                return 'add_min_size'
            else:
                return 'add_connected'
        
        elif round_num == 3:
            if num_patterns < 10:
                return 'remove_constraint'
            else:
                return 'add_diameter'
        
        elif round_num == 4:
            return 'view_more'
        
        else:
            return 'finish'
    
    def _add_constraint_interactive(self, constraint_type: str):
        """Add constraint based on user choice"""
        print(f"\n  Adding {constraint_type} constraint...")
        
        if constraint_type == 'add_max_size':
            max_size = 10  # Reasonable for MUTAG molecules
            constraint = MaxSizeConstraint(max_size)
            self.constraints.append(constraint)
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_min_size':
            min_size = 3
            constraint = MinSizeConstraint(min_size)
            self.constraints.append(constraint)
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_connected':
            constraint = ConnectedConstraint()
            self.constraints.append(constraint)
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'add_diameter':
            diameter = 5
            constraint = DiameterConstraint(diameter)
            self.constraints.append(constraint)
            print(f"  ✓ Added: {constraint.name}")
        
        elif constraint_type == 'remove_constraint':
            if self.constraints:
                removed = self.constraints.pop()
                print(f"  ✓ Removed: {removed.name}")
            else:
                print(f"  ✗ No constraints to remove")
        
        else:
            print(f"  ✗ Unknown constraint type: {constraint_type}")
    
    def interactive_session(self, max_rounds: int = 5) -> List[Tuple[Pattern, int]]:
        """Run interactive mining session"""
        print(f"\n{'─'*70}")
        print("Starting Interactive Session")
        print("User can iteratively refine constraints based on results")
        print(f"{'─'*70}")
        
        round_num = 0
        final_patterns = []
        
        while round_num < max_rounds:
            round_num += 1
            
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
            
            # Mine with current constraints
            print(f"\n  Mining...")
            start_time = time.time()
            patterns = self._mine_current_constraints()
            mine_time = time.time() - start_time
            
            print(f"  ✓ Found {len(patterns)} patterns in {mine_time:.2f}s")
            
            # Display patterns
            self._display_patterns(patterns, limit=10)
            
            # Save to history
            self.session_history.append({
                'round': round_num,
                'constraints': [c.name for c in self.constraints],
                'num_patterns': len(patterns),
                'time': mine_time
            })
            
            # Simulate user action
            print(f"\n  {'─'*66}")
            print(f"  What would you like to do?")
            print(f"    1. Add constraint (refine results)")
            print(f"    2. Remove constraint (broaden results)")
            print(f"    3. View more patterns")
            print(f"    4. Finish and accept results")
            
            action = self._simulate_user_action(round_num, len(patterns))
            print(f"\n  → Simulated action: {action}")
            
            if action == 'finish':
                final_patterns = patterns
                print(f"\n  ✓ Session complete. Accepting current results.")
                break
            
            elif action == 'view_more':
                self._display_patterns(patterns, limit=20)
            
            elif action.startswith('add_') or action.startswith('remove_'):
                self._add_constraint_interactive(action)
            
            final_patterns = patterns
        
        if self.verbose:
            self._print_session_summary(final_patterns)
        
        return final_patterns
    
    def _print_session_summary(self, patterns: List[Tuple[Pattern, int]]):
        """Print summary of interactive session"""
        print(f"\n{'='*70}")
        print(f"{'SESSION SUMMARY':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Final Results:")
        print(f"  Patterns found: {len(patterns)}")
        print(f"  Rounds: {len(self.session_history)}")
        print(f"  Final constraints: {len(self.constraints)}")
        
        print(f"\n📝 Session History:")
        for record in self.session_history:
            print(f"  Round {record['round']}: "
                  f"{record['num_patterns']:3d} patterns, "
                  f"{len(record['constraints']):2d} constraints, "
                  f"{record['time']:5.2f}s")
        
        print(f"\n🔧 Final Constraint Set:")
        if self.constraints:
            for i, c in enumerate(self.constraints, 1):
                print(f"  {i}. {c.name}")
        else:
            print(f"  (no constraints)")
        
        # Statistics
        total_time = sum(r['time'] for r in self.session_history)
        avg_patterns = sum(r['num_patterns'] for r in self.session_history) / len(self.session_history)
        
        print(f"\n📈 Statistics:")
        print(f"  Total mining time: {total_time:.2f}s")
        print(f"  Average patterns per round: {avg_patterns:.1f}")
        
        if len(self.session_history) > 1:
            initial = self.session_history[0]['num_patterns']
            final = len(patterns)
            print(f"  Pattern refinement: {initial} → {final} ({abs(final-initial)} change)")
        
        print(f"\n{'='*70}\n")

def main():
    """Run interactive mining on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - INTERACTIVE CONSTRAINT MINING':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=100)
    
    # Run interactive session
    miner = InteractiveConstraintGSPAN(
        database=graphs,
        min_support=0.15,
        verbose=True
    )
    
    patterns = miner.interactive_session(max_rounds=5)
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualizer = PatternVisualizer()
    fig = visualizer.plot_pattern_distribution(patterns, 
                                               title="MUTAG Interactive Mining Results")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    visualizer.save_all_plots('results')
    
    with open('results/interactive_session.txt', 'w') as f:
        f.write("MUTAG - Interactive Mining Session\n")
        f.write("="*70 + "\n\n")
        f.write(f"Final patterns: {len(patterns)}\n")
        f.write(f"Rounds: {len(miner.session_history)}\n\n")
        
        f.write("Session History:\n")
        f.write("-"*70 + "\n")
        for record in miner.session_history:
            f.write(f"Round {record['round']}: "
                   f"{record['num_patterns']} patterns, "
                   f"constraints={record['constraints']}\n")
    
    print(f"\n✓ Results saved to 'results/interactive_session.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
