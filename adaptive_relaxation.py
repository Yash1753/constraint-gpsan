"""
NOVEL CONTRIBUTION 1: Adaptive Constraint Relaxation with Learning
Optimized for MUTAG dataset

Uses Q-learning to automatically relax constraints when they are too restrictive
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import *
from utils.visualization import PatternVisualizer

class RelaxableConstraint:
    """Wrapper for constraints that can be relaxed"""
    
    def __init__(self, constraint: Constraint, relaxation_step: float = 0.15):
        self.original_constraint = constraint
        self.current_constraint = constraint
        self.relaxation_step = relaxation_step
        self.relaxation_level = 0
        self.effectiveness_score = 1.0
    
    def relax(self):
        """Relax constraint by one step"""
        self.relaxation_level += 1
        
        if isinstance(self.current_constraint, MaxSizeConstraint):
            # Increase max size by 20%
            new_max = int(self.current_constraint.max_size * (1 + self.relaxation_step))
            self.current_constraint = MaxSizeConstraint(new_max)
            return f"Increased max size to {new_max}"
        
        elif isinstance(self.current_constraint, MinSizeConstraint):
            # Decrease min size
            new_min = max(1, int(self.current_constraint.min_size * (1 - self.relaxation_step)))
            self.current_constraint = MinSizeConstraint(new_min)
            return f"Decreased min size to {new_min}"
        
        elif isinstance(self.current_constraint, DiameterConstraint):
            # Increase max diameter
            new_diameter = self.current_constraint.max_diameter + 1
            self.current_constraint = DiameterConstraint(new_diameter)
            return f"Increased max diameter to {new_diameter}"
        
        elif isinstance(self.current_constraint, LabelCountConstraint):
            # Increase max count
            new_max = int(self.current_constraint.max_count * 1.5) if self.current_constraint.max_count < float('inf') else float('inf')
            self.current_constraint = LabelCountConstraint(
                self.current_constraint.label,
                self.current_constraint.min_count,
                new_max
            )
            return f"Increased max count to {new_max}"
        
        return "Relaxed (no change)"
    
    def can_relax(self) -> bool:
        """Check if constraint can be further relaxed"""
        return self.relaxation_level < 5  # Max 5 relaxation steps
    
    def reset(self):
        """Reset to original constraint"""
        self.current_constraint = self.original_constraint
        self.relaxation_level = 0

class AdaptiveConstraintGSPAN:
    """
    GSPAN with adaptive constraint relaxation
    Uses Q-learning to decide which constraints to relax
    """
    
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 constraints: List[Constraint],
                 min_results: int = 10,
                 max_iterations: int = 10,
                 verbose: bool = True):
        
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.min_results = min_results
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Wrap constraints for relaxation
        self.relaxable_constraints = [
            RelaxableConstraint(c) for c in constraints
        ]
        
        # Q-learning parameters
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        
        # Label statistics for mining
        self.label_stats = self._compute_label_stats()
        
        # History
        self.relaxation_history = []
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'total_runtime': 0,
            'relaxations_performed': 0,
            'successful_relaxations': 0
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization info"""
        print(f"\n{'='*70}")
        print(f"{'ADAPTIVE CONSTRAINT RELAXATION':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {self.min_support_count}")
        print(f"  Target patterns: ≥{self.min_results}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"\n  Initial Constraints:")
        for i, rc in enumerate(self.relaxable_constraints, 1):
            print(f"    {i}. {rc.current_constraint.name}")
        print(f"{'='*70}\n")
    
    def _compute_label_stats(self) -> Dict:
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
    
    def _get_current_constraints(self) -> List[Constraint]:
        """Get current (possibly relaxed) constraints"""
        return [rc.current_constraint for rc in self.relaxable_constraints]
    
    def _compute_state(self, num_results: int) -> str:
        """Compute state for Q-learning"""
        if num_results == 0:
            return "NO_RESULTS"
        elif num_results < self.min_results // 2:
            return "VERY_FEW_RESULTS"
        elif num_results < self.min_results:
            return "FEW_RESULTS"
        else:
            return "ENOUGH_RESULTS"
    
    def _select_constraint_to_relax(self, state: str) -> int:
        """Select which constraint to relax using epsilon-greedy"""
        relaxable = [i for i, rc in enumerate(self.relaxable_constraints) 
                    if rc.can_relax()]
        
        if not relaxable:
            return -1
        
        if np.random.random() < self.epsilon:
            # Explore: random choice
            return np.random.choice(relaxable)
        else:
            # Exploit: best Q-value
            q_values = [self.q_table[state][i] for i in relaxable]
            best_idx = relaxable[np.argmax(q_values)]
            return best_idx
    
    def _update_q_value(self, state: str, action: int, reward: float, 
                       next_state: str):
        """Update Q-table using Q-learning"""
        old_q = self.q_table[state][action]
        
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state][action] = new_q
    
    def _mine_with_constraints(self) -> List[Tuple[Pattern, int]]:
        """Mine with current constraint set (simplified)"""
        patterns = []
        constraints = self._get_current_constraints()
        constraint_manager = ConstraintManager(constraints)
        
        # Generate single-vertex patterns
        for label, count in self.label_stats['vertex_counts'].items():
            if count >= self.min_support_count:
                p = Pattern()
                p.add_vertex(label)
                p.support = count
                
                if constraint_manager.can_satisfy_all(p):
                    if constraint_manager.check_antimonotone(p):
                        if constraint_manager.check_monotone(p):
                            patterns.append((p, count))
        
        return patterns
    
    def mine_with_adaptation(self) -> List[Tuple[Pattern, int]]:
        """Main adaptive mining loop"""
        if self.verbose:
            print(f"{'='*70}")
            print(f"{'STARTING ADAPTIVE MINING':^70}")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        
        iteration = 0
        best_patterns = []
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if self.verbose:
                print(f"{'─'*70}")
                print(f"Iteration {iteration}/{self.max_iterations}")
                print(f"{'─'*70}")
                print(f"\nCurrent Constraints:")
                for i, rc in enumerate(self.relaxable_constraints, 1):
                    relax_info = f" [relaxed {rc.relaxation_level}x]" if rc.relaxation_level > 0 else ""
                    print(f"  {i}. {rc.current_constraint.name}{relax_info}")
            
            # Mine with current constraints
            if self.verbose:
                print(f"\nMining...")
            
            iter_start = time.time()
            patterns = self._mine_with_constraints()
            iter_time = time.time() - iter_start
            
            num_results = len(patterns)
            
            if self.verbose:
                print(f"  Found {num_results} patterns in {iter_time:.2f}s")
            
            # Compute state
            state = self._compute_state(num_results)
            
            if self.verbose:
                print(f"  State: {state}")
            
            # Check termination
            if num_results >= self.min_results:
                if self.verbose:
                    print(f"\n✓ Success! Found {num_results} patterns (target: {self.min_results})")
                best_patterns = patterns
                break
            
            if num_results > len(best_patterns):
                best_patterns = patterns
                if self.verbose:
                    print(f"  New best: {num_results} patterns")
            
            # Select constraint to relax
            action = self._select_constraint_to_relax(state)
            
            if action == -1:
                if self.verbose:
                    print(f"\n✗ No more constraints can be relaxed")
                break
            
            # Relax selected constraint
            old_name = self.relaxable_constraints[action].current_constraint.name
            relax_msg = self.relaxable_constraints[action].relax()
            new_name = self.relaxable_constraints[action].current_constraint.name
            
            if self.verbose:
                print(f"\nRelaxing constraint {action + 1}:")
                print(f"  {old_name} → {relax_msg}")
            
            self.stats['relaxations_performed'] += 1
            
            # Compute reward
            if num_results > len(best_patterns):
                reward = 10.0
                self.stats['successful_relaxations'] += 1
            elif num_results == len(best_patterns):
                reward = 0.0
            else:
                reward = -5.0
            
            # Update Q-table
            next_state = self._compute_state(num_results)
            self._update_q_value(state, action, reward, next_state)
            
            # Record history
            self.relaxation_history.append({
                'iteration': iteration,
                'constraint_relaxed': action,
                'old_constraint': old_name,
                'new_constraint': new_name,
                'num_results': num_results,
                'reward': reward,
                'time': iter_time
            })
            
            if self.verbose:
                print(f"  Reward: {reward:.1f}\n")
        
        self.stats['iterations'] = iteration
        self.stats['total_runtime'] = time.time() - start_time
        
        if self.verbose:
            self._print_results(best_patterns)
        
        return best_patterns
    
    def _print_results(self, patterns: List[Tuple[Pattern, int]]):
        """Print results and learning statistics"""
        print(f"\n{'='*70}")
        print(f"{'ADAPTIVE MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results:")
        print(f"  Final patterns found: {len(patterns)}")
        print(f"  Total runtime: {self.stats['total_runtime']:.2f}s")
        print(f"  Iterations: {self.stats['iterations']}")
        
        print(f"\n🤖 Adaptation Statistics:")
        print(f"  Relaxations performed: {self.stats['relaxations_performed']}")
        print(f"  Successful relaxations: {self.stats['successful_relaxations']}")
        
        if self.stats['relaxations_performed'] > 0:
            success_rate = (self.stats['successful_relaxations'] / 
                          self.stats['relaxations_performed'] * 100)
            print(f"  Success rate: {success_rate:.1f}%")
        
        print(f"\n📝 Relaxation History:")
        for record in self.relaxation_history:
            print(f"  Iter {record['iteration']}: {record['old_constraint']} → "
                  f"{record['new_constraint']}, results={record['num_results']}, "
                  f"reward={record['reward']:.1f}")
        
        print(f"\n🧠 Learned Q-Values:")
        for state, actions in sorted(self.q_table.items()):
            print(f"  State '{state}':")
            for action, q_value in sorted(actions.items()):
                constraint_name = self.relaxable_constraints[action].current_constraint.name
                print(f"    Relax constraint {action + 1} ({constraint_name.split('(')[0]}): {q_value:.2f}")
        
        print(f"\n🔧 Final Constraint Configuration:")
        for i, rc in enumerate(self.relaxable_constraints, 1):
            print(f"  {i}. {rc.current_constraint.name}")
            if rc.relaxation_level > 0:
                print(f"      (Original: {rc.original_constraint.name}, "
                      f"relaxed {rc.relaxation_level} times)")
        
        print(f"\n{'='*70}\n")

def main():
    """Run adaptive constraint relaxation on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - ADAPTIVE CONSTRAINT RELAXATION':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=100)  # Use subset for speed
    
    # Define RESTRICTIVE constraints (will need relaxation)
    constraints = [
        MaxSizeConstraint(6),      # Very restrictive for MUTAG
        MinSizeConstraint(4),      # Moderate
        DiameterConstraint(3),     # Restrictive
        ConnectedConstraint()      # Cannot be relaxed
    ]
    
    print(f"Initial (Restrictive) Constraints:")
    for c in constraints:
        print(f"  - {c.name}")
    
    # Run adaptive mining
    miner = AdaptiveConstraintGSPAN(
        database=graphs,
        min_support=0.15,
        constraints=constraints,
        min_results=15,
        max_iterations=10,
        verbose=True
    )
    
    patterns = miner.mine_with_adaptation()
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/adaptive_results.txt', 'w') as f:
        f.write("MUTAG - Adaptive Constraint Relaxation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Final patterns: {len(patterns)}\n")
        f.write(f"Iterations: {miner.stats['iterations']}\n")
        f.write(f"Runtime: {miner.stats['total_runtime']:.2f}s\n")
        f.write(f"Relaxations: {miner.stats['relaxations_performed']}\n\n")
        
        f.write("Relaxation History:\n")
        f.write("-"*70 + "\n")
        for record in miner.relaxation_history:
            f.write(f"Iter {record['iteration']}: {record['old_constraint']} → "
                   f"{record['new_constraint']}, results={record['num_results']}\n")
    
    print(f"\n✓ Results saved to 'results/adaptive_results.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
