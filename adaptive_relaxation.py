"""
NOVEL CONTRIBUTION 1: Adaptive Constraint Relaxation with Proper GSPAN
Uses Q-learning to automatically relax constraints when they are too restrictive

Integrates with real GSPAN algorithm for correct pattern mining
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import *
from utils.gspan_algorithm import ProperGSPAN  # ← Use proper GSPAN!
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
            # Increase max size by ~15-20%
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

class AdaptiveConstraintProperGSPAN:
    """
    Proper GSPAN with adaptive Q-learning based constraint relaxation
    
    APPROACH:
    1. Use proper GSPAN to mine with current constraints
    2. If insufficient results, use Q-learning to select constraint to relax
    3. Repeat until target number of patterns found or max iterations reached
    """
    
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 constraints: List[Constraint],
                 min_results: int = 10,
                 max_iterations: int = 10,
                 max_pattern_size: int = 6,
                 verbose: bool = True):
        
        self.database = database
        self.min_support = min_support
        self.min_results = min_results
        self.max_iterations = max_iterations
        self.max_pattern_size = max_pattern_size
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
        
        # History
        self.relaxation_history = []
        
        # Statistics
        self.stats = {
            'iterations': 0,
            'total_runtime': 0,
            'mining_runtime': 0,
            'relaxations_performed': 0,
            'successful_relaxations': 0,
            'patterns_per_iteration': []
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization info"""
        print(f"\n{'='*70}")
        print(f"{'ADAPTIVE CONSTRAINT RELAXATION + PROPER GSPAN':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {int(self.min_support * len(self.database))}")
        print(f"  Target patterns: ≥{self.min_results}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Max pattern size: {self.max_pattern_size}")
        
        print(f"\n  Initial Constraints:")
        for i, rc in enumerate(self.relaxable_constraints, 1):
            print(f"    {i}. {rc.current_constraint.name}")
        
        print(f"\n  Q-Learning Parameters:")
        print(f"    Learning rate (α): {self.alpha}")
        print(f"    Discount factor (γ): {self.gamma}")
        print(f"    Exploration rate (ε): {self.epsilon}")
        
        print(f"{'='*70}\n")
    
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
        """Select which constraint to relax using epsilon-greedy Q-learning"""
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
        
        # Get max Q-value for next state
        next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update rule
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state][action] = new_q
    
    def _mine_with_current_constraints(self) -> List[Tuple[Pattern, int]]:
        """
        Use proper GSPAN to mine with current constraints
        This is the KEY difference from the original version
        """
        
        constraints = self._get_current_constraints()
        
        # Use PROPER GSPAN algorithm
        gspan = ProperGSPAN(
            database=self.database,
            min_support=self.min_support,
            constraints=constraints,
            max_pattern_size=self.max_pattern_size,
            verbose=False  # Suppress GSPAN output during iterations
        )
        
        patterns = gspan.mine()
        
        return patterns
    
    def mine_with_adaptation(self) -> List[Tuple[Pattern, int]]:
        """Main adaptive mining loop with Q-learning"""
        
        if self.verbose:
            print(f"{'='*70}")
            print(f"{'STARTING ADAPTIVE MINING WITH Q-LEARNING':^70}")
            print(f"{'='*70}\n")
        
        start_time = time.time()
        
        iteration = 0
        best_patterns = []
        previous_num_results = 0
        
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
            
            # Mine with current constraints using PROPER GSPAN
            if self.verbose:
                print(f"\nMining with Proper GSPAN...")
            
            iter_start = time.time()
            patterns = self._mine_with_current_constraints()
            iter_time = time.time() - iter_start
            
            num_results = len(patterns)
            self.stats['patterns_per_iteration'].append(num_results)
            self.stats['mining_runtime'] += iter_time
            
            if self.verbose:
                print(f"  ✓ Found {num_results} patterns in {iter_time:.2f}s")
            
            # Compute state for Q-learning
            state = self._compute_state(num_results)
            
            if self.verbose:
                print(f"  State: {state}")
            
            # Check termination condition
            if num_results >= self.min_results:
                if self.verbose:
                    print(f"\n🎉 Success! Found {num_results} patterns (target: ≥{self.min_results})")
                best_patterns = patterns
                break
            
            # Update best patterns if improved
            if num_results > len(best_patterns):
                best_patterns = patterns
                if self.verbose:
                    print(f"  📈 New best: {num_results} patterns")
            
            # Select constraint to relax using Q-learning
            action = self._select_constraint_to_relax(state)
            
            if action == -1:
                if self.verbose:
                    print(f"\n⚠️  No more constraints can be relaxed")
                break
            
            # Relax selected constraint
            old_name = self.relaxable_constraints[action].current_constraint.name
            relax_msg = self.relaxable_constraints[action].relax()
            new_name = self.relaxable_constraints[action].current_constraint.name
            
            if self.verbose:
                print(f"\n🔧 Relaxing constraint {action + 1}:")
                print(f"  {old_name}")
                print(f"  → {relax_msg}")
            
            self.stats['relaxations_performed'] += 1
            
            # Compute reward for Q-learning
            improvement = num_results - previous_num_results
            
            if num_results > previous_num_results:
                # Reward proportional to improvement
                reward = 10.0 + improvement
                self.stats['successful_relaxations'] += 1
            elif num_results == previous_num_results:
                reward = -2.0  # Small penalty for no improvement
            else:
                reward = -5.0  # Penalty for getting worse
            
            # Update Q-table
            next_state = self._compute_state(num_results)
            self._update_q_value(state, action, reward, next_state)
            
            if self.verbose:
                print(f"  Reward: {reward:.1f}")
                print(f"  Q-value updated: {self.q_table[state][action]:.2f}")
            
            # Record history
            self.relaxation_history.append({
                'iteration': iteration,
                'constraint_relaxed': action,
                'old_constraint': old_name,
                'new_constraint': new_name,
                'num_results': num_results,
                'reward': reward,
                'time': iter_time,
                'q_value': self.q_table[state][action]
            })
            
            previous_num_results = num_results
        
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
        print(f"  Iterations: {self.stats['iterations']}")
        print(f"  Target achieved: {'✓ Yes' if len(patterns) >= self.min_results else '✗ No'}")
        
        print(f"\n⏱️  Runtime:")
        print(f"  Total runtime: {self.stats['total_runtime']:.2f}s")
        print(f"  Mining runtime: {self.stats['mining_runtime']:.2f}s")
        print(f"  Overhead: {(self.stats['total_runtime'] - self.stats['mining_runtime']):.2f}s")
        print(f"  Avg per iteration: {self.stats['total_runtime']/self.stats['iterations']:.2f}s")
        
        print(f"\n🤖 Q-Learning Statistics:")
        print(f"  Relaxations performed: {self.stats['relaxations_performed']}")
        print(f"  Successful relaxations: {self.stats['successful_relaxations']}")
        
        if self.stats['relaxations_performed'] > 0:
            success_rate = (self.stats['successful_relaxations'] / 
                          self.stats['relaxations_performed'] * 100)
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Pattern growth over iterations
        if self.stats['patterns_per_iteration']:
            print(f"\n📈 Pattern Growth:")
            for i, count in enumerate(self.stats['patterns_per_iteration'], 1):
                bar = '█' * min(count // 2, 40)
                print(f"  Iter {i}: {bar} ({count} patterns)")
        
        print(f"\n📝 Relaxation History:")
        for record in self.relaxation_history:
            print(f"  Iter {record['iteration']}: "
                  f"{record['old_constraint']} → {record['new_constraint']}")
            print(f"      Results: {record['num_results']}, "
                  f"Reward: {record['reward']:+.1f}, "
                  f"Q-value: {record['q_value']:.2f}")
        
        print(f"\n🧠 Learned Q-Table:")
        if self.q_table:
            for state in sorted(self.q_table.keys()):
                print(f"  State '{state}':")
                actions_sorted = sorted(self.q_table[state].items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
                for action, q_value in actions_sorted[:3]:  # Top 3 actions
                    if action < len(self.relaxable_constraints):
                        constraint_name = self.relaxable_constraints[action].original_constraint.name
                        print(f"    Action {action} ({constraint_name.split('(')[0]}): "
                              f"Q={q_value:.2f}")
        else:
            print("  (No Q-values learned - target achieved immediately)")
        
        print(f"\n🔧 Final Constraint Configuration:")
        for i, rc in enumerate(self.relaxable_constraints, 1):
            print(f"  {i}. {rc.current_constraint.name}")
            if rc.relaxation_level > 0:
                print(f"      ↳ Original: {rc.original_constraint.name}")
                print(f"      ↳ Relaxed {rc.relaxation_level} times")
        
        # Show top patterns
        if patterns:
            print(f"\n🏆 Top 10 Patterns by Support:")
            sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
            for i, (pattern, support) in enumerate(sorted_patterns[:10], 1):
                labels = [v.label for v in pattern.vertices]
                print(f"  {i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                      f"Edges={len(pattern.edges)}, Support={support}")
        
        print(f"\n{'='*70}\n")

def main():
    """Run adaptive constraint relaxation with proper GSPAN on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - ADAPTIVE RELAXATION + PROPER GSPAN':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=50)  # Use 50 for reasonable speed
    
    # Define RESTRICTIVE constraints (intentionally tight to demonstrate relaxation)
    constraints = [
        MaxSizeConstraint(4),      # Very restrictive - only small patterns
        MinSizeConstraint(3),      # Must have at least 3 vertices
        DiameterConstraint(2),     # Very compact structures only
        ConnectedConstraint()      # Must be connected (cannot be relaxed)
    ]
    
    print(f"Configuration:")
    print(f"  Dataset: {len(graphs)} graphs")
    print(f"  Min support: 0.25 (25%)")
    print(f"  Target: ≥12 patterns")
    print(f"  Max iterations: 10")
    print(f"  Max pattern size: 5")
    
    print(f"\nInitial (Restrictive) Constraints:")
    for c in constraints:
        print(f"  - {c.name}")
    
    print(f"\nNote: These constraints are intentionally restrictive")
    print(f"      to demonstrate the Q-learning relaxation mechanism.")
    
    # Run adaptive mining
    miner = AdaptiveConstraintProperGSPAN(
        database=graphs,
        min_support=0.25,
        constraints=constraints,
        min_results=12,
        max_iterations=10,
        max_pattern_size=5,
        verbose=True
    )
    
    patterns = miner.mine_with_adaptation()
    
    # Visualize if possible
    try:
        visualizer = PatternVisualizer()
        fig = visualizer.plot_pattern_distribution(
            patterns,
            title="MUTAG Adaptive Relaxation Results"
        )
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/adaptive_proper_results.txt', 'w') as f:
        f.write("MUTAG - Adaptive Constraint Relaxation + Proper GSPAN\n")
        f.write("="*70 + "\n\n")
        f.write(f"Final patterns: {len(patterns)}\n")
        f.write(f"Iterations: {miner.stats['iterations']}\n")
        f.write(f"Total runtime: {miner.stats['total_runtime']:.2f}s\n")
        f.write(f"Mining runtime: {miner.stats['mining_runtime']:.2f}s\n")
        f.write(f"Relaxations: {miner.stats['relaxations_performed']}\n")
        f.write(f"Successful relaxations: {miner.stats['successful_relaxations']}\n\n")
        
        f.write("Relaxation History:\n")
        f.write("-"*70 + "\n")
        for record in miner.relaxation_history:
            f.write(f"Iter {record['iteration']}: "
                   f"{record['old_constraint']} → {record['new_constraint']}\n")
            f.write(f"  Results: {record['num_results']}, "
                   f"Reward: {record['reward']:+.1f}, "
                   f"Q-value: {record['q_value']:.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Top 20 Patterns:\n")
        f.write("-"*70 + "\n")
        
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns[:20], 1):
            labels = [v.label for v in pattern.vertices]
            f.write(f"{i:2d}. Labels={labels}, Size={len(pattern.vertices)}, "
                   f"Edges={len(pattern.edges)}, Support={support}\n")
    
    print(f"✓ Results saved to 'results/adaptive_proper_results.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()