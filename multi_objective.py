
import time
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import *
from utils.visualization import PatternVisualizer

@dataclass
class PatternScore:
    pattern: Pattern
    support: int
    size_score: float
    support_score: float
    complexity_score: float
    constraint_score: float
    pareto_rank: int = 0
    weighted_score: float = 0.0
    
    def dominates(self, other: 'PatternScore') -> bool:
        better_in_any = False
        
        objectives = [
            (self.size_score, other.size_score, True),
            (self.support_score, other.support_score, True),
            (self.complexity_score, other.complexity_score, False),  # Minimize
            (self.constraint_score, other.constraint_score, True),
        ]
        
        for self_val, other_val, maximize in objectives:
            if maximize:
                if self_val < other_val:
                    return False
                elif self_val > other_val:
                    better_in_any = True
            else:  # Minimize
                if self_val > other_val:
                    return False
                elif self_val < other_val:
                    better_in_any = True
        
        return better_in_any
    
    def __repr__(self):
        return (f"PatternScore(size={len(self.pattern.vertices)}, sup={self.support}, "
                f"scores=[{self.size_score:.2f}, {self.support_score:.2f}, "
                f"{self.complexity_score:.2f}, {self.constraint_score:.2f}])")

class MultiObjectiveGSPAN:
   
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 constraints: List[Constraint] = None,
                 objectives: Dict[str, float] = None,
                 verbose: bool = True):
        
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.constraints = constraints or []
        self.constraint_manager = ConstraintManager(self.constraints)
        self.verbose = verbose
     
        self.objective_weights = objectives or {
            'size': 0.25,
            'support': 0.35,
            'complexity': 0.20,
            'constraints': 0.20
        }
        
        total = sum(self.objective_weights.values())
        self.objective_weights = {k: v/total for k, v in self.objective_weights.items()}
        
        self.label_stats = self._compute_label_stats()
        
        self.stats = {
            'patterns_evaluated': 0,
            'pareto_optimal_found': 0,
            'runtime': 0
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        print(f"\n{'='*70}")
        print(f"{'MULTI-OBJECTIVE PATTERN MINING':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {self.min_support_count}")
        print(f"  Constraints: {len(self.constraints)}")
        
        print(f"\n  Objective Weights:")
        for obj, weight in self.objective_weights.items():
            print(f"    {obj:15s}: {weight:.3f}")
        
        if self.constraints:
            print(f"\n  Constraints:")
            for i, c in enumerate(self.constraints, 1):
                print(f"    {i}. {c.name}")
        
        print(f"{'='*70}\n")
    
    def _compute_label_stats(self) -> Dict:
        
        stats = {
            'vertex_counts': defaultdict(int),
            'vertex_labels': defaultdict(set)
        }
        
        for graph in self.database:
            for v in graph.vertices:
                if graph.gid not in stats['vertex_labels'][v.label]:
                    stats['vertex_counts'][v.label] += 1
                stats['vertex_labels'][v.label].add(graph.gid)
        
        return stats
    
    def _compute_complexity(self, pattern: Pattern) -> float:
        
        if len(pattern.vertices) == 0:
            return 0.0
        
        edge_vertex_ratio = len(pattern.edges) / len(pattern.vertices)
        complexity = min(edge_vertex_ratio / 3.0, 1.0)
        
        return complexity
    
    def _compute_constraint_satisfaction(self, pattern: Pattern) -> float:
        if not self.constraints:
            return 1.0
        
        satisfied = 0
        for constraint in self.constraints:
            if constraint.check(pattern):
                satisfied += 1
        
        return satisfied / len(self.constraints)
    
    def _evaluate_pattern(self, pattern: Pattern, support: int) -> PatternScore:
       
        self.stats['patterns_evaluated'] += 1
        
        max_size = 20
        max_support = len(self.database)
        
        size_score = min(len(pattern.vertices) / max_size, 1.0)
        support_score = support / max_support
        complexity_score = self._compute_complexity(pattern)
        constraint_score = self._compute_constraint_satisfaction(pattern)
        
        return PatternScore(
            pattern=pattern,
            support=support,
            size_score=size_score,
            support_score=support_score,
            complexity_score=complexity_score,
            constraint_score=constraint_score
        )
    
    def _find_pareto_frontier(self, scored_patterns: List[PatternScore]) -> List[PatternScore]:
        
        pareto_optimal = []
        
        for i, pattern1 in enumerate(scored_patterns):
            is_dominated = False
            
            for j, pattern2 in enumerate(scored_patterns):
                if i != j and pattern2.dominates(pattern1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pattern1.pareto_rank = 0
                pareto_optimal.append(pattern1)
            else:
                pattern1.pareto_rank = 1
        
        self.stats['pareto_optimal_found'] = len(pareto_optimal)
        
        return pareto_optimal
    
    def _compute_weighted_score(self, scored_pattern: PatternScore) -> float:
        score = (
            self.objective_weights['size'] * scored_pattern.size_score +
            self.objective_weights['support'] * scored_pattern.support_score +
            self.objective_weights['complexity'] * (1 - scored_pattern.complexity_score) +  # Minimize
            self.objective_weights['constraints'] * scored_pattern.constraint_score
        )
        return score
    
    def mine_pareto_optimal(self) -> Tuple[List[PatternScore], List[PatternScore]]:
        
        if self.verbose:
            print(f"{'─'*70}")
            print(f"Starting Multi-Objective Mining")
            print(f"{'─'*70}\n")
        
        start_time = time.time()
        if self.verbose:
            print("Generating candidate patterns...")
        
        all_patterns = []
 
        for label, count in self.label_stats['vertex_counts'].items():
            if count >= self.min_support_count:
                p = Pattern()
                p.add_vertex(label)
                p.support = count
                p.graph_ids = self.label_stats['vertex_labels'][label].copy()
                all_patterns.append((p, count))
        
        if self.verbose:
            print(f"  Generated {len(all_patterns)} patterns\n")
        
        if self.verbose:
            print("Evaluating patterns on multiple objectives...")
        
        scored_patterns = []
        for pattern, support in all_patterns:
            score = self._evaluate_pattern(pattern, support)
            scored_patterns.append(score)
        
        if self.verbose:
            print(f"  Evaluated {len(scored_patterns)} patterns\n")
      
        if self.verbose:
            print("Computing Pareto frontier...")
        
        pareto_optimal = self._find_pareto_frontier(scored_patterns)
        
        if self.verbose:
            print(f"  Found {len(pareto_optimal)} Pareto-optimal patterns\n")
        for pattern_score in pareto_optimal:
            pattern_score.weighted_score = self._compute_weighted_score(pattern_score)
        
        pareto_optimal.sort(key=lambda x: x.weighted_score, reverse=True)
        
        self.stats['runtime'] = time.time() - start_time
        
        if self.verbose:
            self._print_results(scored_patterns, pareto_optimal)
        
        return scored_patterns, pareto_optimal
    
    def _print_results(self, all_patterns: List[PatternScore], 
                      pareto_optimal: List[PatternScore]):
        print(f"{'='*70}")
        print(f"{'MULTI-OBJECTIVE MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results:")
        print(f"  Total patterns: {len(all_patterns)}")
        print(f"  Pareto-optimal patterns: {len(pareto_optimal)}")
        print(f"  Pareto ratio: {len(pareto_optimal)/len(all_patterns)*100:.1f}%")
        print(f"  Runtime: {self.stats['runtime']:.2f}s")
        
        print(f"\n🎯 Top 15 Pareto-Optimal Patterns:")
        print(f"  {'─'*66}")
        print(f"  {'Rank':<6} {'Size':<6} {'Supp':<6} {'Cmplx':<7} {'Const':<7} {'Weight':<8}")
        print(f"  {'─'*66}")
        
        for i, ps in enumerate(pareto_optimal[:15], 1):
            print(f"  {i:<6} {ps.size_score:<6.3f} {ps.support_score:<6.3f} "
                  f"{ps.complexity_score:<7.3f} {ps.constraint_score:<7.3f} "
                  f"{ps.weighted_score:<8.3f}")
        dominated = [p for p in all_patterns if p.pareto_rank > 0]
        if dominated:
            print(f"\n📉 Dominated Patterns ({len(dominated)}):")
            print(f"  Average size score: {np.mean([p.size_score for p in dominated]):.3f}")
            print(f"  Average support score: {np.mean([p.support_score for p in dominated]):.3f}")
            print(f"  Average complexity: {np.mean([p.complexity_score for p in dominated]):.3f}")
            print(f"  Average constraint satisfaction: {np.mean([p.constraint_score for p in dominated]):.3f}")
        
        print(f"\n📈 Pareto Front Visualization (Size vs Support):")
        print(f"  {'─'*66}")
        
        self._plot_pareto_front_ascii(all_patterns, pareto_optimal)
        
        print(f"\n{'='*70}\n")
    
    def _plot_pareto_front_ascii(self, all_patterns: List[PatternScore], 
                                 pareto_optimal: List[PatternScore]):
        # Create grid
        width, height = 50, 15
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot Pareto-optimal patterns
        for ps in pareto_optimal:
            x = int(ps.support_score * (width - 1))
            y = int((1 - ps.size_score) * (height - 1))
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '█'
        
        # Plot sample of dominated patterns
        for ps in all_patterns[:30]:
            if ps.pareto_rank > 0:
                x = int(ps.support_score * (width - 1))
                y = int((1 - ps.size_score) * (height - 1))
                if 0 <= x < width and 0 <= y < height and grid[y][x] == ' ':
                    grid[y][x] = '·'

        print("  Size")
        print("    ↑")
        for row in grid:
            print("    │" + ''.join(row))
        print("    └" + "─" * width + "→ Support")
        print("\n    Legend: █ = Pareto-optimal, · = Dominated")

def main():
    print(f"\n{'='*70}")
    print(f"{'MUTAG - MULTI-OBJECTIVE OPTIMIZATION':^70}")
    print(f"{'='*70}\n")

    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=100)
    
    constraints = MUTAGConstraints.basic_chemical()
    
    print(f"Constraints:")
    for c in constraints:
        print(f"  - {c.name}")
    
    objectives = {
        'size': 0.25,        
        'support': 0.35,     
        'complexity': 0.20,  
        'constraints': 0.20  
    }
    
    print(f"\nObjective Weights:")
    for obj, weight in objectives.items():
        print(f"  {obj}: {weight}")
    
    miner = MultiObjectiveGSPAN(
        database=graphs,
        min_support=0.15,
        constraints=constraints,
        objectives=objectives,
        verbose=True
    )
    
    all_patterns, pareto_optimal = miner.mine_pareto_optimal()
   
    print(f"\nGenerating visualizations...")
    visualizer = PatternVisualizer()
    
    pareto_tuples = [(ps.pattern, ps.support) for ps in pareto_optimal]
    
    fig = visualizer.plot_pattern_distribution(pareto_tuples,
                                               title="MUTAG Pareto-Optimal Patterns")
    
    import os
    os.makedirs('results', exist_ok=True)
    visualizer.save_all_plots('results')
    
    with open('results/multi_objective_results.txt', 'w') as f:
        f.write("MUTAG - Multi-Objective Optimization Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total patterns: {len(all_patterns)}\n")
        f.write(f"Pareto-optimal: {len(pareto_optimal)}\n")
        f.write(f"Runtime: {miner.stats['runtime']:.2f}s\n\n")
        
        f.write("Top 20 Pareto-Optimal Patterns:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Size':<6} {'Support':<8} {'Weighted Score':<15}\n")
        f.write("-"*70 + "\n")
        
        for i, ps in enumerate(pareto_optimal[:20], 1):
            f.write(f"{i:<6} {len(ps.pattern.vertices):<6} {ps.support:<8} "
                   f"{ps.weighted_score:<15.3f}\n")
    
    print(f"\n✓ Results saved to 'results/multi_objective_results.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
