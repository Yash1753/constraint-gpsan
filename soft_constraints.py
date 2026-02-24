"""
NOVEL CONTRIBUTION 4: Soft/Probabilistic Constraint Satisfaction
Optimized for MUTAG dataset

Constraints have satisfaction degrees (0-1) instead of binary pass/fail
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import Constraint, ConstraintManager, MUTAGConstraints
from utils.visualization import PatternVisualizer

@dataclass
class SoftConstraint:
    """Constraint with importance weight and fuzzy satisfaction"""
    constraint: Constraint
    weight: float = 1.0
    required: bool = False
    normalized_weight: float = 1.0
    
    def compute_satisfaction(self, pattern: Pattern) -> float:
        """
        Compute degree of satisfaction [0, 1]
        Returns partial satisfaction for some constraint types
        """
        if self.constraint.check(pattern):
            return 1.0
        else:
            return self._compute_partial_satisfaction(pattern)
    
    def _compute_partial_satisfaction(self, pattern: Pattern) -> float:
        """Compute partial satisfaction based on how close pattern is"""
        from utils.constraints import (MaxSizeConstraint, MinSizeConstraint,
                                       DiameterConstraint, LabelCountConstraint)
        
        # MaxSize: gradual penalty for exceeding
        if isinstance(self.constraint, MaxSizeConstraint):
            size = len(pattern.vertices)
            max_size = self.constraint.max_size
            if size <= max_size:
                return 1.0
            excess = size - max_size
            penalty = min(excess / max_size, 1.0)
            return max(1.0 - penalty, 0.0)
        
        # MinSize: gradual reward for approaching
        elif isinstance(self.constraint, MinSizeConstraint):
            size = len(pattern.vertices)
            min_size = self.constraint.min_size
            if size >= min_size:
                return 1.0
            return size / min_size if min_size > 0 else 0.0
        
        # Diameter: gradual penalty for exceeding
        elif isinstance(self.constraint, DiameterConstraint):
            diameter = self.constraint._compute_diameter(pattern)
            max_diameter = self.constraint.max_diameter
            if diameter <= max_diameter:
                return 1.0
            excess = diameter - max_diameter
            return max(1.0 - excess / (max_diameter + 1), 0.0)
        
        # LabelCount: gradual satisfaction
        elif isinstance(self.constraint, LabelCountConstraint):
            count = sum(1 for v in pattern.vertices if v.label == self.constraint.label)
            
            if self.constraint.min_count <= count <= self.constraint.max_count:
                return 1.0
            elif count < self.constraint.min_count:
                return count / self.constraint.min_count if self.constraint.min_count > 0 else 0.0
            else:  # count > max_count
                excess = count - self.constraint.max_count
                penalty = min(excess / self.constraint.max_count, 1.0) if self.constraint.max_count > 0 else 1.0
                return max(1.0 - penalty, 0.0)
        
        # Default: binary
        return 0.0

class SoftConstraintGSPAN:
    """
    GSPAN with soft/fuzzy constraint satisfaction
    Patterns ranked by weighted satisfaction score
    """
    
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 soft_constraints: List[SoftConstraint],
                 min_satisfaction: float = 0.6,
                 verbose: bool = True):
        
        self.database = database
        self.min_support_count = max(1, int(min_support * len(database)))
        self.soft_constraints = soft_constraints
        self.min_satisfaction = min_satisfaction
        self.verbose = verbose
        
        # Separate required vs optional
        self.required_constraints = [sc for sc in soft_constraints if sc.required]
        self.optional_constraints = [sc for sc in soft_constraints if not sc.required]
        
        # Normalize weights
        total_weight = sum(sc.weight for sc in soft_constraints)
        if total_weight > 0:
            for sc in soft_constraints:
                sc.normalized_weight = sc.weight / total_weight
        
        # Label statistics
        self.label_stats = self._compute_label_stats()
        
        # Statistics
        self.stats = {
            'patterns_evaluated': 0,
            'patterns_accepted': 0,
            'patterns_rejected_required': 0,
            'patterns_rejected_satisfaction': 0,
            'runtime': 0
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization info"""
        print(f"\n{'='*70}")
        print(f"{'SOFT CONSTRAINT SATISFACTION MINING':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {self.min_support_count}")
        print(f"  Min satisfaction: {self.min_satisfaction:.2f}")
        print(f"  Required constraints: {len(self.required_constraints)}")
        print(f"  Optional constraints: {len(self.optional_constraints)}")
        
        print(f"\n  Soft Constraints:")
        for i, sc in enumerate(self.soft_constraints, 1):
            req_str = " [REQUIRED]" if sc.required else ""
            print(f"    {i}. {sc.constraint.name}: weight={sc.weight:.2f}{req_str}")
        
        print(f"{'='*70}\n")
    
    def _compute_label_stats(self) -> Dict:
        """Compute label statistics"""
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
    
    def _check_required_constraints(self, pattern: Pattern) -> bool:
        """Check all required constraints (hard filter)"""
        for sc in self.required_constraints:
            if not sc.constraint.check(pattern):
                self.stats['patterns_rejected_required'] += 1
                return False
        return True
    
    def _compute_satisfaction_score(self, pattern: Pattern) -> Tuple[float, Dict[str, Dict]]:
        """Compute weighted satisfaction score"""
        self.stats['patterns_evaluated'] += 1
        
        satisfaction_details = {}
        total_score = 0.0
        
        for sc in self.soft_constraints:
            satisfaction = sc.compute_satisfaction(pattern)
            weighted_satisfaction = satisfaction * sc.normalized_weight
            total_score += weighted_satisfaction
            
            satisfaction_details[sc.constraint.name] = {
                'satisfaction': satisfaction,
                'weight': sc.weight,
                'normalized_weight': sc.normalized_weight,
                'weighted': weighted_satisfaction,
                'required': sc.required
            }
        
        return total_score, satisfaction_details
    
    def mine_with_soft_constraints(self) -> List[Tuple[Pattern, int, float, Dict]]:
        """Mine patterns with soft constraint evaluation"""
        if self.verbose:
            print(f"{'─'*70}")
            print(f"Starting Soft Constraint Mining")
            print(f"{'─'*70}\n")
        
        start_time = time.time()
        
        # Generate candidates
        if self.verbose:
            print("Generating candidate patterns...")
        
        candidates = []
        
        for label, count in self.label_stats['vertex_counts'].items():
            if count >= self.min_support_count:
                p = Pattern()
                p.add_vertex(label)
                p.support = count
                p.graph_ids = self.label_stats['vertex_labels'][label].copy()
                candidates.append((p, count))
        
        if self.verbose:
            print(f"  Generated {len(candidates)} candidates\n")
        
        # Evaluate with soft constraints
        if self.verbose:
            print("Evaluating soft constraints...")
        
        accepted_patterns = []
        
        for pattern, support in candidates:
            # Check required constraints first
            if not self._check_required_constraints(pattern):
                continue
            
            # Compute satisfaction score
            score, details = self._compute_satisfaction_score(pattern)
            
            # Accept if above threshold
            if score >= self.min_satisfaction:
                accepted_patterns.append((pattern, support, score, details))
                self.stats['patterns_accepted'] += 1
            else:
                self.stats['patterns_rejected_satisfaction'] += 1
        
        if self.verbose:
            print(f"  Accepted {len(accepted_patterns)} patterns\n")
        
        # Sort by satisfaction score
        accepted_patterns.sort(key=lambda x: x[2], reverse=True)
        
        self.stats['runtime'] = time.time() - start_time
        
        if self.verbose:
            self._print_results(accepted_patterns)
        
        return accepted_patterns
    
    def _print_results(self, patterns: List[Tuple[Pattern, int, float, Dict]]):
        """Print soft constraint mining results"""
        print(f"{'='*70}")
        print(f"{'SOFT CONSTRAINT MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results:")
        print(f"  Patterns evaluated: {self.stats['patterns_evaluated']}")
        print(f"  Patterns accepted: {self.stats['patterns_accepted']}")
        print(f"  Rejected (required constraints): {self.stats['patterns_rejected_required']}")
        print(f"  Rejected (satisfaction): {self.stats['patterns_rejected_satisfaction']}")
        
        if self.stats['patterns_evaluated'] > 0:
            accept_rate = self.stats['patterns_accepted'] / self.stats['patterns_evaluated'] * 100
            print(f"  Acceptance rate: {accept_rate:.1f}%")
        
        print(f"  Runtime: {self.stats['runtime']:.2f}s")
        
        if patterns:
            scores = [s for _, _, s, _ in patterns]
            supports = [sup for _, sup, _, _ in patterns]
            sizes = [len(p.vertices) for p, _, _, _ in patterns]
            
            print(f"\n📈 Score Statistics:")
            print(f"  Mean: {np.mean(scores):.3f}")
            print(f"  Std Dev: {np.std(scores):.3f}")
            print(f"  Min: {np.min(scores):.3f}")
            print(f"  Max: {np.max(scores):.3f}")
            
            print(f"\n🏆 Top 20 Patterns by Satisfaction Score:")
            print(f"  {'─'*66}")
            print(f"  {'Rank':<6} {'Size':<6} {'Supp':<6} {'Score':<8} {'Top Constraints':<38}")
            print(f"  {'─'*66}")
            
            for i, (pattern, support, score, details) in enumerate(patterns[:20], 1):
                # Get top 2 constraint contributions
                sorted_constraints = sorted(
                    details.items(),
                    key=lambda x: x[1]['weighted'],
                    reverse=True
                )[:2]
                
                constraint_str = ", ".join([
                    f"{name.split('(')[0]}:{info['satisfaction']:.2f}"
                    for name, info in sorted_constraints
                ])
                
                print(f"  {i:<6} {len(pattern.vertices):<6} {support:<6} "
                      f"{score:<8.3f} {constraint_str:<38}")
            
            # Score distribution histogram
            print(f"\n📊 Score Distribution:")
            bins = np.linspace(self.min_satisfaction, 1.0, 11)
            hist, _ = np.histogram(scores, bins=bins)
            max_bar = max(hist) if len(hist) > 0 else 1
            
            for i in range(len(hist)):
                if max_bar > 0:
                    bar_length = int(hist[i] / max_bar * 40)
                    bar = '█' * bar_length
                else:
                    bar = ''
                print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {bar} ({hist[i]})")
            
            # Detailed breakdown for top pattern
            print(f"\n🔍 Detailed Breakdown - Top Pattern:")
            pattern, support, score, details = patterns[0]
            print(f"  Pattern: Size={len(pattern.vertices)}, Support={support}")
            print(f"  Overall Score: {score:.3f}")
            print(f"\n  Constraint Contributions:")
            
            sorted_details = sorted(details.items(), 
                                   key=lambda x: x[1]['weighted'], 
                                   reverse=True)
            
            for name, info in sorted_details:
                req_str = " [REQUIRED]" if info['required'] else ""
                print(f"    {name}:")
                print(f"      Satisfaction: {info['satisfaction']:.3f}")
                print(f"      Weight: {info['weight']:.3f} "
                      f"(normalized: {info['normalized_weight']:.3f})")
                print(f"      Contribution: {info['weighted']:.3f}{req_str}")
        
        print(f"\n{'='*70}\n")

def main():
    """Run soft constraint mining on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - SOFT CONSTRAINT SATISFACTION':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=100)
    
    # Define soft constraints for MUTAG
    from utils.constraints import (MaxSizeConstraint, MinSizeConstraint,
                                   ConnectedConstraint, DiameterConstraint)
    
    soft_constraints = [
        SoftConstraint(MaxSizeConstraint(12), weight=0.25, required=False),
        SoftConstraint(MinSizeConstraint(3), weight=0.20, required=False),
        SoftConstraint(ConnectedConstraint(), weight=0.40, required=True),  # Must be connected
        SoftConstraint(DiameterConstraint(6), weight=0.15, required=False),
    ]
    
    print(f"Soft Constraints:")
    for sc in soft_constraints:
        req = " [REQUIRED]" if sc.required else " [optional]"
        print(f"  {sc.constraint.name}: weight={sc.weight}{req}")
    
    # Run soft constraint mining
    miner = SoftConstraintGSPAN(
        database=graphs,
        min_support=0.15,
        soft_constraints=soft_constraints,
        min_satisfaction=0.65,  # Accept patterns with ≥65% satisfaction
        verbose=True
    )
    
    patterns = miner.mine_with_soft_constraints()
    
    # Visualize
    print(f"\nGenerating visualizations...")
    visualizer = PatternVisualizer()
    
    # Convert to (Pattern, support) tuples
    pattern_tuples = [(p, sup) for p, sup, _, _ in patterns]
    
    fig = visualizer.plot_pattern_distribution(pattern_tuples,
                                               title="MUTAG Soft Constraint Results")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    visualizer.save_all_plots('results')
    
    with open('results/soft_constraint_results.txt', 'w') as f:
        f.write("MUTAG - Soft Constraint Satisfaction Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Patterns accepted: {len(patterns)}\n")
        f.write(f"Min satisfaction: {miner.min_satisfaction}\n")
        f.write(f"Runtime: {miner.stats['runtime']:.2f}s\n\n")
        
        f.write("Top 20 Patterns:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Size':<6} {'Support':<8} {'Score':<10}\n")
        f.write("-"*70 + "\n")
        
        for i, (pattern, support, score, details) in enumerate(patterns[:20], 1):
            f.write(f"{i:<6} {len(pattern.vertices):<6} {support:<8} {score:<10.3f}\n")
    
    print(f"\n✓ Results saved to 'results/soft_constraint_results.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
