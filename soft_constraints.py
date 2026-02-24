"""
NOVEL CONTRIBUTION 4: Soft Constraint Satisfaction with PROPER GSPAN
Integrates fuzzy constraints with real GSPAN algorithm
"""

import time
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import defaultdict
from utils.graph_loader import Graph, Pattern, DatasetLoader
from utils.constraints import Constraint, MUTAGConstraints
from utils.gspan_algorithm import ProperGSPAN  # ← Use proper GSPAN!
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

class SoftConstraintProperGSPAN:
    """
    Proper GSPAN with soft constraint post-filtering and ranking
    
    APPROACH: 
    1. Use proper GSPAN to mine ALL patterns (with hard constraints)
    2. Post-process with soft constraints for ranking
    """
    
    def __init__(self,
                 database: List[Graph],
                 min_support: float,
                 soft_constraints: List[SoftConstraint],
                 hard_constraints: List[Constraint] = None,
                 min_satisfaction: float = 0.6,
                 max_pattern_size: int = 6,
                 verbose: bool = True):
        
        self.database = database
        self.min_support = min_support
        self.soft_constraints = soft_constraints
        self.hard_constraints = hard_constraints or []
        self.min_satisfaction = min_satisfaction
        self.max_pattern_size = max_pattern_size
        self.verbose = verbose
        
        # Separate required soft constraints (use as hard constraints)
        self.required_soft_constraints = [sc for sc in soft_constraints if sc.required]
        
        # Add required soft constraints to hard constraints
        for sc in self.required_soft_constraints:
            self.hard_constraints.append(sc.constraint)
        
        # Normalize weights
        total_weight = sum(sc.weight for sc in soft_constraints)
        if total_weight > 0:
            for sc in soft_constraints:
                sc.normalized_weight = sc.weight / total_weight
        
        # Statistics
        self.stats = {
            'patterns_mined': 0,
            'patterns_evaluated': 0,
            'patterns_accepted': 0,
            'patterns_rejected_satisfaction': 0,
            'mining_runtime': 0,
            'evaluation_runtime': 0,
            'total_runtime': 0
        }
        
        if self.verbose:
            self._print_initialization()
    
    def _print_initialization(self):
        """Print initialization info"""
        print(f"\n{'='*70}")
        print(f"{'SOFT CONSTRAINT PROPER GSPAN':^70}")
        print(f"{'='*70}")
        print(f"  Database: {len(self.database)} graphs")
        print(f"  Min support: {int(self.min_support * len(self.database))}")
        print(f"  Min satisfaction: {self.min_satisfaction:.2f}")
        print(f"  Max pattern size: {self.max_pattern_size}")
        
        print(f"\n  Hard Constraints ({len(self.hard_constraints)}):")
        for i, c in enumerate(self.hard_constraints, 1):
            print(f"    {i}. {c.name}")
        
        print(f"\n  Soft Constraints ({len(self.soft_constraints)}):")
        for i, sc in enumerate(self.soft_constraints, 1):
            req_str = " [REQUIRED]" if sc.required else ""
            print(f"    {i}. {sc.constraint.name}: weight={sc.weight:.2f}{req_str}")
        
        print(f"{'='*70}\n")
    
    def mine(self) -> List[Tuple[Pattern, int, float, Dict]]:
        """
        Main mining with soft constraints
        
        Returns:
            List of (pattern, support, satisfaction_score, details)
        """
        
        start_time = time.time()
        
        # STEP 1: Use proper GSPAN to mine patterns
        if self.verbose:
            print(f"{'─'*70}")
            print("STEP 1: Mining patterns with Proper GSPAN")
            print(f"{'─'*70}\n")
        
        mining_start = time.time()
        
        gspan = ProperGSPAN(
            database=self.database,
            min_support=self.min_support,
            constraints=self.hard_constraints,
            max_pattern_size=self.max_pattern_size,
            verbose=self.verbose
        )
        
        mined_patterns = gspan.mine()  # Returns List[Tuple[Pattern, int]]
        self.stats['patterns_mined'] = len(mined_patterns)
        self.stats['mining_runtime'] = time.time() - mining_start
        
        if self.verbose:
            print(f"\n✓ Mined {len(mined_patterns)} patterns in {self.stats['mining_runtime']:.2f}s")
        
        # STEP 2: Evaluate with soft constraints
        if self.verbose:
            print(f"\n{'─'*70}")
            print("STEP 2: Evaluating patterns with Soft Constraints")
            print(f"{'─'*70}\n")
        
        eval_start = time.time()
        
        scored_patterns = []
        
        for pattern, support in mined_patterns:
            self.stats['patterns_evaluated'] += 1
            
            # Compute soft constraint satisfaction
            score, details = self._compute_satisfaction_score(pattern)
            
            # Accept if above threshold
            if score >= self.min_satisfaction:
                scored_patterns.append((pattern, support, score, details))
                self.stats['patterns_accepted'] += 1
            else:
                self.stats['patterns_rejected_satisfaction'] += 1
        
        # Sort by satisfaction score (descending)
        scored_patterns.sort(key=lambda x: x[2], reverse=True)
        
        self.stats['evaluation_runtime'] = time.time() - eval_start
        self.stats['total_runtime'] = time.time() - start_time
        
        if self.verbose:
            print(f"✓ Evaluated {self.stats['patterns_evaluated']} patterns")
            print(f"✓ Accepted {self.stats['patterns_accepted']} patterns")
            print(f"✓ Evaluation took {self.stats['evaluation_runtime']:.2f}s\n")
        
        if self.verbose:
            self._print_results(scored_patterns)
        
        return scored_patterns
    
    def _compute_satisfaction_score(self, pattern: Pattern) -> Tuple[float, Dict[str, Dict]]:
        """Compute weighted satisfaction score"""
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
    
    def _print_results(self, patterns: List[Tuple[Pattern, int, float, Dict]]):
        """Print mining results"""
        print(f"{'='*70}")
        print(f"{'SOFT CONSTRAINT MINING COMPLETE':^70}")
        print(f"{'='*70}")
        
        print(f"\n📊 Results:")
        print(f"  Patterns mined (GSPAN): {self.stats['patterns_mined']}")
        print(f"  Patterns accepted (Soft): {self.stats['patterns_accepted']}")
        print(f"  Acceptance rate: {self.stats['patterns_accepted']/max(self.stats['patterns_mined'],1)*100:.1f}%")
        
        print(f"\n⏱️  Runtime:")
        print(f"  GSPAN mining: {self.stats['mining_runtime']:.2f}s")
        print(f"  Soft evaluation: {self.stats['evaluation_runtime']:.2f}s")
        print(f"  Total: {self.stats['total_runtime']:.2f}s")
        
        if patterns:
            scores = [s for _, _, s, _ in patterns]
            
            print(f"\n📈 Score Statistics:")
            print(f"  Mean: {np.mean(scores):.3f}")
            print(f"  Std Dev: {np.std(scores):.3f}")
            print(f"  Min: {np.min(scores):.3f}")
            print(f"  Max: {np.max(scores):.3f}")
            
            print(f"\n🏆 Top 15 Patterns by Satisfaction Score:")
            print(f"  {'─'*66}")
            print(f"  {'Rank':<6} {'Size':<6} {'Edges':<6} {'Supp':<6} {'Score':<8} {'Top Constraints':<28}")
            print(f"  {'─'*66}")
            
            for i, (pattern, support, score, details) in enumerate(patterns[:15], 1):
                # Get top 2 constraints
                sorted_constraints = sorted(
                    details.items(),
                    key=lambda x: x[1]['weighted'],
                    reverse=True
                )[:2]
                
                constraint_str = ", ".join([
                    f"{name.split('(')[0][:8]}:{info['satisfaction']:.2f}"
                    for name, info in sorted_constraints
                ])
                
                print(f"  {i:<6} {len(pattern.vertices):<6} {len(pattern.edges):<6} "
                      f"{support:<6} {score:<8.3f} {constraint_str:<28}")
            
            # Score distribution
            print(f"\n📊 Score Distribution:")
            bins = np.linspace(self.min_satisfaction, 1.0, 11)
            hist, _ = np.histogram(scores, bins=bins)
            max_bar = max(hist) if len(hist) > 0 else 1
            
            for i in range(len(hist)):
                bar_length = int(hist[i] / max_bar * 40) if max_bar > 0 else 0
                bar = '█' * bar_length
                print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {bar} ({hist[i]})")
        
        print(f"\n{'='*70}\n")

def main():
    """Run soft constraint mining with proper GSPAN on MUTAG"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - SOFT CONSTRAINTS + PROPER GSPAN':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=50)  # Use 50 for reasonable speed
    
    # Define soft constraints
    from utils.constraints import (MaxSizeConstraint, MinSizeConstraint,
                                   ConnectedConstraint, DiameterConstraint)
    
    soft_constraints = [
        SoftConstraint(MaxSizeConstraint(6), weight=0.25, required=False),
        SoftConstraint(MinSizeConstraint(2), weight=0.20, required=False),
        SoftConstraint(ConnectedConstraint(), weight=0.40, required=True),
        SoftConstraint(DiameterConstraint(4), weight=0.15, required=False),
    ]
    
    # Hard constraints for GSPAN
    hard_constraints = MUTAGConstraints.basic_chemical()
    
    print(f"Configuration:")
    print(f"  Dataset: {len(graphs)} graphs")
    print(f"  Min support: 0.25 (25%)")
    print(f"  Min satisfaction: 0.6 (60%)")
    print(f"  Max pattern size: 5")
    
    # Run mining
    miner = SoftConstraintProperGSPAN(
        database=graphs,
        min_support=0.25,
        soft_constraints=soft_constraints,
        hard_constraints=hard_constraints,
        min_satisfaction=0.6,
        max_pattern_size=5,
        verbose=True
    )
    
    patterns = miner.mine()
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    with open('results/soft_proper_gspan_results.txt', 'w') as f:
        f.write("MUTAG - Soft Constraints + Proper GSPAN Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Patterns mined: {miner.stats['patterns_mined']}\n")
        f.write(f"Patterns accepted: {len(patterns)}\n")
        f.write(f"Mining runtime: {miner.stats['mining_runtime']:.2f}s\n")
        f.write(f"Total runtime: {miner.stats['total_runtime']:.2f}s\n\n")
        
        f.write("Top 20 Patterns:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Size':<6} {'Edges':<6} {'Support':<8} {'Score':<10}\n")
        f.write("-"*70 + "\n")
        
        for i, (pattern, support, score, details) in enumerate(patterns[:20], 1):
            f.write(f"{i:<6} {len(pattern.vertices):<6} {len(pattern.edges):<6} "
                   f"{support:<8} {score:<10.3f}\n")
    
    print(f"✓ Results saved to 'results/soft_proper_gspan_results.txt'")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()