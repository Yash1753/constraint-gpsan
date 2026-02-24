"""
Main execution file for Proper GSPAN
Run this to use the real GSPAN algorithm
"""

import time
import os
from utils.graph_loader import DatasetLoader
from utils.gspan_algorithm import ProperGSPAN
from utils.constraints import MUTAGConstraints, MaxSizeConstraint

def main():
    """Run proper GSPAN on MUTAG dataset"""
    
    print(f"\n{'='*70}")
    print(f"{'MUTAG - PROPER GSPAN ALGORITHM':^70}")
    print(f"{'='*70}\n")
    
    # Load MUTAG dataset
    print("Loading MUTAG dataset...")
    loader = DatasetLoader()
    
    # Use subset for reasonable runtime
    # Full dataset (188) will take longer but is correct
    graphs = loader.load_mutag(subset_size=50)
    
    # Define constraints
    constraints = MUTAGConstraints.basic_chemical()
    
    # Add max size to prevent explosion
    constraints.append(MaxSizeConstraint(6))
    
    print(f"\nConstraints:")
    for i, c in enumerate(constraints, 1):
        print(f"  {i}. {c.name}")
    
    # Run PROPER GSPAN
    print(f"\n{'─'*70}")
    print("Running PROPER GSPAN Algorithm")
    print(f"{'─'*70}")
    
    gspan = ProperGSPAN(
        database=graphs,
        min_support=0.3,  # 30% support
        constraints=constraints,
        max_pattern_size=6,  # Limit pattern size
        verbose=True
    )
    
    patterns = gspan.mine()
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    
    with open('results/proper_gspan_patterns.txt', 'w') as f:
        f.write("MUTAG - Proper GSPAN Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total patterns: {len(patterns)}\n")
        f.write(f"Runtime: {gspan.stats['runtime']:.2f}s\n\n")
        f.write("Patterns (sorted by support):\n")
        f.write("-"*70 + "\n")
        
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
        for i, (pattern, support) in enumerate(sorted_patterns, 1):
            labels = [v.label for v in pattern.vertices]
            f.write(f"{i:3d}. Labels={labels}, Size={len(pattern.vertices)}, "
                   f"Support={support}\n")
    
    print(f"✓ Results saved to 'results/proper_gspan_patterns.txt'")
    
    # Performance summary
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"  Dataset: MUTAG ({len(graphs)} graphs)")
    print(f"  Patterns found: {len(patterns)}")
    print(f"  Runtime: {gspan.stats['runtime']:.2f} seconds")
    print(f"  Avg time per pattern: {gspan.stats['runtime']/max(len(patterns),1):.3f}s")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()