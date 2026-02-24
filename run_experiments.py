"""
Complete Experiment Suite
Runs all 4 novel contributions on MUTAG and generates comparison results
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.graph_loader import DatasetLoader
from utils.constraints import MUTAGConstraints
from utils.visualization import PatternVisualizer

# Import all implementations
from basic_constraint_gspan import BasicConstraintGSPAN
from adaptive_relaxation import AdaptiveConstraintGSPAN, RelaxableConstraint
from interactive_mining import InteractiveConstraintGSPAN
from multi_objective import MultiObjectiveGSPAN
from soft_constraints import SoftConstraintGSPAN, SoftConstraint

def run_all_experiments(dataset_size: int = 100, save_results: bool = True):
    """
    Run all 4 novel contributions and compare results
    
    Args:
        dataset_size: Number of graphs to use (None = all)
        save_results: Whether to save results to files
    """
    
    print("="*80)
    print(" "*20 + "COMPREHENSIVE EXPERIMENT SUITE")
    print(" "*25 + "MUTAG Dataset")
    print("="*80)
    
    # Create results directory
    if save_results:
        os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("\n" + "─"*80)
    print("Loading MUTAG dataset...")
    print("─"*80)
    
    loader = DatasetLoader()
    graphs = loader.load_mutag(subset_size=dataset_size)
    
    print(f"\nUsing {len(graphs)} graphs for experiments")
    
    # Results storage
    results = []
    all_patterns = {}
    
    # Base constraints
    base_constraints = MUTAGConstraints.basic_chemical()
    
    print(f"\nBase constraints:")
    for c in base_constraints:
        print(f"  - {c.name}")
    
    # ========================================================================
    # EXPERIMENT 1: Basic Constraint-Aware GSPAN (Baseline)
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "EXPERIMENT 1: BASIC CONSTRAINT GSPAN")
    print("="*80)
    
    try:
        start = time.time()
        basic_miner = BasicConstraintGSPAN(
            database=graphs,
            min_support=0.15,
            constraints=base_constraints,
            verbose=False
        )
        basic_patterns = basic_miner.mine()
        basic_time = time.time() - start
        
        all_patterns['basic'] = basic_patterns
        
        results.append({
            'Method': 'Basic Constraint GSPAN',
            'Patterns Found': len(basic_patterns),
            'Runtime (s)': basic_time,
            'Pruning Rate (%)': (basic_miner.stats['patterns_pruned_constraints'] / 
                                max(basic_miner.stats['patterns_generated'], 1) * 100),
            'Novel Contribution': 'Baseline',
            'Status': 'Success'
        })
        
        print(f"✓ Basic GSPAN completed: {len(basic_patterns)} patterns in {basic_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Basic GSPAN failed: {e}")
        results.append({
            'Method': 'Basic Constraint GSPAN',
            'Patterns Found': 0,
            'Runtime (s)': 0,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Baseline',
            'Status': f'Failed: {e}'
        })
    
    # ========================================================================
    # EXPERIMENT 2: Adaptive Relaxation
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "EXPERIMENT 2: ADAPTIVE RELAXATION")
    print("="*80)
    
    try:
        # Use restrictive constraints
        from utils.constraints import MaxSizeConstraint, MinSizeConstraint, DiameterConstraint, ConnectedConstraint
        
        restrictive_constraints = [
            MaxSizeConstraint(6),
            MinSizeConstraint(4),
            DiameterConstraint(3),
            ConnectedConstraint()
        ]
        
        start = time.time()
        adaptive_miner = AdaptiveConstraintGSPAN(
            database=graphs,
            min_support=0.15,
            constraints=restrictive_constraints,
            min_results=15,
            max_iterations=8,
            verbose=False
        )
        adaptive_patterns = adaptive_miner.mine_with_adaptation()
        adaptive_time = time.time() - start
        
        all_patterns['adaptive'] = adaptive_patterns
        
        results.append({
            'Method': 'Adaptive Relaxation',
            'Patterns Found': len(adaptive_patterns),
            'Runtime (s)': adaptive_time,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Q-learning based relaxation',
            'Status': 'Success'
        })
        
        print(f"✓ Adaptive completed: {len(adaptive_patterns)} patterns in {adaptive_time:.2f}s")
        print(f"  Relaxations performed: {adaptive_miner.stats['relaxations_performed']}")
        
    except Exception as e:
        print(f"✗ Adaptive failed: {e}")
        results.append({
            'Method': 'Adaptive Relaxation',
            'Patterns Found': 0,
            'Runtime (s)': 0,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Q-learning based relaxation',
            'Status': f'Failed: {e}'
        })
    
    # ========================================================================
    # EXPERIMENT 3: Interactive Mining
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "EXPERIMENT 3: INTERACTIVE MINING")
    print("="*80)
    
    try:
        start = time.time()
        interactive_miner = InteractiveConstraintGSPAN(
            database=graphs,
            min_support=0.15,
            verbose=False
        )
        interactive_patterns = interactive_miner.interactive_session(max_rounds=4)
        interactive_time = time.time() - start
        
        all_patterns['interactive'] = interactive_patterns
        
        results.append({
            'Method': 'Interactive Mining',
            'Patterns Found': len(interactive_patterns),
            'Runtime (s)': interactive_time,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Human-in-the-loop refinement',
            'Status': 'Success'
        })
        
        print(f"✓ Interactive completed: {len(interactive_patterns)} patterns in {interactive_time:.2f}s")
        print(f"  Rounds: {len(interactive_miner.session_history)}")
        
    except Exception as e:
        print(f"✗ Interactive failed: {e}")
        results.append({
            'Method': 'Interactive Mining',
            'Patterns Found': 0,
            'Runtime (s)': 0,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Human-in-the-loop refinement',
            'Status': f'Failed: {e}'
        })
    
    # ========================================================================
    # EXPERIMENT 4: Multi-Objective Optimization
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "EXPERIMENT 4: MULTI-OBJECTIVE")
    print("="*80)
    
    try:
        objectives = {
            'size': 0.25,
            'support': 0.35,
            'complexity': 0.20,
            'constraints': 0.20
        }
        
        start = time.time()
        mo_miner = MultiObjectiveGSPAN(
            database=graphs,
            min_support=0.15,
            constraints=base_constraints,
            objectives=objectives,
            verbose=False
        )
        all_mo_patterns, pareto_patterns = mo_miner.mine_pareto_optimal()
        mo_time = time.time() - start
        
        # Convert PatternScore to (Pattern, support) tuples
        mo_pattern_tuples = [(ps.pattern, ps.support) for ps in pareto_patterns]
        all_patterns['multi_objective'] = mo_pattern_tuples
        
        results.append({
            'Method': 'Multi-Objective',
            'Patterns Found': len(pareto_patterns),
            'Runtime (s)': mo_time,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Pareto-optimal patterns',
            'Status': 'Success'
        })
        
        print(f"✓ Multi-objective completed: {len(pareto_patterns)} Pareto-optimal patterns in {mo_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Multi-objective failed: {e}")
        results.append({
            'Method': 'Multi-Objective',
            'Patterns Found': 0,
            'Runtime (s)': 0,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Pareto-optimal patterns',
            'Status': f'Failed: {e}'
        })
    
    # ========================================================================
    # EXPERIMENT 5: Soft Constraints
    # ========================================================================
    print("\n" + "="*80)
    print(" "*20 + "EXPERIMENT 5: SOFT CONSTRAINTS")
    print("="*80)
    
    try:
        from utils.constraints import MaxSizeConstraint, MinSizeConstraint, ConnectedConstraint, DiameterConstraint
        
        soft_constraints = [
            SoftConstraint(MaxSizeConstraint(12), weight=0.25),
            SoftConstraint(MinSizeConstraint(3), weight=0.20),
            SoftConstraint(ConnectedConstraint(), weight=0.40, required=True),
            SoftConstraint(DiameterConstraint(6), weight=0.15)
        ]
        
        start = time.time()
        soft_miner = SoftConstraintGSPAN(
            database=graphs,
            min_support=0.15,
            soft_constraints=soft_constraints,
            min_satisfaction=0.65,
            verbose=False
        )
        soft_patterns = soft_miner.mine_with_soft_constraints()
        soft_time = time.time() - start
        
        # Convert to (Pattern, support) tuples
        soft_pattern_tuples = [(p, sup) for p, sup, _, _ in soft_patterns]
        all_patterns['soft'] = soft_pattern_tuples
        
        results.append({
            'Method': 'Soft Constraints',
            'Patterns Found': len(soft_patterns),
            'Runtime (s)': soft_time,
            'Pruning Rate (%)': (1 - soft_miner.stats['patterns_accepted'] / 
                                max(soft_miner.stats['patterns_evaluated'], 1)) * 100,
            'Novel Contribution': 'Fuzzy satisfaction scoring',
            'Status': 'Success'
        })
        
        print(f"✓ Soft constraints completed: {len(soft_patterns)} patterns in {soft_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Soft constraints failed: {e}")
        results.append({
            'Method': 'Soft Constraints',
            'Patterns Found': 0,
            'Runtime (s)': 0,
            'Pruning Rate (%)': 0,
            'Novel Contribution': 'Fuzzy satisfaction scoring',
            'Status': f'Failed: {e}'
        })
    
    # ========================================================================
    # COMPARISON RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print(" "*25 + "COMPARATIVE RESULTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    successful_results = df[df['Status'] == 'Success']
    
    if len(successful_results) > 0:
        fastest_idx = successful_results['Runtime (s)'].idxmin()
        print(f"\n⚡ Fastest Method: {successful_results.loc[fastest_idx, 'Method']}")
        print(f"   Runtime: {successful_results.loc[fastest_idx, 'Runtime (s)']:.2f}s")
        
        most_patterns_idx = successful_results['Patterns Found'].idxmax()
        print(f"\n📊 Most Patterns Found: {successful_results.loc[most_patterns_idx, 'Method']}")
        print(f"   Patterns: {successful_results.loc[most_patterns_idx, 'Patterns Found']}")
        
        if successful_results['Pruning Rate (%)'].max() > 0:
            best_pruning_idx = successful_results['Pruning Rate (%)'].idxmax()
            print(f"\n✂️  Best Pruning Rate: {successful_results.loc[best_pruning_idx, 'Method']}")
            print(f"   Rate: {successful_results.loc[best_pruning_idx, 'Pruning Rate (%)']:.1f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    if save_results:
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save comparison table
        df.to_csv('results/comparison_results.csv', index=False)
        print("\n✓ Saved comparison table to 'results/comparison_results.csv'")
        
        # Save detailed report
        with open('results/experiment_report.txt', 'w') as f:
            f.write("MUTAG DATASET - COMPREHENSIVE EXPERIMENT REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset Size: {len(graphs)} graphs\n")
            f.write(f"Experiments Run: {len(results)}\n")
            f.write(f"Successful: {len(successful_results)}\n\n")
            
            f.write("RESULTS:\n")
            f.write("-"*80 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            if len(successful_results) > 0:
                f.write("SUMMARY:\n")
                f.write("-"*80 + "\n")
                fastest_idx = successful_results['Runtime (s)'].idxmin()
                f.write(f"Fastest: {successful_results.loc[fastest_idx, 'Method']} ")
                f.write(f"({successful_results.loc[fastest_idx, 'Runtime (s)']:.2f}s)\n")
                
                most_idx = successful_results['Patterns Found'].idxmax()
                f.write(f"Most patterns: {successful_results.loc[most_idx, 'Method']} ")
                f.write(f"({successful_results.loc[most_idx, 'Patterns Found']} patterns)\n")
        
        print("✓ Saved detailed report to 'results/experiment_report.txt'")
        
        # Generate comparison visualization
        print("\n" + "─"*80)
        print("Generating comparison visualizations...")
        print("─"*80)
        
        visualizer = PatternVisualizer()
        
        # Create comparison stats
        stats_list = []
        method_names = []
        
        for _, row in successful_results.iterrows():
            stats_list.append({
                'runtime': row['Runtime (s)'],
                'patterns_found': row['Patterns Found'],
                'pruning_rate': row['Pruning Rate (%)']
            })
            method_names.append(row['Method'])
        
        if stats_list:
            fig = visualizer.plot_mining_performance(stats_list, method_names)
            fig.savefig('results/method_comparison.png', dpi=150, bbox_inches='tight')
            print("✓ Saved comparison chart to 'results/method_comparison.png'")
        
        plt.close('all')
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nTotal successful experiments: {len(successful_results)}/{len(results)}")
    print(f"Results saved to: ./results/")
    print("="*80 + "\n")
    
    return df, all_patterns

def main():
    """Main execution"""
    
    # Run experiments with subset for speed
    results_df, patterns = run_all_experiments(
        dataset_size=100,  # Use 100 graphs for reasonable speed
        save_results=True
    )
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("Check the 'results/' directory for detailed outputs.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
