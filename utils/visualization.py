"""
Visualization utilities for graph patterns and results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

try:
    from .graph_loader import Graph, Pattern
except ImportError:
    from graph_loader import Graph, Pattern

class PatternVisualizer:
    """Visualize patterns and mining results"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def plot_pattern_distribution(self, patterns: List[Tuple[Pattern, int]], 
                                  title: str = "Pattern Distribution"):
        """Plot distribution of pattern sizes and supports"""
        
        sizes = [len(p.vertices) for p, _ in patterns]
        supports = [s for _, s in patterns]
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Size distribution
        axes[0, 0].hist(sizes, bins=range(min(sizes), max(sizes)+2), 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Pattern Size (vertices)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Support distribution
        axes[0, 1].hist(supports, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Support')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Support Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Size vs Support scatter
        axes[1, 0].scatter(sizes, supports, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Pattern Size')
        axes[1, 0].set_ylabel('Support')
        axes[1, 0].set_title('Size vs Support')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Top patterns
        top_n = min(15, len(patterns))
        sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)[:top_n]
        pattern_labels = [f"P{i+1}" for i in range(top_n)]
        pattern_supports = [s for _, s in sorted_patterns]
        
        axes[1, 1].barh(pattern_labels, pattern_supports, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Support')
        axes[1, 1].set_title(f'Top {top_n} Patterns by Support')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_constraint_effectiveness(self, constraint_stats: List[Dict]):
        """Plot constraint pruning effectiveness"""
        
        names = [s['name'] for s in constraint_stats]
        prune_rates = [s['prune_rate'] * 100 for s in constraint_stats]
        check_counts = [s['checks'] for s in constraint_stats]
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Pruning rates
        colors = ['red' if r > 50 else 'orange' if r > 20 else 'green' 
                 for r in prune_rates]
        axes[0].barh(names, prune_rates, color=colors, alpha=0.7)
        axes[0].set_xlabel('Pruning Rate (%)')
        axes[0].set_title('Constraint Pruning Effectiveness')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Check counts
        axes[1].barh(names, check_counts, color='blue', alpha=0.7)
        axes[1].set_xlabel('Number of Checks')
        axes[1].set_title('Constraint Usage Frequency')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return fig
    
    def plot_mining_performance(self, stats_list: List[Dict], 
                                method_names: List[str]):
        """Compare performance of different mining methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Runtime comparison
        runtimes = [s.get('runtime', 0) for s in stats_list]
        axes[0, 0].bar(method_names, runtimes, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_ylabel('Runtime (seconds)')
        axes[0, 0].set_title('Runtime Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Patterns found
        patterns_found = [s.get('patterns_found', 0) for s in stats_list]
        axes[0, 1].bar(method_names, patterns_found, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Patterns Found')
        axes[0, 1].set_title('Number of Patterns')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Pruning efficiency
        pruning_rates = [s.get('pruning_rate', 0) for s in stats_list]
        axes[1, 0].bar(method_names, pruning_rates, color='coral', alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('Pruning Rate (%)')
        axes[1, 0].set_title('Pruning Efficiency')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Speedup ratio (relative to first method)
        if runtimes[0] > 0:
            speedups = [runtimes[0] / max(r, 0.001) for r in runtimes]
        else:
            speedups = [1.0] * len(runtimes)
        
        axes[1, 1].bar(method_names, speedups, color='gold', alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Speedup Factor')
        axes[1, 1].set_title('Speedup vs Baseline')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Baseline')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].legend()
        
        plt.suptitle('Mining Method Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_all_plots(self, output_dir: str = './results'):
        """Save all current figures"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)
            fig.savefig(f'{output_dir}/figure_{i+1}.png', dpi=150, bbox_inches='tight')
        
        print(f"✓ Saved {len(plt.get_fignums())} figures to {output_dir}")
