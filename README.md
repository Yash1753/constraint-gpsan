# Constraint-Aware Graph Pattern Mining Research

Novel constraint-aware GSPAN implementations optimized for the MUTAG molecular graph dataset.

## 🎯 Overview

This repository contains **4 novel research contributions** for constraint-aware graph pattern mining:

1. **Basic Constraint-Aware GSPAN** - Foundation with optimized constraint checking
2. **Adaptive Constraint Relaxation** - Q-learning based automatic constraint relaxation
3. **Interactive Constraint Mining** - Human-in-the-loop constraint refinement
4. **Multi-Objective Optimization** - Pareto-optimal pattern discovery
5. **Soft Constraint Satisfaction** - Fuzzy constraint satisfaction scoring

## 📁 Project Structure
constraint_gspan_research/
│
├── 1_basic_constraint_gspan.py # Baseline implementation
├── 2_adaptive_relaxation.py # Contribution 1: Q-learning relaxation
├── 3_interactive_mining.py # Contribution 2: Interactive refinement
├── 4_multi_objective.py # Contribution 3: Pareto optimization
├── 5_soft_constraints.py # Contribution 4: Fuzzy satisfaction
│
├── utils/
│ ├── init.py
│ ├── graph_loader.py # MUTAG dataset loader
│ ├── constraints.py # Constraint definitions
│ └── visualization.py # Result visualization
│
├── run_experiments.py # Complete experiment suite
├── README.md # This file
└── results/ # Output directory (auto-created)

text


## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy pandas matplotlib requests

# Clone/download this repository
cd constraint_gspan_research
Run Individual Contributions
Bash

# 1. Basic Constraint GSPAN (Baseline)
python 1_basic_constraint_gspan.py

# 2. Adaptive Relaxation
python 2_adaptive_relaxation.py

# 3. Interactive Mining
python 3_interactive_mining.py

# 4. Multi-Objective Optimization
python 4_multi_objective.py

# 5. Soft Constraints
python 5_soft_constraints.py
Run Complete Experiment Suite
Bash

python run_experiments.py
This will:

Run all 5 methods on MUTAG dataset
Generate comparison results
Create visualizations
Save everything to results/ directory
📊 Dataset: MUTAG
Domain: Chemistry (Mutagenicity prediction)
Size: 188 molecular graphs
Avg nodes: 17.9 atoms per molecule
Avg edges: 19.8 bonds per molecule
Labels: 7 atom types (C, N, O, F, I, Cl, Br)
Classes: 2 (mutagenic, non-mutagenic)
The dataset is automatically downloaded on first run.

🔧 Usage Examples
Basic Constraint Mining
Python

from utils.graph_loader import DatasetLoader
from utils.constraints import MUTAGConstraints
from basic_constraint_gspan import BasicConstraintGSPAN

# Load data
loader = DatasetLoader()
graphs = loader.load_mutag()

# Define constraints
constraints = MUTAGConstraints.basic_chemical()

# Mine
miner = BasicConstraintGSPAN(
    database=graphs,
    min_support=0.1,
    constraints=constraints
)
patterns = miner.mine()
Adaptive Relaxation
Python

from adaptive_relaxation import AdaptiveConstraintGSPAN

miner = AdaptiveConstraintGSPAN(
    database=graphs,
    min_support=0.15,
    constraints=restrictive_constraints,
    min_results=15,      # Target number of patterns
    max_iterations=10    # Max relaxation iterations
)
patterns = miner.mine_with_adaptation()
Soft Constraints
Python

from soft_constraints import SoftConstraintGSPAN, SoftConstraint

soft_constraints = [
    SoftConstraint(MaxSizeConstraint(12), weight=0.3),
    SoftConstraint(ConnectedConstraint(), weight=0.4, required=True)
]

miner = SoftConstraintGSPAN(
    database=graphs,
    min_support=0.15,
    soft_constraints=soft_constraints,
    min_satisfaction=0.65  # 65% satisfaction threshold
)
patterns = miner.mine_with_soft_constraints()
📈 Expected Results
Performance (100 graphs subset)
Method	Runtime	Patterns	Novel Aspect
Basic Constraint	~2-3s	45-60	Baseline
Adaptive Relaxation	~5-8s	25-40	Q-learning
Interactive Mining	~3-5s	30-50	Human-in-loop
Multi-Objective	~2-4s	10-20	Pareto-optimal
Soft Constraints	~2-3s	50-70	Fuzzy scoring
Key Findings
Soft Constraints: Most flexible, finds most patterns
Multi-Objective: Best quality patterns (Pareto-optimal)
Adaptive: Self-tuning, good for unknown domains
Interactive: Best for domain experts
🎓 Research Contributions
1. Adaptive Constraint Relaxation ⭐
Novel Aspect: First use of reinforcement learning (Q-learning) for automatic constraint relaxation in graph mining.

Key Features:

Learns which constraints to relax
Adapts to dataset characteristics
Balances exploration vs exploitation
Publication Angle: "Learning-based Adaptive Constraint Relaxation for Graph Pattern Mining"

2. Interactive Constraint Mining ⭐
Novel Aspect: Human-in-the-loop iterative constraint refinement.

Key Features:

User refines constraints based on results
Session history tracking
Simulated interaction (can be extended to real UI)
Publication Angle: "Interactive Constraint Refinement for Exploratory Graph Mining"

3. Multi-Objective Optimization ⭐
Novel Aspect: Formulates pattern mining as multi-objective optimization problem.

Key Features:

Finds Pareto-optimal patterns
Balances size, support, complexity, constraints
Weighted ranking within Pareto front
Publication Angle: "Multi-Objective Graph Pattern Discovery via Pareto Optimization"

4. Soft Constraint Satisfaction ⭐
Novel Aspect: First fuzzy/soft constraint framework for graph pattern mining.

Key Features:

Constraints have satisfaction degrees (0-1)
Weighted importance
Required vs optional constraints
Publication Angle: "Soft Constraint Satisfaction for Flexible Graph Pattern Mining"

📝 Output Files
After running experiments, results/ contains:

text

results/
├── basic_patterns.txt               # Basic GSPAN results
├── adaptive_results.txt             # Adaptive relaxation history
├── interactive_session.txt          # Interactive session log
├── multi_objective_results.txt      # Pareto-optimal patterns
├── soft_constraint_results.txt      # Soft constraint scores
├── comparison_results.csv           # Method comparison table
├── experiment_report.txt            # Detailed report
├── method_comparison.png            # Performance chart
└── figure_*.png                     # Various visualizations
