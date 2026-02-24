"""
Constraint-Aware GSPAN Research Package
Optimized for MUTAG dataset
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .graph_loader import Graph, Pattern, Vertex, Edge, DatasetLoader
from .constraints import (
    Constraint, MaxSizeConstraint, MinSizeConstraint,
    ConnectedConstraint, ForbiddenLabelConstraint,
    MustContainLabelConstraint, DiameterConstraint,
    LabelCountConstraint, ConstraintManager
)

__all__ = [
    'Graph', 'Pattern', 'Vertex', 'Edge', 'DatasetLoader',
    'Constraint', 'MaxSizeConstraint', 'MinSizeConstraint',
    'ConnectedConstraint', 'ForbiddenLabelConstraint',
    'MustContainLabelConstraint', 'DiameterConstraint',
    'LabelCountConstraint', 'ConstraintManager'
]
