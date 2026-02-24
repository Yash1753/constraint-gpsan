"""
Constraint definitions for graph pattern mining
Optimized for MUTAG chemical constraints
"""

from typing import Set, List
from abc import ABC, abstractmethod
from collections import defaultdict

try:
    from .graph_loader import Pattern, Graph
except ImportError:
    from graph_loader import Pattern, Graph

class Constraint(ABC):
    """Base class for all constraints"""
    
    def __init__(self, name: str, constraint_type: str):
        self.name = name
        self.type = constraint_type  # antimonotone, monotone, succinct, loose
        self.check_count = 0
        self.prune_count = 0
    
    @abstractmethod
    def check(self, pattern: Pattern) -> bool:
        """Check if pattern satisfies constraint"""
        pass
    
    @abstractmethod
    def can_satisfy(self, pattern: Pattern) -> bool:
        """Check if pattern can potentially satisfy constraint"""
        pass
    
    def get_stats(self):
        return {
            'name': self.name,
            'checks': self.check_count,
            'prunes': self.prune_count,
            'prune_rate': self.prune_count / max(self.check_count, 1)
        }

class MaxSizeConstraint(Constraint):
    """Anti-monotone: Maximum pattern size"""
    
    def __init__(self, max_size: int):
        super().__init__(f"MaxSize({max_size})", "antimonotone")
        self.max_size = max_size
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        result = len(pattern.vertices) <= self.max_size
        if not result:
            self.prune_count += 1
        return result
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return self.check(pattern)

class MinSizeConstraint(Constraint):
    """Monotone: Minimum pattern size"""
    
    def __init__(self, min_size: int):
        super().__init__(f"MinSize({min_size})", "monotone")
        self.min_size = min_size
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        return len(pattern.vertices) >= self.min_size
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return True  # Can always grow

class MustContainLabelConstraint(Constraint):
    """Monotone: Pattern must contain specific vertex label"""
    
    def __init__(self, label: int, label_name: str = None):
        name = f"MustContain({label_name or label})"
        super().__init__(name, "monotone")
        self.label = label
        self.label_name = label_name
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        return self.label in [v.label for v in pattern.vertices]
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return True  # Can add label later

class ForbiddenLabelConstraint(Constraint):
    """Anti-monotone: Pattern must not contain specific label"""
    
    def __init__(self, label: int, label_name: str = None):
        name = f"Forbidden({label_name or label})"
        super().__init__(name, "antimonotone")
        self.label = label
        self.label_name = label_name
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        result = self.label not in [v.label for v in pattern.vertices]
        if not result:
            self.prune_count += 1
        return result
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return self.check(pattern)

class ConnectedConstraint(Constraint):
    """Anti-monotone: Pattern must be connected"""
    
    def __init__(self):
        super().__init__("Connected", "antimonotone")
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        result = pattern.is_connected()
        if not result:
            self.prune_count += 1
        return result
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return self.check(pattern)

class LabelCountConstraint(Constraint):
    """Succinct: Specific label count constraint"""
    
    def __init__(self, label: int, min_count: int = 0, 
                 max_count: int = float('inf'), label_name: str = None):
        name = f"LabelCount({label_name or label},{min_count},{max_count})"
        super().__init__(name, "succinct")
        self.label = label
        self.min_count = min_count
        self.max_count = max_count
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        count = sum(1 for v in pattern.vertices if v.label == self.label)
        return self.min_count <= count <= self.max_count
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        count = sum(1 for v in pattern.vertices if v.label == self.label)
        return count <= self.max_count

class DiameterConstraint(Constraint):
    """Anti-monotone: Maximum graph diameter"""
    
    def __init__(self, max_diameter: int):
        super().__init__(f"MaxDiameter({max_diameter})", "antimonotone")
        self.max_diameter = max_diameter
    
    def check(self, pattern: Pattern) -> bool:
        self.check_count += 1
        diameter = self._compute_diameter(pattern)
        result = diameter <= self.max_diameter
        if not result:
            self.prune_count += 1
        return result
    
    def can_satisfy(self, pattern: Pattern) -> bool:
        return self.check(pattern)
    
    def _compute_diameter(self, pattern: Pattern) -> int:
        """BFS-based diameter computation"""
        if len(pattern.vertices) <= 1:
            return 0
        
        from collections import deque
        
        adj = defaultdict(list)
        for e in pattern.edges:
            adj[e.frm].append(e.to)
            adj[e.to].append(e.frm)
        
        max_dist = 0
        
        for start in range(len(pattern.vertices)):
            dist = {start: 0}
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        queue.append(v)
        
        return max_dist

class ConstraintManager:
    """Manages multiple constraints and optimizes checking order"""
    
    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        self.antimonotone = [c for c in constraints if c.type == "antimonotone"]
        self.monotone = [c for c in constraints if c.type == "monotone"]
        self.succinct = [c for c in constraints if c.type == "succinct"]
        self.loose = [c for c in constraints if c.type == "loose"]
        
        # Order antimonotone by estimated cost
        self.antimonotone.sort(key=lambda c: self._estimate_cost(c))
    
    def _estimate_cost(self, constraint: Constraint) -> int:
        """Estimate computational cost"""
        cost_map = {
            'MaxSize': 1,
            'MinSize': 1,
            'ForbiddenLabel': 2,
            'Forbidden': 2,
            'MustContainLabel': 2,
            'MustContain': 2,
            'Connected': 10,
            'LabelCount': 3,
            'MaxDiameter': 50
        }
        
        for key in cost_map:
            if constraint.name.startswith(key):
                return cost_map[key]
        return 100
    
    def check_antimonotone(self, pattern: Pattern) -> bool:
        """Check all antimonotone constraints"""
        for constraint in self.antimonotone:
            if not constraint.check(pattern):
                return False
        return True
    
    def check_monotone(self, pattern: Pattern) -> bool:
        """Check all monotone constraints"""
        for constraint in self.monotone:
            if not constraint.check(pattern):
                return False
        return True
    
    def can_satisfy_all(self, pattern: Pattern) -> bool:
        """Check if pattern can potentially satisfy all constraints"""
        for constraint in self.constraints:
            if not constraint.can_satisfy(pattern):
                return False
        return True
    
    def get_all_stats(self):
        """Get statistics for all constraints"""
        return [c.get_stats() for c in self.constraints]

# MUTAG-specific constraint presets
class MUTAGConstraints:
    """Predefined constraint sets for MUTAG dataset"""
    
    # MUTAG atom labels (approximate, dataset-specific)
    ATOM_LABELS = {
        0: 'C',   # Carbon
        1: 'N',   # Nitrogen
        2: 'O',   # Oxygen
        3: 'F',   # Fluorine
        4: 'I',   # Iodine
        5: 'Cl',  # Chlorine
        6: 'Br'   # Bromine
    }
    
    @staticmethod
    def basic_chemical() -> List[Constraint]:
        """Basic chemical validity constraints"""
        return [
            MaxSizeConstraint(20),        # Typical drug-like molecules
            MinSizeConstraint(3),         # At least 3 atoms
            ConnectedConstraint()         # Must be single molecule
        ]
    
    @staticmethod
    def small_fragments() -> List[Constraint]:
        """Find small molecular fragments"""
        return [
            MaxSizeConstraint(10),
            MinSizeConstraint(3),
            DiameterConstraint(5),
            ConnectedConstraint()
        ]
    
    @staticmethod
    def carbon_backbone() -> List[Constraint]:
        """Patterns with carbon backbone"""
        return [
            MaxSizeConstraint(15),
            MinSizeConstraint(4),
            MustContainLabelConstraint(0, 'C'),  # Must have carbon
            ConnectedConstraint()
        ]
    
    @staticmethod
    def no_halogens() -> List[Constraint]:
        """Exclude halogen-containing patterns"""
        return [
            MaxSizeConstraint(15),
            MinSizeConstraint(3),
            ForbiddenLabelConstraint(3, 'F'),
            ForbiddenLabelConstraint(4, 'I'),
            ForbiddenLabelConstraint(5, 'Cl'),
            ForbiddenLabelConstraint(6, 'Br'),
            ConnectedConstraint()
        ]
    
    @staticmethod
    def functional_groups() -> List[Constraint]:
        """Find functional group patterns"""
        return [
            MaxSizeConstraint(8),
            MinSizeConstraint(2),
            DiameterConstraint(4),
            ConnectedConstraint()
        ]

# Test
if __name__ == "__main__":
    from graph_loader import Pattern
    
    # Create test pattern
    p = Pattern()
    p.add_vertex(0)  # Carbon
    p.add_vertex(1)  # Nitrogen
    p.add_vertex(0)  # Carbon
    p.add_edge(0, 1)
    p.add_edge(1, 2)
    
    # Test constraints
    constraints = MUTAGConstraints.basic_chemical()
    manager = ConstraintManager(constraints)
    
    print("Testing pattern:", p)
    print("Antimonotone check:", manager.check_antimonotone(p))
    print("Monotone check:", manager.check_monotone(p))
    print("Can satisfy all:", manager.can_satisfy_all(p))
