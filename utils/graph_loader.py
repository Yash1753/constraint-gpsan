"""
Graph data structures and MUTAG dataset loader
Optimized for chemical graph datasets
"""

import os
import pickle
import requests
import zipfile
from collections import defaultdict
from typing import List, Set, Dict, Tuple
import numpy as np

class Vertex:
    """Graph vertex with label"""
    def __init__(self, vid: int, label: int):
        self.vid = vid
        self.label = label
    
    def __repr__(self):
        return f"V({self.vid},{self.label})"
    
    def __hash__(self):
        return hash((self.vid, self.label))
    
    def __eq__(self, other):
        return self.vid == other.vid and self.label == other.label

class Edge:
    """Graph edge with label"""
    def __init__(self, frm: int, to: int, elabel: int = 0):
        self.frm = frm
        self.to = to
        self.elabel = elabel
    
    def __repr__(self):
        return f"E({self.frm}-{self.to},{self.elabel})"
    
    def __hash__(self):
        return hash((min(self.frm, self.to), max(self.frm, self.to), self.elabel))
    
    def __eq__(self, other):
        return ((self.frm == other.frm and self.to == other.to) or
                (self.frm == other.to and self.to == other.frm)) and \
               self.elabel == other.elabel

class Graph:
    """Graph structure for molecular/chemical graphs"""
    def __init__(self, gid: int = 0):
        self.gid = gid
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.vertex_map: Dict[int, Vertex] = {}
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.label = None  # Graph class label (e.g., mutagen/non-mutagen)
    
    def add_vertex(self, vid: int, vlabel: int):
        v = Vertex(vid, vlabel)
        self.vertices.append(v)
        self.vertex_map[vid] = v
        return v
    
    def add_edge(self, frm: int, to: int, elabel: int = 0):
        e = Edge(frm, to, elabel)
        self.edges.append(e)
        self.adj[frm].append(to)
        self.adj[to].append(frm)
        return e
    
    def get_vertex_labels(self) -> Set[int]:
        return set(v.label for v in self.vertices)
    
    def get_edge_labels(self) -> Set[int]:
        return set(e.elabel for e in self.edges)
    
    def is_connected(self) -> bool:
        if len(self.vertices) <= 1:
            return True
        
        visited = set()
        stack = [self.vertices[0].vid]
        
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            stack.extend(self.adj[v])
        
        return len(visited) == len(self.vertices)
    
    def __repr__(self):
        return f"Graph({self.gid}, V={len(self.vertices)}, E={len(self.edges)})"

class Pattern:
    """Graph pattern (subgraph structure)"""
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.support = 0
        self.graph_ids: Set[int] = set()
    
    def add_vertex(self, vlabel: int):
        vid = len(self.vertices)
        v = Vertex(vid, vlabel)
        self.vertices.append(v)
        return v
    
    def add_edge(self, frm: int, to: int, elabel: int = 0):
        e = Edge(frm, to, elabel)
        self.edges.append(e)
        return e
    
    def copy(self):
        p = Pattern()
        p.vertices = [Vertex(v.vid, v.label) for v in self.vertices]
        p.edges = [Edge(e.frm, e.to, e.elabel) for e in self.edges]
        p.support = self.support
        p.graph_ids = self.graph_ids.copy()
        return p
    
    def is_connected(self) -> bool:
        if len(self.vertices) <= 1:
            return True
        
        adj = defaultdict(list)
        for e in self.edges:
            adj[e.frm].append(e.to)
            adj[e.to].append(e.frm)
        
        visited = set()
        stack = [self.vertices[0].vid]
        
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            stack.extend(adj[v])
        
        return len(visited) == len(self.vertices)
    
    def __repr__(self):
        labels = [v.label for v in self.vertices]
        return f"Pattern(labels={labels}, V={len(self.vertices)}, E={len(self.edges)}, sup={self.support})"

class DatasetLoader:
    """Load MUTAG and other TU Dortmund datasets"""
    
    DATASETS = {
        'MUTAG': {
            'url': 'https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip',
            'description': 'Mutagenicity prediction (188 molecular graphs)',
            'num_graphs': 188,
            'avg_nodes': 17.9,
            'avg_edges': 19.8,
            'num_classes': 2,
            'node_labels': 7,  # C, N, O, F, I, Cl, Br
            'recommended_min_support': 0.1
        }
    }
    
    def __init__(self, data_dir: str = './data', use_cache: bool = True):
        self.data_dir = data_dir
        self.use_cache = use_cache
        os.makedirs(data_dir, exist_ok=True)
    
    def download_mutag(self) -> str:
        """Download MUTAG dataset"""
        return self._download_dataset('MUTAG')
    
    def _download_dataset(self, name: str) -> str:
        """Download dataset from TU Dortmund"""
        if name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        
        zip_path = os.path.join(self.data_dir, f"{name}.zip")
        extract_path = os.path.join(self.data_dir, name)
        cache_file = os.path.join(self.data_dir, f"{name}_cache.pkl")
        
        # Check cache first
        if self.use_cache and os.path.exists(cache_file):
            print(f"✓ Found cached {name}")
            return extract_path
        
        # Check if already extracted
        if os.path.exists(extract_path):
            print(f"✓ {name} already downloaded")
            return extract_path
        
        # Download
        print(f"Downloading {name}...")
        url = self.DATASETS[name]['url']
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='')
            
            print("\n✓ Download complete")
            
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            raise
        
        # Extract
        print(f"Extracting {name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print(f"✓ Extracted to {extract_path}")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            raise
        
        return extract_path
    
    def load_mutag(self, subset_size: int = None) -> List[Graph]:
        """
        Load MUTAG dataset
        
        Args:
            subset_size: Number of graphs to load (None = all)
        
        Returns:
            List of Graph objects
        """
        return self.load_graphs('MUTAG', subset_size)
    
    def load_graphs(self, dataset_name: str, subset_size: int = None) -> List[Graph]:
        """Load graphs from dataset"""
        
        cache_file = os.path.join(self.data_dir, f"{dataset_name}_cache.pkl")
        
        # Try cache
        if self.use_cache and os.path.exists(cache_file):
            print(f"Loading {dataset_name} from cache...")
            with open(cache_file, 'rb') as f:
                graphs = pickle.load(f)
            print(f"✓ Loaded {len(graphs)} graphs from cache")
        else:
            # Download if needed
            dataset_path = self._download_dataset(dataset_name)
            base_path = os.path.join(dataset_path, dataset_name)
            
            print(f"Parsing {dataset_name}...")
            graphs = self._parse_dataset(base_path, dataset_name)
            
            # Save cache
            print(f"Caching {dataset_name}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(graphs, f)
            print(f"✓ Saved cache")
        
        # Apply subset
        if subset_size is not None and subset_size < len(graphs):
            graphs = graphs[:subset_size]
            print(f"Using subset of {subset_size} graphs")
        
        # Print statistics
        self._print_statistics(dataset_name, graphs)
        
        return graphs
    
    def _parse_dataset(self, base_path: str, dataset_name: str) -> List[Graph]:
        """Parse TU Dortmund format dataset"""
        
        # Read graph indicator
        with open(f"{base_path}_graph_indicator.txt", 'r') as f:
            graph_indicator = [int(line.strip()) for line in f]
        
        # Read node labels
        with open(f"{base_path}_node_labels.txt", 'r') as f:
            node_labels = [int(line.strip()) for line in f]
        
        # Read edges
        edges = []
        with open(f"{base_path}_A.txt", 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                edges.append((int(parts[0]), int(parts[1])))
        
        # Read edge labels (if exists)
        edge_labels = {}
        edge_label_file = f"{base_path}_edge_labels.txt"
        if os.path.exists(edge_label_file):
            with open(edge_label_file, 'r') as f:
                for i, line in enumerate(f):
                    edge_labels[i] = int(line.strip())
        
        # Read graph labels (if exists)
        graph_labels = {}
        graph_label_file = f"{base_path}_graph_labels.txt"
        if os.path.exists(graph_label_file):
            with open(graph_label_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    graph_labels[i] = int(line.strip())
        
        # Build graphs
        graphs_dict = defaultdict(lambda: Graph())
        
        # Add vertices
        for node_id, (graph_id, node_label) in enumerate(zip(graph_indicator, node_labels), 1):
            graph = graphs_dict[graph_id]
            graph.gid = graph_id
            graph.add_vertex(node_id, node_label)
            if graph_id in graph_labels:
                graph.label = graph_labels[graph_id]
        
        # Add edges
        for edge_id, (frm, to) in enumerate(edges):
            graph_id = graph_indicator[frm - 1]
            elabel = edge_labels.get(edge_id, 0)
            graphs_dict[graph_id].add_edge(frm, to, elabel)
        
        graphs = list(graphs_dict.values())
        print(f"✓ Parsed {len(graphs)} graphs")
        
        return graphs
    
    def _print_statistics(self, dataset_name: str, graphs: List[Graph]):
        """Print dataset statistics"""
        
        num_graphs = len(graphs)
        nodes_per_graph = [len(g.vertices) for g in graphs]
        edges_per_graph = [len(g.edges) for g in graphs]
        
        all_vertex_labels = set()
        all_edge_labels = set()
        class_distribution = defaultdict(int)
        
        for g in graphs:
            all_vertex_labels.update(v.label for v in g.vertices)
            all_edge_labels.update(e.elabel for e in g.edges)
            if g.label is not None:
                class_distribution[g.label] += 1
        
        print(f"\n{'='*60}")
        print(f"{dataset_name} Dataset Statistics")
        print(f"{'='*60}")
        print(f"Graphs: {num_graphs}")
        print(f"Nodes per graph: {np.mean(nodes_per_graph):.1f} ± {np.std(nodes_per_graph):.1f}")
        print(f"  Min/Max: {min(nodes_per_graph)} / {max(nodes_per_graph)}")
        print(f"Edges per graph: {np.mean(edges_per_graph):.1f} ± {np.std(edges_per_graph):.1f}")
        print(f"  Min/Max: {min(edges_per_graph)} / {max(edges_per_graph)}")
        print(f"Unique vertex labels: {len(all_vertex_labels)}")
        print(f"Unique edge labels: {len(all_edge_labels)}")
        
        if class_distribution:
            print(f"Class distribution:")
            for label, count in sorted(class_distribution.items()):
                print(f"  Class {label}: {count} graphs ({count/num_graphs*100:.1f}%)")
        
        print(f"{'='*60}\n")

# Test
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Load full MUTAG
    print("Loading MUTAG dataset...")
    graphs = loader.load_mutag()
    
    # Show sample graph
    print(f"\nSample graph (Graph {graphs[0].gid}):")
    print(f"  Vertices: {len(graphs[0].vertices)}")
    print(f"  Edges: {len(graphs[0].edges)}")
    print(f"  Vertex labels: {graphs[0].get_vertex_labels()}")
    print(f"  Class: {graphs[0].label}")
