#!/usr/bin/env python3
"""
Visualize the crawled website structure as an interactive graph.
Reads metadata.json from CrawlerUDC and generates a network visualization.
"""

import json
import argparse
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Color scheme matching your crawler
COLORS = {
    'pdf': '#b3b3b3',      # (179,179,179) - gray
    'html': '#b43b86',     # (180,59,134) - magenta
    'image': '#00ffff',    # cyan
    'root': '#ff6b6b'      # red for starting point
}

def load_metadata(state_dir="crawl"):
    """Load metadata.json - this is the main data source"""
    metadata_path = Path(state_dir) / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {state_dir}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata

def categorize_file(url, file_metadata):
    """Determine node type based on metadata"""
    if 'original_format' in file_metadata:
        format_type = file_metadata['original_format'].lower()
        
        if format_type == 'pdf':
            return 'pdf'
        elif format_type in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg', 'image']:
            return 'image'
        elif format_type in ['html', 'htm']:
            return 'html'
    
    # Fallback to URL-based detection
    url_lower = url.lower()
    if '.pdf' in url_lower:
        return 'pdf'
    elif any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
        return 'image'
    
    return 'html'

def build_graph_from_metadata(metadata, base_url=None):
    """
    Build graph structure directly from metadata.json.
    Each entry is a node, discovered_from creates edges.
    """
    G = nx.DiGraph()
    
    print(f"Building graph from {len(metadata)} files in metadata")
    
    # Add all nodes
    for url, data in metadata.items():
        node_type = categorize_file(url, data)
        G.add_node(url, 
                   type=node_type,
                   format=data.get('original_format', 'unknown'),
                   last_crawl=data.get('last_crawl', 'unknown'))
    
    # Build edges from discovered_from relationships
    edges_added = 0
    orphan_nodes = []  # Nodes with no parent
    
    for url, data in metadata.items():
        if 'discovered_from' in data and data['discovered_from']:
            parent_url = data['discovered_from']
            
            # Only add edge if parent exists in our graph
            if parent_url in metadata:
                G.add_edge(parent_url, url)
                edges_added += 1
            else:
                # Parent not in metadata (external link or root page)
                orphan_nodes.append((url, parent_url))
        else:
            # No discovered_from field - this is likely a root/entry point
            orphan_nodes.append((url, None))
    
    print(f"Added {edges_added} edges from discovered_from relationships")
    
    if orphan_nodes:
        print(f"\nFound {len(orphan_nodes)} nodes without parents in the graph:")
        for url, parent in orphan_nodes[:5]:  # Show first 5
            if parent:
                print(f"  - {shorten_url(url)} (parent not downloaded: {shorten_url(parent)})")
            else:
                print(f"  - {shorten_url(url)} (no parent - likely entry point)")
        if len(orphan_nodes) > 5:
            print(f"  ... and {len(orphan_nodes) - 5} more")
    
    # Mark base_url as root if provided and exists
    if base_url and base_url in metadata:
        G.nodes[base_url]['type'] = 'root'
        print(f"\nMarked '{base_url}' as root node")
    elif base_url:
        print(f"\nWarning: base_url '{base_url}' not found in metadata")
    else:
        # Try to identify root nodes (nodes with no parent)
        root_candidates = [url for url, parent in orphan_nodes if parent is None]
        if root_candidates:
            # Mark the first one as root
            G.nodes[root_candidates[0]]['type'] = 'root'
            print(f"\nAuto-detected root node: {root_candidates[0]}")
    
    return G

def shorten_url(url, max_len=50):
    """Shorten URL for display"""
    if not url:
        return "None"
    
    parsed = urlparse(url)
    
    # Try to show domain + path
    domain = parsed.netloc
    path = parsed.path or '/'
    
    # Remove common prefixes
    if domain.startswith('www.'):
        domain = domain[4:]
    
    full = f"{domain}{path}"
    
    if len(full) > max_len:
        # Show domain + truncated path
        remaining = max_len - len(domain) - 3
        if remaining > 10:
            return f"{domain}...{path[-remaining:]}"
        else:
            return f"...{path[-max_len:]}"
    
    return full

def visualize_graph(G, output_file="crawler_graph.png", layout="spring", silent=False):
    """Create visualization of the crawl graph"""
    
    if len(G.nodes()) == 0:
        print("No nodes to visualize!")
        return
    
    # Prepare node colors based on type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'html')
        node_colors.append(COLORS.get(node_type, COLORS['html']))
        
        # Make root nodes bigger
        if node_type == 'root':
            node_sizes.append(600)
        else:
            node_sizes.append(300)
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "hierarchical":
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            print("Hierarchical layout failed, falling back to spring")
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#666666',
                          arrows=True,
                          arrowsize=15,
                          arrowstyle='->',
                          width=1.5,
                          alpha=0.6,
                          ax=ax,
                          connectionstyle='arc3,rad=0.1')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          ax=ax)
    
    # Draw labels only if not silent
    if not silent:
        labels = {node: shorten_url(node, max_len=40) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos,
                               labels=labels,
                               font_size=7,
                               font_weight='bold',
                               font_family='sans-serif',
                               ax=ax)
    
    # Create legend
    legend_elements = [
        Patch(facecolor=COLORS['root'], label='Root/Entry Point'),
        Patch(facecolor=COLORS['html'], label='HTML Pages'),
        Patch(facecolor=COLORS['pdf'], label='PDF Documents'),
        Patch(facecolor=COLORS['image'], label='Images')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title('Website Crawl Graph', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to: {output_file}")
    
    # Print statistics
    print(f"\n=== Graph Statistics ===")
    print(f"Total nodes (files): {G.number_of_nodes()}")
    print(f"Total edges (links): {G.number_of_edges()}")
    
    # Count by type
    node_types = defaultdict(int)
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'html')
        node_types[node_type] += 1
    
    print(f"\nBy file type:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type.upper()}: {count}")
    
    # Graph connectivity info
    if G.number_of_nodes() > 0:
        print(f"\nConnectivity:")
        print(f"  Connected: {nx.is_weakly_connected(G)}")
        print(f"  Components: {nx.number_weakly_connected_components(G)}")
        
        # Find nodes with most children
        out_degrees = dict(G.out_degree())
        if out_degrees:
            max_out = max(out_degrees.values())
            if max_out > 0:
                hub_nodes = [n for n, d in out_degrees.items() if d == max_out]
                print(f"\nHub node(s) with most links ({max_out} outgoing):")
                for hub in hub_nodes[:3]:
                    print(f"  - {shorten_url(hub)}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize crawler graph structure from metadata.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_graph.py
  python visualize_graph.py --state-dir /path/to/crawl
  python visualize_graph.py --layout hierarchical --output graph.png
  python visualize_graph.py --base-url https://www.udc.es/es/
  python visualize_graph.py -s  # Silent mode for large graphs
        """
    )
    parser.add_argument('--state-dir', default='../../crawl', 
                       help='Crawler state directory containing metadata.json')
    parser.add_argument('--output', default='memoria/imaxes/grafo.png', 
                       help='Output image file path')
    parser.add_argument('--layout', choices=['spring', 'hierarchical', 'circular'], 
                       default='spring', 
                       help='Graph layout algorithm (default: spring)')
    parser.add_argument('--base-url', 
                       help='Base URL to mark as root node (optional)')
    parser.add_argument('-s', '--silent', action='store_true',
                       help='Silent mode - no labels on nodes (better for large graphs)')
    
    args = parser.parse_args()
    
    # Resolve state_dir relative to script location
    script_dir = Path(__file__).parent
    state_path = (script_dir / args.state_dir).resolve()
    
    if not state_path.exists():
        print(f"Error: State directory not found: {state_path}")
        return 1
    
    print(f"Loading metadata from: {state_path}")
    print("=" * 60)
    
    try:
        metadata = load_metadata(str(state_path))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Found {len(metadata)} files in metadata.json\n")
    
    # Build graph
    print("Building graph structure...")
    G = build_graph_from_metadata(metadata, args.base_url)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    print(f"\nGenerating visualization with '{args.layout}' layout...")
    visualize_graph(G, args.output, args.layout, args.silent)
    
    return 0

if __name__ == "__main__":
    exit(main())
