import csv
import json
from collections import defaultdict
from enum import Enum
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering


class ClusteringAlgorithm(str, Enum):
    Leiden = "Leiden"
    Spectral = "Spectral"


def read_igraph_from_csv(file_path):
    """
    Reads a directed and weighted graph from a CSV file.
    Each row in the file should have the format: source,target,weight

    Parameters:
        file_path (str): Path to the input CSV file.

    Returns:
        igraph.Graph: A directed igraph graph with weights.
    """
    edges = []
    weights = []
    nodes_set = set()

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for source, target, weight in reader:
            edges.append((source, target))
            weights.append(float(weight))
            nodes_set.update([source, target])

    nodes = sorted(nodes_set)
    node_indices = {name: idx for idx, name in enumerate(nodes)}
    edge_indices = [(node_indices[s], node_indices[t]) for s, t in edges]

    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes
    g.add_edges(edge_indices)
    g.es["weight"] = weights

    return g


def apply_leiden(g):
    """
    Applies the Leiden community detection algorithm using modularity for directed graphs.

    Parameters:
        g (igraph.Graph): A directed graph with weights.

    Returns:
        VertexClustering: The partition of the graph into communities.
    """
    return leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight"
    )


def apply_spectral(g, n_clusters):
    """
    Applies Spectral Clustering to the graph using the weighted adjacency matrix.

    Parameters:
        g (igraph.Graph): A directed graph with weights.
        n_clusters (int): Number of clusters to form.

    Returns:
        list[list[int]]: List of communities, each a list of vertex indices.
    """
    adj = np.array(g.get_adjacency(attribute="weight").data)
    clustering = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans"
    ).fit(adj)

    labels = clustering.labels_
    communities = {}
    for idx, label in enumerate(labels):
        communities.setdefault(label, []).append(idx)

    return list(communities.values())


def apply_clustering(g, algorithm: ClusteringAlgorithm, preferred_size: int):
    """
    Dispatches the appropriate clustering algorithm based on user selection.

    Parameters:
        g (igraph.Graph): A directed graph.
        algorithm (ClusteringAlgorithm): The community detection algorithm to apply.
        preferred_size (int): Preferred size for the communities.

    Returns:
        VertexClustering or list[list[int]]: Partition of the graph.
    """
    if algorithm == ClusteringAlgorithm.Leiden:
        return apply_leiden(g)
    elif algorithm == ClusteringAlgorithm.Spectral:
        return apply_spectral(g, n_clusters=max(1, g.vcount() // preferred_size))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def split_communities_recursively(
    g,
    preferred_size,
    min_size,
    max_depth=10,
    depth=0,
    algorithm=ClusteringAlgorithm.Leiden,
):
    """
    Recursively splits oversized communities using the selected algorithm until preferred size is met.

    Parameters:
        g (igraph.Graph): The graph to be partitioned.
        preferred_size (int): Maximum size of each community.
        min_size (int): Minimum size to consider a valid community.
        max_depth (int): Maximum recursion depth.
        depth (int): Current recursion depth.
        algorithm (ClusteringAlgorithm): Clustering algorithm to apply.

    Returns:
        dict: Mapping from node name to community ID.
    """
    partition = apply_clustering(g, algorithm, preferred_size)
    final_partition = {}
    current_label = 0

    for community in partition:
        if len(community) > preferred_size and depth < max_depth:
            subgraph = g.subgraph(community)
            sub_partition = split_communities_recursively(
                subgraph, preferred_size, min_size, max_depth, depth + 1, algorithm
            )
            for name, label in sub_partition.items():
                final_partition[name] = current_label + label
            current_label += max(sub_partition.values()) + 1
        elif len(community) >= min_size:
            for idx in community:
                name = g.vs[idx]["name"]
                final_partition[name] = current_label
            current_label += 1

    return final_partition


def write_partition(partition, output_dir):
    """
    Writes the community partition to a JSON file in inverted form:
    from {node: community} to {community: [nodes]}.

    Parameters:
        partition (dict): Mapping from node name to community ID.
        output_dir (str): Directory where the partition file will be saved.
    """
    # Invertimos el diccionario
    inverted = defaultdict(list)
    for node, community in partition.items():
        inverted[community].append(node)

    # Convertimos defaultdict a dict normal para el JSON
    inverted = dict(inverted)

    # Nos aseguramos de que el directorio existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guardamos en JSON
    with open(f"{output_dir}/partition.json", "w") as f:
        json.dump(inverted, f, indent=4)


def write_communities(g, partition, output_folder):
    """
    Writes community assignments to individual CSV files and saves inter-community edges.

    Parameters:
        g (igraph.Graph): The original graph.
        partition (dict): Mapping from node name to community ID.
        output_folder (str): Directory where output files will be saved.
    """
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Save internal community edges
    for cid, nodes in communities.items():
        with open(f"{output_folder}/community_{cid}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for edge in g.es:
                s = g.vs[edge.source]["name"]
                t = g.vs[edge.target]["name"]
                if s in nodes and t in nodes:
                    writer.writerow([s, t, edge["weight"]])

    # Save inter-community edges
    with open(f"{output_folder}/intermediate_relations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for edge in g.es:
            s = g.vs[edge.source]["name"]
            t = g.vs[edge.target]["name"]
            if partition.get(s) != partition.get(t):
                writer.writerow([s, t, edge["weight"]])

def palette_color_for(c, palette="tab20", default_color="#000000"):
    """
    Returns a color from a given palette corresponding to an integer ID.
    
    Parameters:
        c (int): The ID of the node.
        palette (str): The name of the color palette.
        default_color (str): The default color to return if ID is None.
    
    Returns:
        str: The color in hexadecimal format.
    """
    # If the ID is None, return the default color
    if c is None:
        return default_color
    else:
        # Get the list of colors from the palette
        cmap = plt.get_cmap(palette)
        n_colors = cmap.N
        # Convert the ID to an index within the palette
        idx = int(c) % n_colors
        rgb = cmap(idx)[:3]
        # Convert to hexadecimal format
        return '#%02x%02x%02x' % tuple(int(255*x) for x in rgb)