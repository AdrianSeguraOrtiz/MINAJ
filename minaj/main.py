import json
import multiprocessing
import random
import string
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import graphistry
import networkx as nx
import numpy as np
import pandas as pd
from geneci.main import SimpleConsensusCriteria, Technique, infer_network
from geneci.utils import (cpus_dict, get_expression_data_from_module,
                          simple_consensus)
from rich import print

from minaj.utils import (ClusteringAlgorithm, palette_color_for,
                         read_igraph_from_csv, split_communities_recursively,
                         write_communities, write_partition)


def main_modular_inference(
    expression_data: Path,
    global_techniques: Optional[List[Technique]],
    modular_techniques: Optional[List[Technique]],
    consensus_criteria: SimpleConsensusCriteria = SimpleConsensusCriteria.MeanWeights,
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.Leiden,
    preferred_size: int = 100,
    threads: int = multiprocessing.cpu_count(),
    output_dir: Path = Path("./modular_inference"),
):
    """
    Apply modular inference to a gene network
    """

    # Report information to the user
    print(
        f"Applying modular inference to the gene network {expression_data} with the following global techniques: {global_techniques} and the following modular techniques: {modular_techniques}"
    )

    # Set temporal folder
    temp_folder_str = "tmp-" + "".join(random.choices(string.ascii_lowercase, k=10))

    # Set time file and global start time
    time_file = Path(f"{output_dir}/times.txt")
    time_file.parent.mkdir(exist_ok=True, parents=True)
    open(time_file, "w").write(f"Times of modular inference for {expression_data}:\n")
    global_start_time = time.time()

    # 1. Global inference
    ## Start time
    step1_start_time = time.time()

    ## Create global inference folder
    global_inference_folder = Path(f"{output_dir}/global_inference/")
    global_inference_folder.mkdir(exist_ok=True, parents=True)
    ## Carry out the global inference of the network using the specified light techniques.
    infer_network(
        expression_data=expression_data,
        technique=global_techniques,
        threads=threads,
        str_threads=None,
        temp_folder_str=temp_folder_str,
        output_dir=global_inference_folder,
    )
    ## Extract results
    global_confidence_list = list(
        Path(f"./{global_inference_folder}/{expression_data.stem}/lists/").glob(
            "GRN_*.csv"
        )
    )

    ## Report completion and register time
    print("Step 1/7: Global inference completed.")
    step1_end_time = time.time()
    step1_elapsed = step1_end_time - step1_start_time
    open(time_file, "a").write(
        f"\t - Step 1 -> Global inference: {step1_elapsed:.2f} seconds.\n"
    )

    # 2. Simple consensus of global network
    ## Start time
    step2_start_time = time.time()

    ## Simple consensus
    global_network_file = f"{global_inference_folder}/consensus_global_network.csv"
    simple_consensus(global_confidence_list, consensus_criteria, global_network_file)

    ## Report completion and register time
    print("Step 2/7: Simple consensus of global network completed.")
    step2_end_time = time.time()
    step2_elapsed = step2_end_time - step2_start_time
    open(time_file, "a").write(
        f"\t - Step 2 -> Simple consensus of global network: {step2_elapsed:.2f} seconds.\n"
    )

    # 3. Extract modules from global consensus network
    ## Start time
    step3_start_time = time.time()

    ## Creates modules folder
    modules_folder = Path(f"{output_dir}/modules/")
    modules_network_folder = Path(f"{modules_folder}/networks/")
    modules_network_folder.mkdir(exist_ok=True, parents=True)
    ## Extract modules
    main_cluster_network(
        confidence_list=global_network_file,
        algorithm=algorithm,
        preferred_size=preferred_size,
        min_size=5,
        output_dir=modules_network_folder,
    )

    ## Report completion and register time
    print("Step 3/7: Extraction of modules completed.")
    step3_end_time = time.time()
    step3_elapsed = step3_end_time - step3_start_time
    open(time_file, "a").write(
        f"\t - Step 3 -> Extraction of modules: {step3_elapsed:.2f} seconds.\n"
    )

    # 4. Obtain subdivisions of expression data based on modules
    ## Start time
    step4_start_time = time.time()

    ## Obtain subdivisions
    modules_files = list(modules_network_folder.glob("community_*.csv"))
    modules_expression_folder = Path(f"{modules_folder}/expression/")
    modules_expression_folder.mkdir(exist_ok=True, parents=True)
    for module_file in modules_files:
        get_expression_data_from_module(
            expression_data,
            module_file,
            f"{modules_expression_folder}/{module_file.stem}_expression.csv",
        )

    ## Report completion and register time
    print("Step 4/7: Expression data from modules extracted.")
    step4_end_time = time.time()
    step4_elapsed = step4_end_time - step4_start_time
    open(time_file, "a").write(
        f"\t - Step 4 -> Expression data from modules extracted: {step4_elapsed:.2f} seconds.\n"
    )

    # 5. Modular inference
    ## Start time
    step5_start_time = time.time()
    open(time_file, "a").write(f"\t - Step 5 -> Inference of partial networks:\n")

    ## Create modular inference folder
    modular_inference_folder = Path(f"{output_dir}/modular_inference/")
    modular_inference_folder.mkdir(exist_ok=True, parents=True)

    ## Carry out the modular inference of the network using the specified techniques.

    ### Get all expression files generated per module
    expression_files = list(modules_expression_folder.glob("*.csv"))

    ### Build list of tasks: one per (expression_file, technique) pair
    tasks = [
        (expr_file, technique)
        for expr_file in expression_files
        for technique in modular_techniques
    ]

    ### Sort tasks in descending order of required threads (to avoid fragmentation)
    tasks.sort(key=lambda t: cpus_dict[t[1]], reverse=True)

    ### Initialize pool of thread IDs and list of active futures
    all_thread_ids = list(range(threads))
    free_threads = deque(all_thread_ids)
    active_futures = []

    ### Submit tasks asynchronously while respecting thread ID availability
    with ProcessPoolExecutor(max_workers=threads) as executor:
        while tasks or active_futures:
            # Try to launch any task for which enough thread IDs are available
            for i, (exp_file, technique) in enumerate(tasks):
                n_needed = cpus_dict[technique]
                if len(free_threads) >= n_needed:
                    # Reserve the required thread IDs for the task
                    thread_ids = [free_threads.popleft() for _ in range(n_needed)]
                    str_threads = ",".join(str(t) for t in thread_ids)

                    # Generate a temporary folder name for Docker volume isolation
                    tmp_folder = "tmp-" + "".join(
                        random.choices(string.ascii_lowercase, k=10)
                    )

                    # Launch the inference task as a separate process
                    future = executor.submit(
                        infer_network,
                        exp_file,
                        [technique],
                        None,
                        str_threads,
                        tmp_folder,
                        modular_inference_folder,
                    )

                    # Track the active task and its associated resources
                    active_futures.append(
                        (future, thread_ids, exp_file, technique, time.time())
                    )
                    del tasks[i]
                    break
            else:
                # No task could be launched due to thread limits → wait before retrying
                time.sleep(1)

            # Check for completed tasks
            for (
                fut,
                thread_ids,
                exp_file,
                technique,
                start_time,
            ) in active_futures.copy():
                if fut.done():
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Error in {technique.value} on {exp_file.name}: {e}")
                    else:
                        expected_file = f"{modular_inference_folder}/{Path(exp_file.name).stem}/lists/GRN_{technique.value}.csv"
                        if not Path(expected_file).is_file():
                            # If the file has not been generated, keep it in the task tail
                            print(
                                f"Expected file {expected_file} not found after {technique.value} on {exp_file.name}. Retrying..."
                            )
                            tasks.append((exp_file, technique))
                        else:
                            # Log elapsed time of completed task
                            elapsed = time.time() - start_time
                            open(time_file, "a").write(
                                f"\t\t - {technique.value} on {exp_file.name}: {elapsed:.2f} seconds.\n"
                            )
                            # Standardize the results between the minimum and the maximum of the community in the initial global consensus network
                            ## Read initial community network to get min and max weights
                            initial_community_network = f"{modules_network_folder}/{exp_file.name.removesuffix('_expression.csv')}.csv"
                            df_initial_community = pd.read_csv(
                                initial_community_network,
                                header=None,
                                names=["source", "target", "weight"],
                            )
                            initial_community_max = df_initial_community["weight"].max()
                            initial_community_min = df_initial_community["weight"].min()
                            ## Read refined community network and standardize weights
                            df_refined_community = pd.read_csv(
                                expected_file,
                                header=None,
                                names=["source", "target", "weight"],
                            )
                            refined_max = df_refined_community["weight"].max()
                            refined_min = df_refined_community["weight"].min()
                            if refined_max != refined_min:
                                df_refined_community[
                                    "weight"
                                ] = initial_community_min + (
                                    df_refined_community["weight"] - refined_min
                                ) / (
                                    refined_max - refined_min
                                ) * (
                                    initial_community_max - initial_community_min
                                )
                            else:
                                # If all values ​​are the same, assign the average of the original range
                                df_refined_community["weight"] = (
                                    initial_community_min + initial_community_max
                                ) / 2

                            ## Save refined community network
                            standarized_file = f"{modular_inference_folder}/{Path(exp_file.name).stem}/standarized_lists/GRN_{technique.value}.csv"
                            Path(standarized_file).parent.mkdir(
                                exist_ok=True, parents=True
                            )
                            df_refined_community.to_csv(
                                standarized_file, index=False, header=False
                            )

                        # Remove task from active list and release its thread IDs
                        active_futures.remove(
                            (fut, thread_ids, exp_file, technique, start_time)
                        )
                        free_threads.extend(thread_ids)

    ## Report completion and register time
    print("Step 5/7: Modular inference completed.")
    step5_end_time = time.time()
    step5_elapsed = step5_end_time - step5_start_time
    open(time_file, "a").write(f"\t\t - Total: {step5_elapsed:.2f} seconds.\n")

    # 6. Simple consensus of each subnetwork
    ## Start time
    step6_start_time = time.time()

    ## Simple consensus
    inferred_modules_folder = list(modular_inference_folder.glob("*"))
    for inferred_module_folder in inferred_modules_folder:
        modular_confidence_list = list(
            inferred_module_folder.glob("standarized_lists/GRN_*.csv")
        )
        modular_network_file = f"{inferred_module_folder}/consensus_modular_network.csv"
        simple_consensus(
            modular_confidence_list,
            consensus_criteria,
            modular_network_file,
        )

    ## Report completion and register time
    print("Step 6/7: Simple consensus of each subnetwork completed.")
    step6_end_time = time.time()
    step6_elapsed = step6_end_time - step6_start_time
    open(time_file, "a").write(
        f"\t - Step 6 -> Simple consensus of each subnetwork: {step6_elapsed:.2f} seconds.\n"
    )

    # 7. Merge all subnetworks and intermediate_relations into a single file
    ## Start time
    step7_start_time = time.time()

    ## Merge subnetworks and intermediate relations
    inferred_consensus_modules = list(
        modular_inference_folder.glob("*/consensus_modular_network.csv")
    )
    intermediate_relations = f"{modules_network_folder}/intermediate_relations.csv"
    files_to_merge = inferred_consensus_modules + [intermediate_relations]
    merged_network_file = f"{output_dir}/merged_network.csv"
    dfs = [pd.read_csv(fp, header=None) for fp in files_to_merge]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(merged_network_file, index=False, header=False)

    ## Report completion and register time
    print("Step 7/7: Merging of subnetworks and intermediate relations completed.")
    step7_end_time = time.time()
    step7_elapsed = step7_end_time - step7_start_time
    open(time_file, "a").write(
        f"\t - Step 7 -> Merging of subnetworks and intermediate relations: {step7_elapsed:.2f} seconds.\n"
    )

    # Register total time
    global_end_time = time.time()
    global_elapsed = global_end_time - global_start_time
    open(time_file, "a").write(f"Total time: {global_elapsed:.2f} seconds.\n")


def main_cluster_network(
    confidence_list: str,
    algorithm: ClusteringAlgorithm,
    preferred_size: int,
    min_size: int,
    output_dir: str,
):
    """
    Main pipeline to read a network, apply community detection, and save results.
    """
    g = read_igraph_from_csv(confidence_list)
    partition = split_communities_recursively(
        g, preferred_size, min_size, algorithm=algorithm
    )
    write_partition(partition, output_dir)
    write_communities(g, partition, output_dir)
    
    
def main_calculate_community_metrics(confidence_list: str, partition_file: str) -> dict:
    """
    Calculate metrics for each community in the graph.
    
    Args:
        confidence_list (str): Path to the CSV file containing the confidence list.
        partition_file (str): Path to the JSON file containing the community partition.
    
    Returns:
        dict: Dictionary with community ID as key and metrics as value.
    """
    # Load the confidence list and partition data
    edges = pd.read_csv(confidence_list, header=None, names=["source", "destination", "confidence"])
    partition_data = json.load(open(partition_file))

    # Create a directed graph from the edges
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["source"], row["destination"], weight=row["confidence"])
        
    metrics = {}
    
    for comm_id, nodes in partition_data.items():
        # Create subgraph for the community
        subgraph = G.subgraph(nodes).copy()
        
        # Size: Number of nodes
        size = len(nodes)
        
        # Density: Ratio of actual edges to possible edges
        num_edges = subgraph.number_of_edges()
        possible_edges = size * (size - 1)  # Directed graph: n * (n-1)
        density = num_edges / possible_edges if possible_edges > 0 else 0.0
        
        # Diameter: Longest shortest path (only for weakly connected components)
        try:
            if nx.is_weakly_connected(subgraph):
                diameter = nx.diameter(subgraph)
            else:
                # For disconnected graphs, compute diameter of largest weakly connected component
                largest_cc = max(nx.weakly_connected_components(subgraph), key=len)
                diameter = nx.diameter(subgraph.subgraph(largest_cc))
        except (nx.NetworkXError, ValueError):
            diameter = float('inf')  # If no valid path exists
        
        # Isolation: Proportion of edges going outside the community
        external_edges = sum(1 for u, v in G.edges() if (u in nodes and v not in nodes) or (v in nodes and u not in nodes))
        total_edges = sum(1 for u, v in G.edges() if u in nodes or v in nodes)
        isolation = external_edges / total_edges if total_edges > 0 else 0.0
        
        # Degrees: In-degree + Out-degree for each node
        degrees = [subgraph.in_degree(n) + subgraph.out_degree(n) for n in nodes]
        mean_degree = np.mean(degrees) if degrees else 0.0
        median_degree = np.median(degrees) if degrees else 0.0
        
        # Weights: Edge weights within the community
        weights = [d['weight'] for _, _, d in subgraph.edges(data=True)]
        mean_weight = np.mean(weights) if weights else 0.0
        median_weight = np.median(weights) if weights else 0.0
        
        metrics[comm_id] = {
            'size': size,
            'density': density,
            'diameter': diameter,
            'isolation': isolation,
            'mean_degree': mean_degree,
            'median_degree': median_degree,
            'mean_weight': mean_weight,
            'median_weight': median_weight
        }
        
        # Print metrics for user feedback
        print(f"Community {comm_id}: Size={size}, Density={density:.3f}, "
              f"Diameter={diameter}, Isolation={isolation:.3f}, "
              f"Mean Degree={mean_degree:.2f}, Median Degree={median_degree:.2f}, "
              f"Mean Weight={mean_weight:.2f}, Median Weight={median_weight:.2f}")  
    
    return metrics


def main_plot_communities(
    complete_network: Path,
    partition: Path,
    username: str,
    password: str,
    weight_threshold: float = 0.1,
    include_community_metrics: bool = True,
):
    """
    Visualize a directed weighted graph with community coloring using Graphistry.

    Args:
        complete_network (Path): Path to the network CSV file.
        partition (Path): Path to the partition JSON file.
        username (str): Graphistry username.
        password (str): Graphistry password.
        weight_threshold (float, optional): Minimum edge weight to include in visualization. Default is 0.1.
            Adjust this value to filter out weak edges and reduce visual clutter.
    """

    # 1. Register user in Graphistry
    graphistry.register(api=3, username=username, password=password)

    # 2. Load network
    edges = pd.read_csv(complete_network, header=None, names=["source", "destination", "confidence"])
    edges["linkname"] = edges["source"] + "-->" + edges["destination"]
    edges = edges[edges["confidence"] > weight_threshold]

    # 3. Get set of nodes
    nodes_set = set(edges["source"]).union(edges["destination"])
    nodes = pd.DataFrame({"genename": list(nodes_set)})

    # 4. Load communities from JSON
    with open(partition) as f:
        partition_data = json.load(f)

    # 5. Invert communities to get node → community
    inverted_partition = {node: None for node in nodes["genename"]}
    for comm_id_str, gene_list in partition_data.items():
        for gene in gene_list:
            inverted_partition[gene] = comm_id_str

    # 7. Add community and community metrics to nodes
    nodes["community"] = nodes["genename"].map(inverted_partition)
    if include_community_metrics:
        community_metrics = main_calculate_community_metrics(complete_network, partition)
        nodes["community_metrics"] = nodes["community"].apply(
            lambda comm_id: (
                f"Community: {comm_id}<br>"
                f"Size: {community_metrics[comm_id]['size']}<br>"
                f"Density: {community_metrics[comm_id]['density']:.3f}<br>"
                f"Diameter: {community_metrics[comm_id]['diameter']}<br>"
                f"Isolation: {community_metrics[comm_id]['isolation']:.3f}<br>"
                f"Mean Degree: {community_metrics[comm_id]['mean_degree']:.2f}<br>"
                f"Median Degree: {community_metrics[comm_id]['median_degree']:.2f}<br>"
                f"Mean Weight: {community_metrics[comm_id]['mean_weight']:.2f}<br>"
                f"Median Weight: {community_metrics[comm_id]['median_weight']:.2f}"
            ) if comm_id else "None"
        )
    else:
        nodes["community_metrics"] = "No metrics available"

    # 9. Visualize in Graphistry
    plot = (
        graphistry
        .nodes(nodes, node="genename")
        .edges(edges, source="source", destination="destination", edge="linkname")
        .bind(edge_label="linkname", edge_weight="confidence", point_color="community_metrics")
        .encode_point_color(
            'community',
            categorical_mapping={str(c): palette_color_for(c) for c in nodes['community'].unique()},
            default_mapping="#000000",
            as_categorical=True
        )
        .plot()
    )

    print(f"Visualization Loaded: {plot}\n")