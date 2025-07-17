import multiprocessing
import random
import string
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import pandas as pd
from geneci.main import SimpleConsensusCriteria, Technique, infer_network
from geneci.utils import (cpus_dict, get_expression_data_from_module,
                          simple_consensus)
from rich import print

from minaj.utils import (ClusteringAlgorithm, read_igraph_from_csv,
                         split_communities_recursively, write_communities)


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
    partition = split_communities_recursively(g, preferred_size, min_size, algorithm=algorithm)
    print(f"Original network of {g.vcount()} nodes was clustered into {len(set(partition.values()))} communities:")
    for cid, nodes in pd.Series(partition).value_counts().items():
        print(f"\t - Community {cid}: {nodes} nodes")
    write_communities(g, partition, output_dir)
