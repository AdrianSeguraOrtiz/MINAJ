import glob
import multiprocessing
import random
import re
import string
import time
from pathlib import Path

import pandas as pd
from geneci.main import (SimpleConsensusCriteria, Technique,
                         generic_list_of_links, infer_network)

from minaj.main import main_modular_inference
from minaj.utils import ClusteringAlgorithm


def evaluate_and_append(
    techniques_with_paths, gold_standard_file, group, df_aupr, df_auroc
):
    """
    Evaluate techniques against a gold standard file and append results to DataFrames.

    Args:
        techniques_with_paths (list): List of tuples containing technique name and path to its output file.
        gold_standard_file (Path): Path to the gold standard file.
        group (str): The group name to categorize the results.
        df_aupr (pd.DataFrame): DataFrame to append AUPR results.
        df_auroc (pd.DataFrame): DataFrame to append AUROC results.

    Returns:
        tuple: Updated DataFrames for AUPR and AUROC.
    """
    for tec, path in techniques_with_paths:
        # Evaluate the technique using the generic_list_of_links function
        logs = generic_list_of_links(path, gold_standard_file)

        # Extract AUPR and AUROC values from logs using regex
        aupr = float(re.search(r'AUPR: (.*)"', logs).group(1))
        auroc = float(re.search(r'AUROC: (.*)"', logs).group(1))

        # Append the results to the respective DataFrame
        row_aupr = {"Technique": tec, "AUPR": aupr}
        row_auroc = {"Technique": tec, "AUROC": auroc}
        if group:
            row_aupr["Group"] = group
            row_auroc["Group"] = group
        df_aupr = pd.concat([df_aupr, pd.DataFrame([row_aupr])], ignore_index=True)
        df_auroc = pd.concat([df_auroc, pd.DataFrame([row_auroc])], ignore_index=True)

    return df_aupr, df_auroc


# Establecer carpeta de trabajo y lista de datos de expresion de más de 500 genes
working_dir = "modular_experiments_test/"
expression_data = glob.glob(f"{working_dir}/*/*_exp.csv")
print(f"Expression data files found: {expression_data}")

# Establecer parámetros con valores fijos
global_techniques = [Technique.CLR, Technique.MRNET, Technique.ARACNE, Technique.C3NET]
modular_techniques = [
    Technique.CMI2NI,
    Technique.LOCPCACMI,
    Technique.RSNET,
    Technique.RSNET,
    Technique.PCACMI,
    Technique.GENIE3_ET,
]

# Establecer opciones de parámetros
consensus_options = [
    SimpleConsensusCriteria.RankAverage,
    SimpleConsensusCriteria.MeanWeights,
]
clustering_algorithms = [ca for ca in ClusteringAlgorithm]
preferred_size = [50, 100, 150, 200]
parameter_combinations = [
    (consensus, clustering_algorithm, size)
    for consensus in consensus_options
    for clustering_algorithm in clustering_algorithms
    for size in preferred_size
]

parameter_combinations = [
    (SimpleConsensusCriteria.RankAverage, ClusteringAlgorithm.Leiden, 100),
    (SimpleConsensusCriteria.MeanWeights, ClusteringAlgorithm.Leiden, 100),
    (SimpleConsensusCriteria.RankAverage, ClusteringAlgorithm.Spectral, 100),
    (SimpleConsensusCriteria.MeanWeights, ClusteringAlgorithm.Spectral, 100),
]

# 1. Inferencia modular de todas las redes con todas las opciones de parametros
for expression_file in expression_data:
    for (consensus, clustering_algorithm, size) in parameter_combinations:
        print(
            f"Running modular inference for {expression_file} with consensus {consensus}, algorithm {clustering_algorithm}, and preferred size {size}"
        )
        output_dir = (
            f"{Path(expression_file).parent}/{consensus}_{clustering_algorithm}_{size}"
        )
        main_modular_inference(
            expression_data=Path(expression_file),
            global_techniques=global_techniques,
            modular_techniques=modular_techniques,
            consensus_criteria=consensus,
            algorithm=clustering_algorithm,
            preferred_size=size,
            threads=multiprocessing.cpu_count(),
            output_dir=Path(output_dir),
        )

# 2. Calcular precisión de redes parciales para técnicas ligeras (extracción de la global), técnicas pesadas, consenso de ligeras (consenso de extracciones) y consenso de pesadas
# for expression_file in expression_data:
    for (consensus, clustering_algorithm, size) in parameter_combinations:
        ## Crear carpeta de medidas de comunidades
        experiment_folder = (
            f"{Path(expression_file).parent}/{consensus}_{clustering_algorithm}_{size}"
        )
        community_measurements_folder = f"{experiment_folder}/measurements/communities/"
        Path(community_measurements_folder).mkdir(parents=True, exist_ok=True)

        ## Especificar comunidades generadas durante el experiento
        community_files = glob.glob(
            f"{experiment_folder}/modules/expression/community_*_expression.csv"
        )
        community_names = [Path(f).stem for f in community_files]

        ## Extraer redes parciales de las redes globales
        global_extracted_networks_folder = (
            f"{experiment_folder}/measurements/preprocessed/partial_global_extracted/"
        )
        print(
            f"Extracting partial networks for communities in {global_extracted_networks_folder}"
        )
        for community_file in community_files:
            partial_nets_community_folder = (
                f"{global_extracted_networks_folder}/{Path(community_file).stem}"
            )
            Path(partial_nets_community_folder).mkdir(parents=True, exist_ok=True)
            genes = pd.read_csv(community_file).iloc[:, 0].tolist()
            for tec in global_techniques:
                global_network = f"{experiment_folder}/global_inference/{Path(expression_file).stem}/lists/GRN_{tec}.csv"
                partial_network = pd.read_csv(
                    global_network, header=None, names=["source", "target", "weight"]
                )
                partial_network = partial_network[
                    partial_network["source"].isin(genes)
                    & partial_network["target"].isin(genes)
                ]
                partial_network.to_csv(
                    f"{partial_nets_community_folder}/GRN_{tec}.csv",
                    index=False,
                    header=False,
                )

        ## Extraer gold standard de cada comunidad
        subnetwork_gs_folder = (
            f"{experiment_folder}/measurements/preprocessed/partial_gold_standard/"
        )
        print(f"Extracting gold standard for communities in {subnetwork_gs_folder}")
        gold_standard_file = (
            f"{Path(expression_file).parent}/{Path(expression_file).parent.stem}_gs.csv"
        )
        Path(subnetwork_gs_folder).mkdir(parents=True, exist_ok=True)
        for community_file in community_files:
            genes = pd.read_csv(community_file).iloc[:, 0].tolist()
            gold_standard = pd.read_csv(gold_standard_file, index_col=0)
            gold_standard = gold_standard.loc[genes, genes]
            gold_standard.to_csv(
                f"{subnetwork_gs_folder}/{Path(community_file).stem}_gs.csv",
                index=True,
                header=True,
            )

        ## Calcular precisiones
        df_all_aupr = pd.DataFrame(
            columns=["Community", "Size", "NumGSLinks", "Technique", "Group", "AUPR"]
        )
        df_all_auroc = pd.DataFrame(
            columns=["Community", "Size", "NumGSLinks", "Technique", "Group", "AUROC"]
        )
        for community_name in community_names:
            ## Comprobar si la comunidad tiene suficientes enlaces positivos en el gold standard
            gold_standard_file = f"{subnetwork_gs_folder}/{community_name}_gs.csv"
            gold_standard = pd.read_csv(gold_standard_file, index_col=0)
            num_gs_ones = (gold_standard.values == 1).sum()
            if num_gs_ones <= 3:
                print(
                    f"Skipping community {community_name} in {experiment_folder} due to insufficient positive links in gold standard (only {num_gs_ones} found)"
                )
                continue

            ## Crear DataFrames para almacenar los resultados de AUPR y AUROC
            df_aupr = pd.DataFrame(columns=["Technique", "Group", "AUPR"])
            df_auroc = pd.DataFrame(columns=["Technique", "Group", "AUROC"])

            ## 2.1 Calcular precisión de redes parciales para técnicas ligeras (extraidas de la global)
            print(
                f"Calculating precision for partial networks of light techniques for community {community_name} in {experiment_folder}"
            )
            global_extracted_networks = [
                (
                    tec,
                    f"{global_extracted_networks_folder}/{community_name}/GRN_{tec}.csv",
                )
                for tec in global_techniques
            ]
            df_aupr, df_auroc = evaluate_and_append(
                global_extracted_networks,
                gold_standard_file,
                "Global",
                df_aupr,
                df_auroc,
            )

            ## 2.2 Calcular precisión de redes parciales para técnicas pesadas
            print(
                f"Calculating precision for partial networks of heavy techniques for community {community_name} in {experiment_folder}"
            )
            modular_inferred_networks = [
                (
                    tec,
                    f"{experiment_folder}/modular_inference/{community_name}/standarized_lists/GRN_{tec}.csv",
                )
                for tec in modular_techniques
            ]
            df_aupr, df_auroc = evaluate_and_append(
                modular_inferred_networks,
                gold_standard_file,
                "Modular",
                df_aupr,
                df_auroc,
            )

            ## 2.3 Calcular precisión de consenso de técnicas ligeras (consenso de extracciones)
            consensus_light = [
                (
                    "Consensus_Light",
                    f"{experiment_folder}/modules/networks/{community_name.removesuffix('_expression')}.csv",
                )
            ]
            df_aupr, df_auroc = evaluate_and_append(
                consensus_light, gold_standard_file, "Global", df_aupr, df_auroc
            )

            ## 2.4 Calcular precisión de consenso de técnicas pesadas
            consensus_heavy = [
                (
                    "Consensus_Heavy",
                    f"{experiment_folder}/modular_inference/{community_name}/consensus_modular_network.csv",
                )
            ]
            df_aupr, df_auroc = evaluate_and_append(
                consensus_heavy, gold_standard_file, "Modular", df_aupr, df_auroc
            )

            ## Guardar resultados en CSV
            df_aupr.to_csv(
                f"{community_measurements_folder}/{community_name}_AUPR.csv",
                index=False,
            )
            df_auroc.to_csv(
                f"{community_measurements_folder}/{community_name}_AUROC.csv",
                index=False,
            )

            ## Agregar resultados al DataFrame global
            community_size = gold_standard.shape[0]
            df_all_aupr = pd.concat(
                [
                    df_all_aupr,
                    df_aupr.assign(
                        Community=community_name,
                        Size=community_size,
                        NumGSLinks=num_gs_ones,
                    ),
                ],
                ignore_index=True,
            )
            df_all_auroc = pd.concat(
                [
                    df_all_auroc,
                    df_auroc.assign(
                        Community=community_name,
                        Size=community_size,
                        NumGSLinks=num_gs_ones,
                    ),
                ],
                ignore_index=True,
            )

        ## Guardar DataFrames globales
        df_all_aupr.to_csv(
            f"{experiment_folder}/measurements/communities/all_communities_AUPR.csv",
            index=False,
        )
        df_all_auroc.to_csv(
            f"{experiment_folder}/measurements/communities/all_communities_AUROC.csv",
            index=False,
        )

# 3. Calcular precisión de consenso global de técnicas ligeras y producto final construido
# for expression_file in expression_data:
    ## Especificar ruta del gold standard
    gold_standard_file = (
        f"{Path(expression_file).parent}/{Path(expression_file).parent.stem}_gs.csv"
    )

    ## Crear DataFrames para almacenar todos los resultados de AUPR y AUROC
    df_all_aupr = pd.DataFrame(
        columns=["Consensus", "Clustering", "Size", "Technique", "AUPR"]
    )
    df_all_auroc = pd.DataFrame(
        columns=["Consensus", "Clustering", "Size", "Technique", "AUROC"]
    )

    for (consensus, clustering_algorithm, size) in parameter_combinations:
        ## Crear carpeta de medidas globales
        experiment_folder = (
            f"{Path(expression_file).parent}/{consensus}_{clustering_algorithm}_{size}"
        )
        global_measurements_folder = f"{experiment_folder}/measurements/global/"
        Path(global_measurements_folder).mkdir(parents=True, exist_ok=True)

        ## Crear DataFrames para almacenar los resultados de esta experimentación de AUPR y AUROC
        df_aupr = pd.DataFrame(columns=["Technique", "AUPR"])
        df_auroc = pd.DataFrame(columns=["Technique", "AUROC"])

        ## 3.1 Calcular precisión de consenso global de técnicas ligeras
        global_consensus_light = (
            "Consensus_Light",
            f"{experiment_folder}/global_inference/consensus_global_network.csv",
        )
        df_aupr, df_auroc = evaluate_and_append(
            [global_consensus_light], gold_standard_file, None, df_aupr, df_auroc
        )

        ## 3.2 Calcular precisión de consenso global de técnicas pesadas
        global_consensus_heavy = ("Proposal", f"{experiment_folder}/merged_network.csv")
        df_aupr, df_auroc = evaluate_and_append(
            [global_consensus_heavy], gold_standard_file, None, df_aupr, df_auroc
        )

        ## Guardar resultados en CSV
        df_aupr.to_csv(f"{global_measurements_folder}/AUPR.csv", index=False)
        df_auroc.to_csv(f"{global_measurements_folder}/AUROC.csv", index=False)

        ## Agregar resultados al DataFrame global
        df_all_aupr = pd.concat(
            [
                df_all_aupr,
                df_aupr.assign(
                    Consensus=consensus, Clustering=clustering_algorithm, Size=size
                ),
            ],
            ignore_index=True,
        )
        df_all_auroc = pd.concat(
            [
                df_all_auroc,
                df_auroc.assign(
                    Consensus=consensus, Clustering=clustering_algorithm, Size=size
                ),
            ],
            ignore_index=True,
        )

    ## Guardar DataFrames globales
    df_all_aupr.to_csv(
        f"{Path(expression_file).parent}/consensus_AUPR.csv", index=False
    )
    df_all_auroc.to_csv(
        f"{Path(expression_file).parent}/consensus_AUROC.csv", index=False
    )

# 4. Calcular precisión de red global de todas las técnicas (ligeras y pesadas) con el producto final
for expression_file in expression_data:
    ## 4.1. Inferir redes completas con técnicas modulares
    output_folder = f"{Path(expression_file).parent}/global_heavy_inference/"
    df_times = pd.DataFrame(columns=["Technique", "Time"])
    for tec in modular_techniques:
        start_time = time.time()
        infer_network(
            expression_data=Path(expression_file),
            technique=[tec],
            threads=multiprocessing.cpu_count(),
            str_threads=None,
            temp_folder_str="tmp-"
            + "".join(random.choices(string.ascii_lowercase, k=10)),
            output_dir=Path(output_folder),
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        df_times = pd.concat(
            [
                df_times,
                pd.DataFrame({"Technique": [tec], "Time": [f"{elapsed_time:.2f}"]}),
            ],
            ignore_index=True,
        )
        print(df_times)
    df_times.to_csv(f"{output_folder}/measurements/inference_times.csv", index=False)

    ## 4.2. Evaluar precisión de redes completas inferidas por técnicas modulares y globales
    gold_standard_file = (
        f"{Path(expression_file).parent}/{Path(expression_file).parent.stem}_gs.csv"
    )
    df_aupr = pd.DataFrame(columns=["Technique", "Group", "AUPR"])
    df_auroc = pd.DataFrame(columns=["Technique", "Group", "AUROC"])
    global_networks_modular_tecs = [
        (tec, f"{output_folder}/{Path(expression_file).stem}/lists/GRN_{tec}.csv")
        for tec in modular_techniques
    ]
    df_aupr, df_auroc = evaluate_and_append(
        global_networks_modular_tecs, gold_standard_file, "Modular", df_aupr, df_auroc
    )
    (consensus, clustering_algorithm, size) = parameter_combinations[0]
    global_networks_global_tecs = [
        (
            tec,
            f"{Path(expression_file).parent}/{consensus}_{clustering_algorithm}_{size}/global_inference/{Path(expression_file).stem}/lists/GRN_{tec}.csv",
        )
        for tec in global_techniques
    ]
    df_aupr, df_auroc = evaluate_and_append(
        global_networks_global_tecs, gold_standard_file, "Global", df_aupr, df_auroc
    )

    df_aupr.to_csv(
        f"{output_folder}/{Path(expression_file).stem}/measurements/AUPR.csv",
        index=False,
    )
    df_auroc.to_csv(
        f"{output_folder}/{Path(expression_file).stem}/measurements/AUROC.csv",
        index=False,
    )

# 5. Crear gráfico de comparación de BIO-INSIGHT, MO-GENECI vs. Estrategia modular
