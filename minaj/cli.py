import multiprocessing
from pathlib import Path
from typing import List, Optional

import typer
from geneci.main import SimpleConsensusCriteria, Technique
from rich.console import Console

from minaj.main import main_cluster_network, main_modular_inference
from minaj.utils import ClusteringAlgorithm

console = Console()

HEADER = """[bold gold1]
   __     __)    _____ __     __) _____     _____ 
  (, /|  /|     (, /  (, /|  /   (, /  |   (, /   
    / | / |       /     / | /      /---|     /    
 ) /  |/  |_  ___/__ ) /  |/    ) /    |____/__   
(_/   '     (__ /   (_/   '    (_/     /   /      
                                      (__ /      
[/bold gold1]
[bright_white]Modular Inference for Network Aggregation and Joint learning[/bright_white]
[dim]Author: Adrián Segura Ortiz <adrianseor.99@gmail.com>[/dim]
"""

console.print(HEADER)

app = typer.Typer(rich_markup_mode="rich")

@app.command()
def modular_inference(
    expression_data: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        help="Path to the CSV file with the expression data. Genes are distributed in rows and experimental conditions in columns.",
    ),
    global_techniques: Optional[List[Technique]] = typer.Option(
        ...,
        case_sensitive=False,
        help="Light and less precise techniques to carry out a direct inference of the global network.",
    ),
    modular_techniques: Optional[List[Technique]] = typer.Option(
        ...,
        case_sensitive=False,
        help="Higher cost and accuracy techniques for inferring small subdivisions of the global network.",
    ),
    consensus_criteria: SimpleConsensusCriteria = typer.Option(
        SimpleConsensusCriteria.MeanWeights,
        case_sensitive=False,
        help="Simple criterion for agreeing networks from different techniques.",
    ),
    algorithm: ClusteringAlgorithm = typer.Option(
        ClusteringAlgorithm.Leiden,
        help="Clustering algorithm to be used for extracting modules from the global network",
    ),
    preferred_size: int = typer.Option(100, help="Preferred size of the modules"),
    threads: int = typer.Option(
        multiprocessing.cpu_count(),
        help="Number of threads to be used during parallelization. By default, the maximum number of threads available in the system is used.",
    ),
    output_dir: Path = typer.Option(
        Path("./modular_inference"), help="Path to the output folder."
    ),
):
    """
    Apply modular inference to a gene network
    """

    main_modular_inference(
        expression_data=expression_data,
        global_techniques=global_techniques,
        modular_techniques=modular_techniques,
        consensus_criteria=consensus_criteria,
        algorithm=algorithm,
        preferred_size=preferred_size,
        threads=threads,
        output_dir=output_dir,
    )

@app.command()
def cluster_network(
    confidence_list: str = typer.Option(
        ..., help="Path of the CSV file with the confidence list to be clustered"
    ),
    algorithm: ClusteringAlgorithm = typer.Option(
        "Infomap", help="Clustering algorithm to apply"
    ),
    preferred_size: int = typer.Option(
        100, help="Preferred number of nodes per community"
    ),
    min_size: int = typer.Option(5, help="Minimum size of the communities"),
    output_folder: str = typer.Option(
        ..., help="Output folder to write community files"
    ),
):

    """
    Main pipeline to read a network, apply community detection, and save results.
    """

    main_cluster_network(
        confidence_list=confidence_list,
        algorithm=algorithm,
        preferred_size=preferred_size,
        min_size=min_size,
        output_folder=output_folder,
    )