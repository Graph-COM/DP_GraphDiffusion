import numpy as np
import random
import argparse
from datasets.dataset import *
from model import *
from downstream.node_ranking import *
from privacy.noise_calibration import *
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arg Parse for graph diffusion privacy")
    parser.add_argument('--seed', dest='seed', type=str, default='123', help='the random seed set in the experiments')
    parser.add_argument('--k', type=int, default=100, help='top k nodes in ranking')
    parser.add_argument('--m', type=int, default=100, help='number of sampled nodes on graphs')
    parser.add_argument('--epsilon', type=float, default=0.1, help='DP privacy budget of graph diffusion')
    parser.add_argument('--delta', type=float, default=None, help='DP failure probability')
    parser.add_argument('--beta', type=float, default=0.8, help='teleport probability for PageRank')
    parser.add_argument('--eta', type=float, default=1e-8, help='threshold')
    parser.add_argument('--max_iter', type=int, default=100, help='propagation iteration for PageRank')
    parser.add_argument('--dataset', type=str, choices=['BlogCatalog', 'Themarker', 'Flickr'], default='BlogCatalog', help='Dataset to use')
    parser.add_argument('--method', type=str, choices=['our', 'pushflow', 'edgeflipping', 'all'], default='all', help='Method to run')

    args = parser.parse_args()

    if args.delta is None:
        if args.dataset == 'BlogCatalog':
            args.delta = 3e-6
        elif args.dataset == 'Themarker':
            args.delta = 6e-7
        elif args.dataset == 'Flickr':
            args.delta = 2e-7

    # Create a console instance
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Argument", style="dim", width=12)
    table.add_column("Value")

    # Add rows to the table
    for arg in vars(args):
        table.add_row(arg, str(getattr(args, arg)))

    # Print the table
    console.print(table)

    k = args.k
    m = args.m
    max_iter = args.max_iter
    eta = args.eta
    beta = args.beta

    epsilon = args.epsilon
    delta = args.delta

    setup_seed(int(args.seed))

    if args.dataset == 'BlogCatalog':
        dataset = BlogCatalogDataset()
    elif args.dataset == 'Themarker':
        dataset = ThemarkerDataset()
    elif args.dataset == 'Flickr':
        dataset = FlickrDataset()

    # Node sampling
    sampled_indices = np.random.choice(dataset.nodes, size=m, replace=False)

    # Noise calibration for our method
    sigma, _, _ = noise_calibration_laplace(epsilon=epsilon, delta=delta, eta=eta, eta_l1=4 * eta * beta / (1 - beta), beta=beta, max_iter=max_iter)

    # Run selected method(s)
    if args.method in ['our', 'all']:
        pagerank = PageRankGlobal(dataset, beta=beta, max_iter=max_iter)
        pagerank_private_clip = PrivatePageRankClip(dataset, epsilon=epsilon, eta=eta, delta=delta, sigma=sigma, beta=beta, max_iter=max_iter, sample=False)
        sampled_score_matrix = pagerank.propagate_selected(sampled_indices)
        sampled_score_matrix_private_our = pagerank_private_clip.propagate_selected(sampled_indices, noise_type='Laplacian')
        top_k_indices, top_k_scores = get_top_k_indices_scores(sampled_score_matrix, sampled_indices, k)
        top_k_indices_private_our, top_k_scores_private_our = get_top_k_indices_scores(sampled_score_matrix_private_our, sampled_indices, k)
        avg_ndcg_our, std_ndcg_our = calculate_ndcg(top_k_indices, top_k_scores, top_k_indices_private_our)
        avg_recall_our, std_recall_our = calculate_recall(top_k_indices, top_k_indices_private_our, r=100)
        print(f'Our Avg_ndcg: {avg_ndcg_our}')
        print(f'Our Avg_recall: {avg_recall_our}')

    if args.method in ['pushflow', 'all']:
        pushflow = PrivatePushFlowEfficient(dataset, beta=beta, epsilon=epsilon, delta=delta, zeta=1e-6, max_iter=max_iter, eta=eta)
        sampled_score_matrix_private_pushflow = pushflow.compute_private_pushflow_selected(sampled_indices, noise_type='Laplacian')
        top_k_indices_private_pushflow, top_k_scores_private_pushflow = get_top_k_indices_scores(sampled_score_matrix_private_pushflow, sampled_indices, k)
        avg_ndcg_pushflow, std_ndcg_pushflow = calculate_ndcg(top_k_indices, top_k_scores, top_k_indices_private_pushflow)
        avg_recall_pushflow, std_recall_pushflow = calculate_recall(top_k_indices, top_k_indices_private_pushflow, r=100)
        print(f'Pushflow Avg_ndcg: {avg_ndcg_pushflow}')
        print(f'Pushflow Avg_recall: {avg_recall_pushflow}')

    if args.method in ['edgeflipping', 'all']:
        edgeflip = EdgeFlipping(dataset, epsilon=epsilon, beta=beta, max_iter=max_iter)
        sampled_score_matrix_private_edgeflip = edgeflip.propagate_selected_personalized(sampled_indices)
        top_k_indices_private_edgeflip, top_k_scores_private_edgeflip = get_top_k_indices_scores(sampled_score_matrix_private_edgeflip, sampled_indices, k)
        avg_ndcg_edgeflip, std_ndcg_edgeflip = calculate_ndcg(top_k_indices, top_k_scores, top_k_indices_private_edgeflip)
        avg_recall_edgeflip, std_recall_edgeflip = calculate_recall(top_k_indices, top_k_indices_private_edgeflip, r=100)
        print(f'EdgeFlipping Avg_ndcg: {avg_ndcg_edgeflip}')
        print(f'EdgeFlipping Avg_recall: {avg_recall_edgeflip}')
