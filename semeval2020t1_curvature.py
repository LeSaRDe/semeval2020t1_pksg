import logging
import math
import sys
import time
from os import walk
import threading
import multiprocessing

import pandas as pd
from scipy import stats, sparse
import numpy as np
import networkx as nx

import semeval2020t1_global_settings as global_settings


def compute_semantic_measure(df_embed, embed_col_name, nx_graph, min_comp_size=1000):
    '''
    'df_embed' should be indexed by node labels of 'nx_graph'.
    '''
    logging.debug('[compute_semantic_measure] starts.')
    timer_start = time.time()

    l_sub_nx_graph = []
    for comp in nx.connected_components(nx_graph):
        if len(comp) < min_comp_size:
            continue
        l_sub_nx_graph.append(nx_graph.subgraph(comp))
    logging.debug('[compute_semantic_measure] nx_graph contains %s non-trivial subgraphs.' % len(l_sub_nx_graph))

    for sub_nx_graph in l_sub_nx_graph:
        # compute pairwise cosine distance
        np_embed_for_subgraph = df_embed.loc[list(sub_nx_graph.nodes())][embed_col_name]
        np_cos = np.matmul(np_embed_for_subgraph, np.transpose(np_embed_for_subgraph)).astype(np.float32)
        np_cos = (1.0 - np_cos) / 2.0
        np_cos = np.triu(np_cos, k=1)
        sp_cos = sparse.coo_matrix(np_cos)
        del np_cos

        # compute globally normalized weights
        sp_adj = nx.linalg.adjacency_matrix(sub_nx_graph)
        sp_adj = sparse.triu(sp_adj, k=1)
        sp_adj_sum = np.sum(sp_adj)
        if sp_adj_sum == 0.0:
            logging.error('[compute_semantic_measure] trivial sub_nx_graph: %s' % nx.info(sub_nx_graph))
            continue
        sp_adj = sp_adj / sp_adj_sum
        if sp_adj.shape != sp_cos.shape:
            raise Exception('[compute_semantic_measure] shape mismatched sp_cos and sp_adj')




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]