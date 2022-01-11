import logging
import math
import time
import sys
from os import walk

import numpy as np
import pandas as pd
from sklearn import preprocessing
import networkx as nx
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from scipy.stats import entropy


import semeval2020t1_global_settings as global_settings


def collect_token_embed_by_tau(l_ds_name, l_tau, l_tokens, task_name):
    logging.debug('[compare_adj_token_embed_by_tau] starts.')
    timer_start = time.time()

    d_df_adj_token_embed = dict()
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        d_df_adj_token_embed_per_tau = dict()
        for ds_name in l_ds_name:
            df_adj_token_embed = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name + '#' + tau_str))
            d_df_adj_token_embed_per_tau[ds_name] = df_adj_token_embed
        d_df_adj_token_embed[tau_str] = d_df_adj_token_embed_per_tau
    logging.debug('[compare_adj_token_embed_by_tau] load in all adj_token_embed in %s secs.'
                  % str(time.time() - timer_start))

    d_df_update_adj_token_embed = dict()
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        d_df_update_adj_token_embed_per_tau = dict()
        df_update_adj_token_embed_12 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt
                                                      .format(l_ds_name[0] + '#rel#' + l_ds_name[1] + '#' + tau_str))
        d_df_update_adj_token_embed[tau_str] = dict()
        d_df_update_adj_token_embed[tau_str][l_ds_name[0]] = df_update_adj_token_embed_12
        df_update_adj_token_embed_21 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt
                                                      .format(l_ds_name[1] + '#rel#' + l_ds_name[0] + '#' + tau_str))
        d_df_update_adj_token_embed[tau_str][l_ds_name[1]] = df_update_adj_token_embed_21
    logging.debug('[compare_adj_token_embed_by_tau] load in all update_adj_token_embed in %s secs.'
                  % str(time.time() - timer_start))


    d_df_orig_token_embed = dict()
    for ds_name in l_ds_name:
        df_orig_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
        d_df_orig_token_embed[ds_name] = df_orig_token_embed
    logging.debug('[compare_adj_token_embed_by_tau] load in all orig_token_embed in %s secs.'
                  % str(time.time() - timer_start))

    l_token_embed = []
    l_col_name = ['token']
    col_name_done = False
    for token in l_tokens:
        embed_rec = [token]
        for ds_name in l_ds_name:
            token_embed = d_df_orig_token_embed[ds_name].loc[token]['token_embed']
            embed_rec.append(token_embed)
            if not col_name_done:
                l_col_name.append(ds_name + '#orig')
        for tau in d_df_adj_token_embed:
            for ds_name in l_ds_name:
                token_embed = d_df_adj_token_embed[tau][ds_name].loc[token]['adj_token_embed']
                embed_rec.append(token_embed)
                if not col_name_done:
                    l_col_name.append(ds_name + '#adj#' + tau)
                token_embed = d_df_update_adj_token_embed[tau][ds_name].loc[token]['adj_token_embed']
                embed_rec.append(token_embed)
                if not col_name_done:
                    l_col_name.append(ds_name + '#update_adj#' + tau)
        l_token_embed.append(tuple(embed_rec))
        col_name_done = True
    df_token_embed = pd.DataFrame(l_token_embed, columns=l_col_name)
    df_token_embed = df_token_embed.set_index('token')
    logging.debug('[compare_adj_token_embed_by_tau] df_token_embed done %s in %s secs.'
                  % (len(df_token_embed), time.time() - timer_start))
    pd.to_pickle(df_token_embed, global_settings.g_token_embed_collect_file_fmt.format(task_name))
    logging.debug('[compare_adj_token_embed_by_tau] all done in %s secs.'
                  % str(time.time() - timer_start))


def compare_token_with_fixed_points(l_tau, fp_cos_threshold, fp_deg_threshold):
    df_orig_ds1 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('ccoha1'))
    df_orig_ds2 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('ccoha2'))

    tkg_ds1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha1'))
    tkg_ds2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha2'))

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    l_gt = [1] * 16 + [0] * 21

    np_eval_orig = np.stack(df_orig_ds1['token_embed'].reindex(l_eval_token).to_list())

    d_df_adj_ds1 = dict()
    d_df_adj_ds2 = dict()
    d_np_eval_adj_ds1 = dict()
    d_np_eval_adj_ds2 = dict()
    d_l_fp_token_share = dict()
    d_np_fp_share_orig = dict()
    d_np_fp_share_adj_ds1 = dict()
    d_np_fp_share_adj_ds2 = dict()
    d_np_eval_adj_vs_fp_orig_ds1 = dict()
    d_np_eval_adj_vs_fp_orig_ds2 = dict()
    d_np_eval_adj_vs_fp_adj_ds1 = dict()
    d_np_eval_adj_vs_fp_adj_ds2 = dict()
    d_l_eval_token_score_fp_share_orig = dict()
    d_l_eval_token_score_fp_share_adj = dict()
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        df_adj_ds1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('ccoha1#' + tau_str))
        d_df_adj_ds1[tau_str] = df_adj_ds1
        df_adj_ds2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('ccoha2#' + tau_str))
        d_df_adj_ds2[tau_str] = df_adj_ds2

        np_eval_adj_ds1 = np.stack(df_adj_ds1['adj_token_embed'].reindex(l_eval_token).to_list())
        d_np_eval_adj_ds1[tau_str] = np_eval_adj_ds1
        np_eval_adj_ds2 = np.stack(df_adj_ds2['adj_token_embed'].reindex(l_eval_token).to_list())
        d_np_eval_adj_ds2[tau_str] = np_eval_adj_ds2

        # # find shared fixed points
        # np_cos_orig_adj_ds1 = np.einsum('ij..., ij...->i',
        #                                 np.stack(df_orig_ds1['token_embed'].reindex(df_adj_ds1.index).to_list()),
        #                                 np.stack(df_adj_ds1['adj_token_embed'].to_list()))
        # l_fp_idx_ds1 = np.where(np_cos_orig_adj_ds1 >= fp_cos_threshold)[0]
        # if len(l_fp_idx_ds1) <= 0:
        #     continue
        # l_fp_token_ds1 = [df_adj_ds1.index.to_list()[fp_idx] for fp_idx in l_fp_idx_ds1]
        # np_cos_orig_adj_ds2 = np.einsum('ij..., ij...->i',
        #                                 np.stack(df_orig_ds2['token_embed'].reindex(df_adj_ds2.index).to_list()),
        #                                 np.stack(df_adj_ds2['adj_token_embed'].to_list()))
        # l_fp_idx_ds2 = np.where(np_cos_orig_adj_ds2 >= fp_cos_threshold)[0]
        # if len(l_fp_idx_ds2) <= 0:
        #     continue
        # l_fp_token_ds2 = [df_adj_ds2.index.to_list()[fp_idx] for fp_idx in l_fp_idx_ds2]
        #
        # l_fp_token_share = list(set(l_fp_token_ds1).intersection(set(l_fp_token_ds2)))
        # if len(l_fp_token_share) <= 0:
        #     continue
        #
        # l_fp_share_deg_ds1 = [item[0] for item in list(tkg_ds1.degree(l_fp_token_share)) if item[1] >= fp_deg_threshold]
        # if len(l_fp_share_deg_ds1) <= 0:
        #     continue
        # l_fp_share_deg_ds2 = [item[0] for item in list(tkg_ds2.degree(l_fp_token_share)) if item[1] >= fp_deg_threshold]
        # if len(l_fp_share_deg_ds2) <= 0:
        #     continue
        #
        # l_fp_token_share = list(set(l_fp_share_deg_ds1).intersection(set(l_fp_share_deg_ds2)))
        # if len(l_fp_token_share) <= 0:
        #     continue
        #
        # d_l_fp_token_share[tau_str] = l_fp_token_share
        #
        # np_fp_share_orig = np.stack(df_orig_ds1['token_embed'].reindex(l_fp_token_share).to_list())
        # d_np_fp_share_orig[tau_str] = np_fp_share_orig
        #
        # np_fp_share_adj_ds1 = np.stack(df_adj_ds1['adj_token_embed'].reindex(l_fp_token_share).to_list())
        # d_np_fp_share_adj_ds1[tau_str] = np_fp_share_adj_ds1
        # np_fp_share_adj_ds2 = np.stack(df_adj_ds2['adj_token_embed'].reindex(l_fp_token_share).to_list())
        # d_np_fp_share_adj_ds2[tau_str] = np_fp_share_adj_ds2
        #
        # # map eval token to fp share orig
        # np_eval_adj_vs_fp_orig_ds1 = np.matmul(np_eval_adj_ds1, np.transpose(np_fp_share_orig))
        # d_np_eval_adj_vs_fp_orig_ds1[tau_str] = np_eval_adj_vs_fp_orig_ds1
        # np_eval_adj_vs_fp_orig_ds2 = np.matmul(np_eval_adj_ds2, np.transpose(np_fp_share_orig))
        # d_np_eval_adj_vs_fp_orig_ds2[tau_str] = np_eval_adj_vs_fp_orig_ds2
        #
        # # map eval token to fp share adj
        # np_eval_adj_vs_fp_adj_ds1 = np.matmul(np_eval_adj_ds1, np.transpose(np_fp_share_adj_ds1))
        # d_np_eval_adj_vs_fp_adj_ds1[tau_str] = np_eval_adj_vs_fp_adj_ds1
        # np_eval_adj_vs_fp_adj_ds2 = np.matmul(np_eval_adj_ds2, np.transpose(np_fp_share_adj_ds2))
        # d_np_eval_adj_vs_fp_adj_ds2[tau_str] = np_eval_adj_vs_fp_adj_ds2
        #
        # # eval token ds1 vs ds2 by fp share orig
        # l_eval_token_score_fp_share_orig = []
        # for token_idx in range(len(l_eval_token)):
        #     js_token = jensenshannon(softmax(np_eval_adj_vs_fp_orig_ds1[token_idx]),
        #                              softmax(np_eval_adj_vs_fp_orig_ds2[token_idx]))
        #     l_eval_token_score_fp_share_orig.append(js_token)
        # d_l_eval_token_score_fp_share_orig[tau_str] = l_eval_token_score_fp_share_orig
        #
        # # eval token ds1 vs ds2 by fp share adj
        # l_eval_token_score_fp_share_adj = []
        # for token_idx in range(len(l_eval_token)):
        #     js_token = jensenshannon(softmax(np_eval_adj_vs_fp_adj_ds1[token_idx]),
        #                              softmax(np_eval_adj_vs_fp_adj_ds2[token_idx]))
        #     l_eval_token_score_fp_share_adj.append(js_token)
        # d_l_eval_token_score_fp_share_adj[tau_str] = l_eval_token_score_fp_share_adj

    print()


def find_fixed_points(l_tau, fp_cos_threshold, fp_deg_threshold):
    '''
    !!!CAUTION!!!
    Is it true that the tokens whose cosine sims between t and t-1 are large do not have significant semantic change?
    We don't know this! As the embed adjust alg may have adjusted the token vectors in a relative manner, i.e. relative
    to each other within the given context, rather than absolutely keeping some ones unchanged globally if any, then
    if we see a token doesn't have much change in its vector from t-1 to t, it may not mean that its semantics is kept
    unchanged.
    '''
    logging.debug('[find_fixed_points] starts.')
    timer_start = time.time()

    df_orig_ds1 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('ccoha1'))
    df_orig_ds2 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('ccoha2'))

    tkg_ds1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha1'))
    tkg_ds2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha2'))

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    l_l_fp_token_share = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        df_adj_ds1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('ccoha1#' + tau_str))
        df_adj_ds2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('ccoha2#' + tau_str))

        # find shared fixed points
        np_cos_orig_adj_ds1 = np.einsum('ij..., ij...->i',
                                        np.stack(df_orig_ds1['token_embed'].reindex(df_adj_ds1.index).to_list()),
                                        np.stack(df_adj_ds1['adj_token_embed'].to_list()))
        l_fp_idx_ds1 = np.where(np_cos_orig_adj_ds1 >= fp_cos_threshold)[0]
        if len(l_fp_idx_ds1) <= 0:
            continue
        l_fp_token_ds1 = [df_adj_ds1.index.to_list()[fp_idx] for fp_idx in l_fp_idx_ds1]
        np_cos_orig_adj_ds2 = np.einsum('ij..., ij...->i',
                                        np.stack(df_orig_ds2['token_embed'].reindex(df_adj_ds2.index).to_list()),
                                        np.stack(df_adj_ds2['adj_token_embed'].to_list()))
        l_fp_idx_ds2 = np.where(np_cos_orig_adj_ds2 >= fp_cos_threshold)[0]
        if len(l_fp_idx_ds2) <= 0:
            continue
        l_fp_token_ds2 = [df_adj_ds2.index.to_list()[fp_idx] for fp_idx in l_fp_idx_ds2]

        l_fp_token_share = list(set(l_fp_token_ds1).intersection(set(l_fp_token_ds2)))
        if len(l_fp_token_share) <= 0:
            continue

        l_fp_share_deg_ds1 = [item[0] for item in list(tkg_ds1.degree(l_fp_token_share)) if item[1] >= fp_deg_threshold]
        if len(l_fp_share_deg_ds1) <= 0:
            continue
        l_fp_share_deg_ds2 = [item[0] for item in list(tkg_ds2.degree(l_fp_token_share)) if item[1] >= fp_deg_threshold]
        if len(l_fp_share_deg_ds2) <= 0:
            continue

        l_fp_token_share = list(set(l_fp_share_deg_ds1).intersection(set(l_fp_share_deg_ds2)))
        if len(l_fp_token_share) <= 0:
            continue

        l_l_fp_token_share.append((tau_str, l_fp_token_share))
        logging.debug('[find_fixed_points] get %s fixed points for tau %s in %s secs.'
                      % (len(l_fp_token_share), tau_str, time.time() - timer_start))
    df_fp = pd.DataFrame(l_l_fp_token_share, columns=['tau', 'fp'])
    pd.to_pickle(df_fp, global_settings.g_fixed_points_file)
    logging.debug('[find_fixed_points] all done in %s secs.' % str(time.time() - timer_start))


def find_fixed_points_by_deg(use_cckg=True):
    logging.debug('[find_fixed_points_by_deg] starts.')
    timer_start = time.time()

    # tkg_ds1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha1'))
    # tkg_ds2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('ccoha2'))

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    init_sel_node_guess = 200
    node_cover_ratio = 0.95
    deg_threshold = 3000
    l_ret = []
    for ds_name in ['ccoha1', 'ccoha2']:
        if use_cckg:
            ds_name = 'cckg#' + ds_name
        tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
        large_subtkg = sorted([comp for comp in nx.connected_components(tkg)], key=lambda k: len(k), reverse=True)[0]
        large_subtkg = tkg.subgraph(large_subtkg)
        total_num_nodes = len(large_subtkg.nodes)
        l_fp = []

        l_sorted_deg = sorted(list(large_subtkg.degree()), key=lambda k: k[1], reverse=True)
        l_sel_node = [item[0] for item in l_sorted_deg][:init_sel_node_guess]
        l_sel_node = [token for token in l_sel_node if token not in l_eval_token]
        l_covered_node = []
        for node in l_sel_node:
            l_covered_node += list(large_subtkg.neighbors(node))
        l_covered_node = set(l_covered_node)

        sel_node_cur = init_sel_node_guess
        while sel_node_cur < total_num_nodes and len(l_covered_node) < total_num_nodes * node_cover_ratio:
            new_sel_node = l_sorted_deg[sel_node_cur][0]
            if new_sel_node in l_eval_token:
                sel_node_cur += 1
                continue
            l_sel_node.append(new_sel_node)
            l_covered_node = l_covered_node.union(set(large_subtkg.neighbors(new_sel_node)))
            sel_node_cur += 1
        l_fp = l_sel_node
        logging.debug('[find_fixed_points_by_deg] %s fp for %s in %s secs'
                      % (len(l_fp), ds_name, time.time() - timer_start))

        l_ret.append((ds_name, l_fp))
    l_fp_share = list(set([token for token in l_ret[0][1] if token in l_ret[1][1]]))
    with open(global_settings.g_shared_fixed_points_by_deg_file, 'w+') as out_fd:
        out_str = '\n'.join(l_fp_share)
        out_fd.write(out_str)
        out_fd.close()
    logging.debug('[find_fixed_points_by_deg] %s shared fp in %s secs' % (len(l_fp_share), time.time() - timer_start))
    df_ret = pd.DataFrame(l_ret, columns=['ds_name', 'fp'])
    df_ret = df_ret.set_index('ds_name')
    pd.to_pickle(df_ret, global_settings.g_fixed_points_by_deg_file)
    logging.debug('[find_fixed_points_by_deg] all done in %s secs.' % str(time.time() - timer_start))


def find_fp_by_measure(use_cckg=True):
    logging.debug('[find_fp_by_measure] starts.')
    timer_start = time.time()

    l_tkg = []
    l_large_subtkg = []
    for ds_name in ['ccoha1', 'ccoha2']:
        if use_cckg:
            ds_name = 'cckg#' + ds_name
        tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
        l_tkg.append(tkg)
        large_subtkg = sorted([comp for comp in nx.connected_components(tkg)], key=lambda k: len(k), reverse=True)[0]
        large_subtkg = tkg.subgraph(large_subtkg)
        l_large_subtkg.append(large_subtkg)
    tkg_ds1 = l_tkg[0]
    tkg_ds2 = l_tkg[1]
    l_tkg_node_ds1 = list(tkg_ds1.nodes())
    l_tkg_node_ds2 = list(tkg_ds2.nodes())
    large_subtkg_ds1 = l_large_subtkg[0]
    large_subtkg_ds2 = l_large_subtkg[0]
    l_large_subtkg_node_idx_ds1 = [l_tkg_node_ds1.index(token) for token in large_subtkg_ds1.nodes()]
    l_large_subtkg_node_idx_ds2 = [l_tkg_node_ds2.index(token) for token in large_subtkg_ds2.nodes()]

    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        l_df_adj_ds = []
        for ds_name in l_ds_name:
            ds_name = ds_name + '#' + tau_str
            if use_cckg:
                ds_name = 'cckg#' + ds_name
            df_adj_ds = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name))
            l_df_adj_ds.append(df_adj_ds)
        df_adj_ds1 = l_df_adj_ds[0]
        df_adj_ds2 = l_df_adj_ds[1]

        # find cos distances on all edges




def compare_token_by_fp(l_tau, use_cckg=True, use_fp=True):
    logging.debug('[compare_token_by_fp] starts.')
    timer_start = time.time()

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    l_shared_fp = []
    with open(global_settings.g_shared_fixed_points_by_deg_file, 'r') as in_fd:
        for ln in in_fd:
            l_shared_fp.append(ln.strip())
        in_fd.close()

    l_ds_name = ['ccoha1', 'ccoha2']

    l_np_token_cos_fp_ds1 = []
    l_np_token_cos_fp_ds2 = []
    l_np_token_ds1_cos_ds2 = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))

        l_df_adj_ds = []
        for ds_name in l_ds_name:
            ds_name = ds_name + '#' + tau_str
            if use_cckg:
                ds_name = 'cckg#' + ds_name
            if use_fp:
                ds_name = 'fp#' + ds_name
            df_adj_ds = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name))
            l_df_adj_ds.append(df_adj_ds)

        df_adj_ds1 = l_df_adj_ds[0]
        df_adj_ds2 = l_df_adj_ds[1]

        l_fp_adj_ds1 = []
        l_fp_adj_ds2 = []
        for fp in l_shared_fp:
            fp_adj_ds1 = df_adj_ds1.loc[fp]['adj_token_embed']
            l_fp_adj_ds1.append(fp_adj_ds1)
            fp_adj_ds2 = df_adj_ds2.loc[fp]['adj_token_embed']
            l_fp_adj_ds2.append(fp_adj_ds2)
        np_fp_adj_ds1 = np.stack(l_fp_adj_ds1)
        np_fp_adj_ds2 = np.stack(l_fp_adj_ds2)

        l_token_adj_ds1 = []
        l_token_adj_ds2 = []
        for token in l_eval_token:
            token_adj_ds1 = df_adj_ds1.loc[token]['adj_token_embed']
            l_token_adj_ds1.append(token_adj_ds1)
            token_adj_ds2 = df_adj_ds2.loc[token]['adj_token_embed']
            l_token_adj_ds2.append(token_adj_ds2)
        np_token_adj_ds1 = np.stack(l_token_adj_ds1)
        np_token_adj_ds2 = np.stack(l_token_adj_ds2)

        np_token_cos_fp_ds1 = np.matmul(np_token_adj_ds1, np.transpose(np_fp_adj_ds1))
        l_np_token_cos_fp_ds1.append(np_token_cos_fp_ds1)
        np_token_cos_fp_ds2 = np.matmul(np_token_adj_ds2, np.transpose(np_fp_adj_ds2))
        l_np_token_cos_fp_ds2.append(np_token_cos_fp_ds2)

        np_token_ds1_cos_ds2 = np.einsum('ij..., ij...->i', np_token_adj_ds1, np_token_adj_ds2)
        l_np_token_ds1_cos_ds2.append(np_token_ds1_cos_ds2)

    l_np_js_token_ds1_vs_ds2 = []
    for tau_i in range(len(l_tau)):
        l_js_token_ds1_vs_ds2_per_tau = []
        for token_j in range(len(l_eval_token)):
            js_token_ds1_vs_ds2_per_tau = entropy(softmax(l_np_token_cos_fp_ds1[tau_i][token_j]),
                                                        softmax(l_np_token_cos_fp_ds2[tau_i][token_j]))
            l_js_token_ds1_vs_ds2_per_tau.append(js_token_ds1_vs_ds2_per_tau)
        np_js_token_ds1_vs_ds2_per_tau = np.asarray(l_js_token_ds1_vs_ds2_per_tau)
        l_np_js_token_ds1_vs_ds2.append(np_js_token_ds1_vs_ds2_per_tau)

    np_mean_js_token_ds1_vs_ds2 = np.mean(l_np_js_token_ds1_vs_ds2, axis=0)
    print(np_mean_js_token_ds1_vs_ds2)


def compare_token_embeds(task_name, l_tau, ordered_tokens):
    logging.debug('[compare_token_embeds] starts.')
    timer_start = time.time()

    df_token_embed = pd.read_pickle(global_settings.g_token_embed_collect_file_fmt.format(task_name))
    df_token_embed = df_token_embed.reindex(ordered_tokens)
    logging.debug('[compare_token_embeds] load in df_token_embed: %s' % len(df_token_embed))

    np_orig_ds1 = np.stack(df_token_embed['ccoha1#orig'].to_list())
    np_orig_ds2 = np.stack(df_token_embed['ccoha2#orig'].to_list())

    l_np_adj_ds1 = []
    l_np_adj_ds2 = []
    l_np_update_adj_ds1_ds2 = []
    l_np_update_adj_ds2_ds1 = []
    for tau in l_tau:
        np_adj_ds1 = np.stack(df_token_embed['ccoha1#adj#' + str(np.round(tau, decimals=1))].to_list())
        np_adj_ds1 = preprocessing.normalize(np_adj_ds1)
        l_np_adj_ds1.append(np_adj_ds1)
        np_adj_ds2 = np.stack(df_token_embed['ccoha2#adj#' + str(np.round(tau, decimals=1))].to_list())
        np_adj_ds2 = preprocessing.normalize(np_adj_ds2)
        l_np_adj_ds2.append(np_adj_ds2)
        np_update_adj_ds1 = np.stack(df_token_embed['ccoha1#update_adj#' + str(np.round(tau, decimals=1))].to_list())
        np_update_adj_ds1 = preprocessing.normalize(np_update_adj_ds1)
        l_np_update_adj_ds1_ds2.append(np_update_adj_ds1)
        np_update_adj_ds2 = np.stack(df_token_embed['ccoha2#update_adj#' + str(np.round(tau, decimals=1))].to_list())
        np_update_adj_ds2 = preprocessing.normalize(np_update_adj_ds2)
        l_np_update_adj_ds2_ds1.append(np_update_adj_ds2)

    l_np_orig_ds1_vs_adj_ds1 = []
    l_np_orig_ds2_vs_adj_ds2 = []
    l_np_adj_ds1_vs_adj_ds2 = []
    l_np_update_adj_ds1_vs_adj_ds2 = []
    l_np_update_adj_ds2_vs_adj_ds1 = []
    for tau_id, tau in enumerate(l_tau):
        np_orig_ds1_vs_adj_ds1 = np.einsum('ij..., ij...->i', np_orig_ds1, l_np_adj_ds1[tau_id])
        l_np_orig_ds1_vs_adj_ds1.append(np_orig_ds1_vs_adj_ds1)

        np_orig_ds2_vs_adj_ds2 = np.einsum('ij..., ij...->i', np_orig_ds2, l_np_adj_ds2[tau_id])
        l_np_orig_ds2_vs_adj_ds2.append(np_orig_ds2_vs_adj_ds2)

        np_adj_ds1_vs_adj_ds2 = np.einsum('ij..., ij...->i', l_np_adj_ds1[tau_id], l_np_adj_ds2[tau_id])
        l_np_adj_ds1_vs_adj_ds2.append(np_adj_ds1_vs_adj_ds2)

        np_update_ds1_vs_adj_ds1 = np.einsum('ij..., ij...->i', l_np_update_adj_ds1_ds2[tau_id], l_np_adj_ds2[tau_id])
        l_np_update_adj_ds1_vs_adj_ds2.append(np_update_ds1_vs_adj_ds1)

        np_update_ds2_vs_adj_ds2 = np.einsum('ij..., ij...->i', l_np_update_adj_ds2_ds1[tau_id], l_np_adj_ds1[tau_id])
        l_np_update_adj_ds2_vs_adj_ds1.append(np_update_ds2_vs_adj_ds2)

    print()


def compare_token_by_adj_embed_merge():
    logging.debug('[compare_token_by_adj_embed_merge] starts.')

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    l_np_ds1_cos_merge = []
    l_np_ds2_cos_merge = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))

        ds_name_1 = 'cckg#ccoha1#' + tau_str
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1))
        ds_name_2 = 'cckg#ccoha2#' + tau_str
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2))

        ds_name_merge = 'merge#' + tau_str
        df_adj_embed_merge = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_merge))

        np_eval_adj_embed_1 = np.stack(df_adj_embed_1.loc[l_eval_token]['adj_token_embed'].to_list())
        np_eval_adj_embed_2 = np.stack(df_adj_embed_2.loc[l_eval_token]['adj_token_embed'].to_list())
        np_eval_adj_embed_merge = np.stack(df_adj_embed_merge.loc[l_eval_token]['adj_token_embed'].to_list())

        np_ds1_cos_merge = np.einsum('ij..., ij...->i', np_eval_adj_embed_1, np_eval_adj_embed_merge)
        l_np_ds1_cos_merge.append(np_ds1_cos_merge)
        np_ds2_cos_merge = np.einsum('ij..., ij...->i', np_eval_adj_embed_2, np_eval_adj_embed_merge)
        l_np_ds2_cos_merge.append(np_ds2_cos_merge)

    print()


def compare_token_by_curv():
    logging.debug('[compare_token_by_curv] starts.')

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    ds_name_1 = 'cckg#ccoha1'
    ds_name_2 = 'cckg#ccoha2'

    tkg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1))
    tkg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_2))

    d_neig = dict()
    d_neig_ew = dict()
    for token in l_eval_token:
        l_neig_1 = list(tkg_1.neighbors(token))
        l_neig_2 = list(tkg_2.neighbors(token))
        d_neig[token] = [l_neig_1, l_neig_2]

        l_neig_ew_1 = []
        for neig_1 in l_neig_1:
            ew_1 = tkg_1.edges[token, neig_1]['weight']
            l_neig_ew_1.append(ew_1)
        np_neig_ew_1 = np.asarray(l_neig_ew_1)
        # np_neig_ew_1 = preprocessing.normalize(np.asarray(l_neig_ew_1).reshape(1, -1), axis=1, norm='l1')[0]
        l_neig_ew_2 = []
        for neig_2 in l_neig_2:
            ew_2 = tkg_2.edges[token, neig_2]['weight']
            l_neig_ew_2.append(ew_2)
        np_neig_ew_2 = np.asarray(l_neig_ew_2)
        # np_neig_ew_2 = preprocessing.normalize(np.asarray(l_neig_ew_2).reshape(1, -1), axis=1, norm='l1')[0]
        d_neig_ew[token] = [np_neig_ew_1, np_neig_ew_2]


    # d_neig_overlap_weight_ratio = dict()
    # for token in l_eval_token:
    #     s_overlap_neig = set(d_neig[token][0]).intersection(set(d_neig[token][1]))
    #     l_w_1 = []
    #     for neig in s_overlap_neig:
    #         w_1 = tkg_1.edges[token, neig]['weight']
    #         l_w_1.append(w_1)
    #     sum_overlap_w_1 = np.sum(l_w_1)
    #     sum_all_w_1 = np.sum(d_neig_ew[token][0])
    #     w_ratio_1 = sum_overlap_w_1 / sum_all_w_1
    #     l_w_2 = []
    #     for neig in s_overlap_neig:
    #         w_2 = tkg_2.edges[token, neig]['weight']
    #         l_w_2.append(w_2)
    #     sum_overlap_w_2 = np.sum(l_w_2)
    #     sum_all_w_2 = np.sum(d_neig_ew[token][1])
    #     w_ratio_2 = sum_overlap_w_2 / sum_all_w_2
    #
    #     d_neig_overlap_weight_ratio[token] = [w_ratio_1, w_ratio_2]



    df_orig_embed_1 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name_1))
    df_orig_embed_2 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name_2))

    l_ws_all_orig = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))

        ds_name_1 = 'fp#cckg#ccoha1#' + tau_str
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1))

        ds_name_2 = 'fp#cckg#ccoha2#' + tau_str
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2))

        np_eval_adj_embed_1 = np.stack(df_adj_embed_1.loc[l_eval_token]['adj_token_embed'].to_list())
        np_eval_orig_embed_1 = np.stack(df_orig_embed_1.loc[l_eval_token]['token_embed'].to_list())
        np_eval_adj_embed_2 = np.stack(df_adj_embed_2.loc[l_eval_token]['adj_token_embed'].to_list())
        np_eval_orig_embed_2 = np.stack(df_orig_embed_2.loc[l_eval_token]['token_embed'].to_list())

        d_neig_adj_embed = dict()
        d_neig_orig_embed = dict()
        l_ws_all_orig_per_tau = []
        for token in l_eval_token:
            token_orig_embed_1 = np.asarray(df_orig_embed_1.loc[token]['token_embed']).reshape(1, -1)
            token_orig_embed_2 = np.asarray(df_orig_embed_2.loc[token]['token_embed']).reshape(1, -1)

            token_adj_embed_1 = np.asarray(df_adj_embed_1.loc[token]['adj_token_embed']).reshape(1, -1)
            token_adj_embed_2 = np.asarray(df_adj_embed_2.loc[token]['adj_token_embed']).reshape(1, -1)

            l_neig_1 = d_neig[token][0]
            l_neig_2 = d_neig[token][1]

            l_neig_w_1 = d_neig_ew[token][0]
            l_neig_w_2 = d_neig_ew[token][1]

            np_neig_adj_embed_1 = np.stack(df_adj_embed_1.loc[l_neig_1]['adj_token_embed'].to_list())
            np_neig_adj_embed_2 = np.stack(df_adj_embed_2.loc[l_neig_2]['adj_token_embed'].to_list())
            d_neig_adj_embed[token] = [np_neig_adj_embed_1, np_neig_adj_embed_2]

            np_neig_orig_embed_1 = np.stack(df_orig_embed_1.loc[l_neig_1]['token_embed'].to_list())
            np_neig_orig_embed_2 = np.stack(df_orig_embed_2.loc[l_neig_2]['token_embed'].to_list())
            d_neig_orig_embed[token] = [np_neig_orig_embed_1, np_neig_orig_embed_2]

            # ws_all_orig = ws_dist(token_orig_embed_1, token_orig_embed_2, np_neig_orig_embed_1, np_neig_orig_embed_2,
            #                       l_neig_w_1, l_neig_w_2, tau)
            # ws_all_adj = ws_dist(token_adj_embed_1, token_adj_embed_2, np_neig_adj_embed_1, np_neig_adj_embed_2,
            #                       l_neig_w_1, l_neig_w_2, tau)

            curv_adj_score = curv(token_adj_embed_1, token_adj_embed_2, np_neig_adj_embed_1, np_neig_adj_embed_2,
                                  l_neig_w_1, l_neig_w_2)
            l_ws_all_orig_per_tau.append(curv_adj_score)
        l_ws_all_orig.append(l_ws_all_orig_per_tau)
        print()
    print()


def ws_dist(token_embed_1, token_embed_2, neig_embed_1, neig_embed_2, neig_w_1, neig_w_2, tau):
    np_embed_1 = np.concatenate([token_embed_1, neig_embed_1])
    np_embed_2 = np.concatenate([token_embed_2, neig_embed_2])

    np_w_1 = np.concatenate([[1.0 - tau], tau * neig_w_1])
    np_w_2 = np.concatenate([[1.0 - tau], tau * neig_w_2])

    np_joint_w = np.matmul(np.transpose(np_w_1.reshape(1, -1)), np_w_2.reshape(1, -1))
    np_dc = 1.0 - np.matmul(np_embed_1, np.transpose(np_embed_2))

    ws_dis = np.sum(np_joint_w * np_dc)
    return ws_dis


def curv(token_embed_1, token_embed_2, neig_embed_1, neig_embed_2, neig_w_1, neig_w_2):
    np_joint_w = np.matmul(np.transpose(neig_w_1.reshape(1, -1)), neig_w_2.reshape(1, -1))
    np_dc = 1.0 - np.matmul(neig_embed_1, np.transpose(neig_embed_2))
    ws_neig = np.sum(np_joint_w * np_dc)

    ws_neig = np.divide(ws_neig, np_joint_w.shape[0] * np_joint_w.shape[1])

    # token_dc = 1.0 - np.matmul(token_embed_1.reshape(1, -1), np.transpose(token_embed_2.reshape(1, -1)))
    # token_dc = np.sum(token_dc)

    # k = 1.0 - ws_neig / token_dc
    k = ws_neig
    return k


def compare_token_node_embed(l_tau):
    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    ds_name_1 = 'reg_cpu#cckg#ccoha1#'
    ds_name_2 = 'reg_cpu#cckg#ccoha2#'

    l_np_cos_1_hat_vs_2 = []
    l_np_cos_2_hat_vs_1 = []
    for tau in l_tau:
        # tau = 0.0
        tau_str = str(np.round(tau, decimals=1))
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1 + tau_str))
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2 + tau_str))

        print()
        continue
        np_adj_embed_1_idx = df_adj_embed_1.index.get_indexer(l_eval_token)
        np_adj_embed_2_idx = df_adj_embed_2.index.get_indexer(l_eval_token)

        np_adj_embed_1 = np.stack(df_adj_embed_1['adj_token_embed'].to_list())
        np_adj_embed_2 = np.stack(df_adj_embed_2['adj_token_embed'].to_list())

        u_1, s_1, v_1 = np.linalg.svd(np_adj_embed_1, full_matrices=False)
        u_2, s_2, v_2 = np.linalg.svd(np_adj_embed_2, full_matrices=False)
        b_1 = np.matmul(np.diag(s_1), v_1.T)
        b_2 = np.matmul(np.diag(s_2), v_2.T)

        p_1_to_2 = np.matmul(b_2.T, np.linalg.inv(b_1.T))
        p_2_to_1 = np.matmul(b_1.T, np.linalg.inv(b_2.T))

        np_adj_embed_1_hat = np.matmul(p_1_to_2, np_adj_embed_1[np_adj_embed_1_idx].T).T
        np_adj_embed_2_hat = np.matmul(p_2_to_1, np_adj_embed_2[np_adj_embed_2_idx].T).T

        np_cos_1_hat_vs_2 = np.einsum('ij..., ij...->i',
                                      preprocessing.normalize(np_adj_embed_1_hat),
                                      np_adj_embed_2[np_adj_embed_2_idx])
        l_np_cos_1_hat_vs_2.append(np_cos_1_hat_vs_2)
        np_cos_2_hat_vs_1 = np.einsum('ij..., ij...->i',
                                      preprocessing.normalize(np_adj_embed_2_hat),
                                      np_adj_embed_1[np_adj_embed_1_idx])
        l_np_cos_2_hat_vs_1.append(np_cos_2_hat_vs_1)

        print()


def compare_token_by_intersection(l_tau):
    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    ds_name_1 = 'reg_cpu#diff#cckg#ccoha1#'
    ds_name_2 = 'reg_cpu#diff#cckg#ccoha2#'
    ds_name_int = 'reg_cpu#cckg#ccoha1#INT#cckg#ccoha2#'

    l_np_cos_1_hat_vs_2 = []
    l_np_cos_2_hat_vs_1 = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1 + tau_str))
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2 + tau_str))
        df_adj_embed_int = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_int + tau_str))

        print()


def compare_token_by_relative_diff(l_tau):
    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    # ds_name_1 = 'cckg#ccoha1'
    # ds_name_2 = 'cckg#ccoha2'
    #
    # tkg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1))
    # tkg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_2))

    ds_name_1 = 'reg_cpu#cckg#ccoha1#'
    ds_name_2 = 'reg_cpu#cckg#ccoha2#'

    l_cos_1_vs_share = []
    l_cos_2_vs_share = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1 + tau_str))
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2 + tau_str))

        l_shared_node = list(set(df_adj_embed_1.index).intersection(set(df_adj_embed_2.index)))

        np_shared_embed_1 = np.stack(df_adj_embed_1.loc[l_shared_node]['adj_token_embed'].to_list())
        np_shared_embed_2 = np.stack(df_adj_embed_2.loc[l_shared_node]['adj_token_embed'].to_list())

        d_cos_1_vs_share = dict()
        d_cos_2_vs_share = dict()
        for token in l_eval_token:
            np_token_embed_1 = df_adj_embed_1.loc[token]['adj_token_embed']
            np_token_embed_1 = np.stack([np_token_embed_1] * len(l_shared_node))

            np_token_embed_2 = df_adj_embed_2.loc[token]['adj_token_embed']
            np_token_embed_2 = np.stack([np_token_embed_2] * len(l_shared_node))

            cos_1_vs_share = np.einsum('ij..., ij...->i', np_shared_embed_1, np_token_embed_1)
            d_cos_1_vs_share[token] = cos_1_vs_share
            cos_2_vs_share = np.einsum('ij..., ij...->i', np_shared_embed_2, np_token_embed_2)
            d_cos_2_vs_share[token] = cos_2_vs_share
            print()

        l_cos_1_vs_share.append(d_cos_1_vs_share)
        l_cos_2_vs_share.append(d_cos_2_vs_share)

        print()


def compare_token_by_squeeze(l_tau):
    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    # ds_name_1 = 'cckg#ccoha1'
    # ds_name_2 = 'cckg#ccoha2'
    #
    # tkg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1))
    # tkg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_2))

    ds_name_1 = 'squeeze_reg_cpu#cckg#ccoha1#'
    ds_name_2 = 'reg_cpu#cckg#ccoha2#'

    l_cos_1_vs_share = []
    l_cos_2_vs_share = []
    l_cos_2_vs_2 = []
    for tau in l_tau:
        tau_str = str(np.round(tau, decimals=1))
        df_adj_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_1 + tau_str))
        df_adj_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name_2 + tau_str))

        l_shared_node = list(set(df_adj_embed_1.index).intersection(set(df_adj_embed_2.index)))

        np_shared_embed_1 = np.stack(df_adj_embed_1.loc[l_shared_node]['adj_token_embed'].to_list())
        np_shared_embed_2 = np.stack(df_adj_embed_2.loc[l_shared_node]['adj_token_embed'].to_list())

        d_cos_1_vs_share = dict()
        d_cos_2_vs_share = dict()
        d_cos_1_vs_2 = dict()

        print('tau = %s' % tau)
        for token in l_eval_token:
            np_token_embed_1 = df_adj_embed_1.loc[token]['adj_token_embed']
            np_token_embed_2 = df_adj_embed_2.loc[token]['adj_token_embed']

            cos_1_vs_2 = np.dot(np_token_embed_2, np_token_embed_2)
            d_cos_1_vs_2[token] = cos_1_vs_2

            np_token_embed_1 = np.stack([np_token_embed_1] * len(l_shared_node))
            np_token_embed_2 = np.stack([np_token_embed_2] * len(l_shared_node))

            cos_1_vs_share = np.einsum('ij..., ij...->i', np_shared_embed_1, np_token_embed_1)
            d_cos_1_vs_share[token] = cos_1_vs_share
            cos_2_vs_share = np.einsum('ij..., ij...->i', np_shared_embed_2, np_token_embed_2)
            d_cos_2_vs_share[token] = cos_2_vs_share
            print(token, cos_1_vs_2, np.mean(np.abs((1.0 - cos_1_vs_share) / 2.0 - (1.0 - cos_2_vs_share) / 2.0)))

        l_cos_1_vs_share.append(d_cos_1_vs_share)
        l_cos_2_vs_share.append(d_cos_2_vs_share)
        l_cos_2_vs_2.append(d_cos_1_vs_2)

        print()

################################################################################
#   TEST ONLY START
################################################################################
def test_only():
    cckg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('cckg#ccoha1'))
    cckg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format('cckg#ccoha2'))

    l_eval_token = []
    with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        for ln in in_fd:
            l_eval_token.append(ln.strip())
        in_fd.close()

    df_adj_token_embed_1 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('reg_cpu#cckg#ccoha1#0.0'))
    df_adj_token_embed_2 = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format('reg_cpu#cckg#ccoha2#0.0'))

    df_orig_token_embed_1 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('cckg#ccoha1'))
    df_orig_token_embed_2 = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('cckg#ccoha2'))

    cckg_shared_nodes = set(cckg_1.nodes()).intersection(set(cckg_2.nodes()))
    cckg_shared_nodes = cckg_shared_nodes.difference(set(l_eval_token))
    cckg_shared_nodes = cckg_shared_nodes.intersection(set(df_adj_token_embed_1.index))
    cckg_shared_nodes = cckg_shared_nodes.intersection(set(df_adj_token_embed_2.index))

    l_token_info = []
    for token in l_eval_token:
        token_deg_1 = cckg_1.degree(token)
        token_deg_2 = cckg_2.degree(token)

        token_neig_1 = set(cckg_1.neighbors(token))
        token_neig_2 = set(cckg_2.neighbors(token))

        shared_neig = token_neig_1.intersection(token_neig_2)
        cnt_shared_neig = len(shared_neig)

        avg_shared_neig_deg_1 = np.mean([item[1] for item in list(cckg_1.degree(shared_neig))])
        avg_shared_neig_deg_2 = np.mean([item[1] for item in list(cckg_2.degree(shared_neig))])

        avg_exclusive_neig_deg_1 = np.mean([item[1] for item in list(cckg_1.degree(token_neig_1.difference(shared_neig)))])
        avg_exclusive_neig_deg_2 = np.mean([item[1] for item in list(cckg_2.degree(token_neig_2.difference(shared_neig)))])

        avg_shared_neig_deg_diff = np.mean([np.abs(cckg_1.degree(token) - cckg_2.degree(token)) for token in shared_neig])

        avg_shared_neig_weight_1 = np.mean([cckg_1.edges[token, neig]['weight'] for neig in shared_neig])
        avg_shared_neig_weight_2 = np.mean([cckg_2.edges[token, neig]['weight'] for neig in shared_neig])

        avg_shared_neig_weight_diff = np.mean([np.abs(cckg_1.edges[token, neig]['weight'] - cckg_2.edges[token, neig]['weight']) for neig in shared_neig])

        avg_exclusive_neig_weight_1 = np.mean([cckg_1.edges[token, neig]['weight'] for neig in token_neig_1.difference(shared_neig)])
        avg_exclusive_neig_weight_2 = np.mean([cckg_2.edges[token, neig]['weight'] for neig in token_neig_2.difference(shared_neig)])

        avg_token_cos_neig_1 = np.mean(np.dot(df_adj_token_embed_1.loc[token]['adj_token_embed'],
                                              np.stack(df_adj_token_embed_1.loc[token_neig_1]['adj_token_embed'].to_list()).T))
        avg_token_cos_neig_2 = np.mean(np.dot(df_adj_token_embed_2.loc[token]['adj_token_embed'],
                                              np.stack(df_adj_token_embed_2.loc[token_neig_2]['adj_token_embed'].to_list()).T))

        avg_token_cos_shared_neig_1 = np.mean(np.dot(df_adj_token_embed_1.loc[token]['adj_token_embed'],
                                                     np.stack(df_adj_token_embed_1.loc[shared_neig]['adj_token_embed'].to_list()).T))
        avg_token_cos_shared_neig_2 = np.mean(np.dot(df_adj_token_embed_2.loc[token]['adj_token_embed'],
                                                     np.stack(df_adj_token_embed_2.loc[shared_neig]['adj_token_embed'].to_list()).T))

        avg_token_cos_exclusive_neig_1 = np.mean(np.dot(df_adj_token_embed_1.loc[token]['adj_token_embed'],
                                                        np.stack(df_adj_token_embed_1.loc[token_neig_1.difference(shared_neig)]['adj_token_embed'].to_list()).T))
        avg_token_cos_exclusive_neig_2 = np.mean(np.dot(df_adj_token_embed_2.loc[token]['adj_token_embed'],
                                                        np.stack(df_adj_token_embed_2.loc[token_neig_2.difference(shared_neig)]['adj_token_embed'].to_list()).T))

        avg_orig_token_cos_neig_1 = np.mean(np.dot(df_orig_token_embed_1.loc[token]['token_embed'],
                                              np.stack(df_orig_token_embed_1.loc[token_neig_1]['token_embed'].to_list()).T))
        avg_orig_token_cos_neig_2 = np.mean(np.dot(df_orig_token_embed_2.loc[token]['token_embed'],
                                              np.stack(df_orig_token_embed_2.loc[token_neig_2]['token_embed'].to_list()).T))

        avg_orig_token_cos_shared_neig_1 = np.mean(np.dot(df_orig_token_embed_1.loc[token]['token_embed'],
                                                     np.stack(df_orig_token_embed_1.loc[shared_neig]['token_embed'].to_list()).T))
        avg_orig_token_cos_shared_neig_2 = np.mean(np.dot(df_orig_token_embed_2.loc[token]['token_embed'],
                                                     np.stack(df_orig_token_embed_2.loc[shared_neig]['token_embed'].to_list()).T))

        avg_orig_token_cos_exclusive_neig_1 = np.mean(np.dot(df_orig_token_embed_1.loc[token]['token_embed'],
                                                        np.stack(df_orig_token_embed_1.loc[token_neig_1.difference(shared_neig)]['token_embed'].to_list()).T))
        avg_orig_token_cos_exclusive_neig_2 = np.mean(np.dot(df_orig_token_embed_2.loc[token]['token_embed'],
                                                        np.stack(df_orig_token_embed_2.loc[token_neig_2.difference(shared_neig)]['token_embed'].to_list()).T))

        avg_token_cos_shared_nodes_1 = np.mean(np.dot(df_adj_token_embed_1.loc[token]['adj_token_embed'],
                                                        np.stack(df_adj_token_embed_1.loc[cckg_shared_nodes]['adj_token_embed'].to_list()).T))
        avg_token_cos_shared_nodes_2 = np.mean(np.dot(df_adj_token_embed_2.loc[token]['adj_token_embed'],
                                                        np.stack(df_adj_token_embed_2.loc[cckg_shared_nodes]['adj_token_embed'].to_list()).T))

        avg_orig_token_cos_shared_nodes_1 = np.mean(np.dot(df_orig_token_embed_1.loc[token]['token_embed'],
                                                             np.stack(df_orig_token_embed_1.loc[cckg_shared_nodes]['token_embed'].to_list()).T))
        avg_orig_token_cos_shared_nodes_2 = np.mean(np.dot(df_orig_token_embed_2.loc[token]['token_embed'],
                                                             np.stack(df_orig_token_embed_2.loc[cckg_shared_nodes]['token_embed'].to_list()).T))

        l_token_info.append((token, token_deg_1, token_deg_2, cnt_shared_neig, avg_shared_neig_deg_1, avg_shared_neig_deg_2,
                             avg_exclusive_neig_deg_1, avg_exclusive_neig_deg_2, avg_shared_neig_deg_diff,
                             avg_shared_neig_weight_1, avg_shared_neig_weight_2, avg_exclusive_neig_weight_1,
                             avg_exclusive_neig_weight_2, avg_shared_neig_weight_diff, avg_token_cos_neig_1,
                             avg_token_cos_neig_2, avg_token_cos_shared_neig_1, avg_token_cos_shared_neig_2,
                             avg_token_cos_exclusive_neig_1, avg_token_cos_exclusive_neig_2, avg_orig_token_cos_neig_1,
                             avg_orig_token_cos_neig_2, avg_orig_token_cos_shared_neig_1, avg_orig_token_cos_shared_neig_2,
                             avg_orig_token_cos_exclusive_neig_1, avg_orig_token_cos_exclusive_neig_2,
                             avg_token_cos_shared_nodes_1, avg_token_cos_shared_nodes_2,
                             avg_orig_token_cos_shared_nodes_1, avg_orig_token_cos_shared_nodes_2))
    df_token_into = pd.DataFrame(l_token_info, columns=['token', 'token_deg_1', 'token_deg_2', 'cnt_shared_neig',
                                                        'avg_shared_neig_deg_1', 'avg_shared_neig_deg_2',
                                                        'avg_exclusive_neig_deg_1', 'avg_exclusive_neig_deg_2',
                                                        'avg_shared_neig_deg_diff', 'avg_shared_neig_weight_1',
                                                        'avg_shared_neig_weight_2', 'avg_exclusive_neig_weight_1',
                                                        'avg_exclusive_neig_weight_2', 'avg_shared_neig_weight_diff',
                                                        'avg_token_cos_neig_1', 'avg_token_cos_neig_2',
                                                        'avg_token_cos_shared_neig_1', 'avg_token_cos_shared_neig_2',
                                                        'avg_token_cos_exclusive_neig_1', 'avg_token_cos_exclusive_neig_2',
                                                        'avg_orig_token_cos_neig_1', 'avg_orig_token_cos_neig_2',
                                                        'avg_orig_token_cos_shared_neig_1', 'avg_orig_token_cos_shared_neig_2',
                                                        'avg_orig_token_cos_exclusive_neig_1', 'avg_orig_token_cos_exclusive_neig_2',
                                                        'avg_token_cos_shared_nodes_1', 'avg_token_cos_shared_nodes_2',
                                                        'avg_orig_token_cos_shared_nodes_1', 'avg_orig_token_cos_shared_nodes_2'])
    df_token_into = df_token_into.set_index('token')
    df_token_into.to_excel('cctg_ccoha.xlsx')
    print(df_token_into)



################################################################################
#   TEST ONLY END
################################################################################

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'token_embed_collect':
        l_ds_name = ['ccoha1', 'ccoha2']
        l_tau = np.arange(0.0, 0.9, 0.1)
        l_tokens = []
        with open(global_settings.g_token_list_file, 'r') as in_fd:
            for ln in in_fd:
                l_tokens.append(ln.strip())
            in_fd.close()
        # regular M, no POS removal.
        # task_name = 'M_noposrm'
        # regular M, with POS removal, with relative adjusted embedding update.
        task_name = 'M_posrm_rel'
        collect_token_embed_by_tau(l_ds_name, l_tau, l_tokens, task_name)
    elif cmd == 'compare_token_embeds':
        # task_name = 'M_noposrm'
        # task_name = 'M_posrm_rel'
        l_tau = np.arange(0.0, 1.0, 0.1)
        # ordered_tokens = []
        # with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
        #     for ln in in_fd:
        #         ordered_tokens.append(ln.strip())
        #     in_fd.close()
        # compare_token_embeds(task_name, l_tau, ordered_tokens)

        # fp_cos_threshold = 0.8
        # fp_deg_threshold = 1000
        # compare_token_with_fixed_points(l_tau, fp_cos_threshold, fp_deg_threshold)

        # compare_token_by_fp(l_tau, True, True)

        # compare_token_by_adj_embed_merge()

        # compare_token_by_curv()

        # compare_token_node_embed(l_tau)

        # compare_token_by_intersection(l_tau)

        # compare_token_by_relative_diff(l_tau)

        compare_token_by_squeeze(l_tau)
    elif cmd == 'fp':
        # l_tau = np.arange(0.0, 1.0, 0.1)
        # fp_cos_threshold = float(sys.argv[2])
        # fp_deg_threshold = int(sys.argv[3])
        # find_fixed_points(l_tau, fp_cos_threshold, fp_deg_threshold)

        find_fixed_points_by_deg()
    elif cmd == 'test':
        test_only()
