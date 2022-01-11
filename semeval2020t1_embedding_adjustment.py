import logging
import math
import time
import sys
from os import walk
from copy import deepcopy

import scipy as sp
from scipy import sparse
import numpy as np
import torch as th
import torch.nn.functional as F
import networkx as nx
from sklearn import preprocessing
import pandas as pd

import semeval2020t1_global_settings as global_settings


def retrieve_orig_embed_for_tkg(tkg, df_token_embed, min_size, tune_ds_name=None):
    logging.debug('[retrieve_orig_and_rel_embed_for_tkg] starts.')
    timer_start = time.time()

    # df_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
    # logging.debug('[retrieve_orig_and_rel_embed_for_tkg] load in df_token_embed with %s recs in %s secs.'
    #               % (len(df_token_embed), time.time() - timer_start))

    l_sub_tkg = []
    for comp in nx.connected_components(tkg):
        if len(comp) < min_size:
            continue
        sub_tkg = tkg.subgraph(comp)
        l_sub_tkg.append(sub_tkg)

    l_orig_token_embed = []
    for sub_tkg in l_sub_tkg:
        sub_tkg_nodes = sub_tkg.nodes()
        df_token_embed = df_token_embed.reindex(sub_tkg_nodes)
        if tune_ds_name is None:
            np_orig_token_embed = np.stack(df_token_embed['token_embed'].to_list()).astype(np.float32)
        else:
            np_orig_token_embed = np.stack(df_token_embed['adj_token_embed'].to_list()).astype(np.float32)
        l_orig_token_embed.append((np_orig_token_embed, sub_tkg))
        logging.debug('[retrieve_orig_and_rel_embed_for_tkg] get np_orig_token_embed in %s secs: %s'
                      % (time.time() - timer_start, np_orig_token_embed.shape))
    logging.debug('[retrieve_orig_and_rel_embed_for_tkg] all done with %s np_orig_token_embed in %s secs'
                  % (len(l_orig_token_embed), time.time() - timer_start))
    return l_orig_token_embed


def update_adj_embed_with_relative_embed(tkg, df_token_embed, df_rel_token_embed, min_size):
    logging.debug('[update_orig_embed_with_relative_embed] starts.')
    timer_start = time.time()

    if df_rel_token_embed is None:
        return df_token_embed

    df_token_embed.update(df_rel_token_embed)

    l_sub_tkg = []
    for comp in nx.connected_components(tkg):
        if len(comp) < min_size:
            continue
        sub_tkg = tkg.subgraph(comp)
        l_sub_tkg.append(sub_tkg)

    l_token_embed = []
    for sub_tkg in l_sub_tkg:
        sub_tkg_nodes = sub_tkg.nodes()
        df_token_embed = df_token_embed.reindex(sub_tkg_nodes)
        np_adj_token_embed = np.stack(df_token_embed['adj_token_embed'].to_list())
        l_token_embed.append((np_adj_token_embed, sub_tkg))
        logging.debug('[update_adj_embed_with_relative_embed] get np_orig_token_embed in %s secs: %s'
                      % (time.time() - timer_start, np_adj_token_embed.shape))
    logging.debug('[update_adj_embed_with_relative_embed] all done with %s np_orig_token_embed in %s secs'
                  % (len(l_token_embed), time.time() - timer_start))
    return l_token_embed


def retrieve_orig_embed_for_pksg(pksg, ds_name):
    logging.debug('[retrieve_orig_embed_for_pksg] starts.')
    timer_start = time.time()

    df_phrase_embed = pd.read_pickle(global_settings.g_phrase_embed_file_fmt.format(ds_name))
    logging.debug('[retrieve_orig_embed_for_pksg] load in df_phrase_embed with %s recs in %s secs.'
                  % (len(df_phrase_embed), time.time() - timer_start))

    pksg_nodes = pksg.nodes()
    df_phrase_embed = df_phrase_embed.reindex(pksg_nodes)
    np_orig_ph_embed = np.stack(df_phrase_embed['phrase_embed'].to_list())
    logging.debug('[retrieve_orig_embed_for_pksg] all done with np_orig_ph_embed in %s secs: %s'
                  % (time.time() - timer_start, np_orig_ph_embed.shape))
    return np_orig_ph_embed


def compute_measure_matrix_from_pksg(pksg, tau):
    '''
    'pksg' needs to be connected
    '''
    logging.debug('[compute_measure_matrix_from_pksg] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(pksg)
    A = preprocessing.normalize(A, axis=1, norm='l1')
    A = (1 - tau) * A
    A = A.astype(np.float32)
    if tau == 0.0:
        logging.debug('[compute_measure_matrix_from_pksg] all done with M in %s secs: %s'
                      % (time.time() - timer_start, A.shape))
        return A
    T = sparse.diags([tau], shape=A.shape, dtype=np.float32)
    M = T + A
    logging.debug('[compute_measure_matrix_from_pksg] all done with M in %s secs: %s'
                  % (time.time() - timer_start, M.shape))
    return M


def compute_globally_normalized_upper_adjacency(pksg):
    logging.debug('[compute_globally_normalized_upper_adjacency] starts.')
    timer_start = time.time()

    A = nx.linalg.adjacency_matrix(pksg)
    A = sparse.triu(A)
    A = np.divide(A, np.sum(A))

    # sym_gnu_A_one_mask = np.divide(A, np.sum(A)).astype(np.bool).astype(np.int32)
    # sym_gnu_A_one_mask = sym_gnu_A_one_mask + sparse.diags([1] * A.shape[0], shape=A.shape)
    sym_gnu_A_one_mask = None

    logging.debug('[compute_globally_normalized_upper_adjacency] all done with A and sym_gnu_A_one_mask in %s secs: %s'
                  % (time.time() - timer_start, A.shape))
    return A, sym_gnu_A_one_mask


def custom_cosine_loss(t_adj_embed_t, t_adj_emebd_t_1, use_cuda=True):
    if use_cuda:
        t_adj_embed_t = t_adj_embed_t.to('cuda')
        t_adj_emebd_t_1 = t_adj_emebd_t_1.to('cuda')
    t_adj_embed_t_norm = th.nn.functional.normalize(t_adj_embed_t)
    t_adj_embed_t_1_norm = th.nn.functional.normalize(t_adj_emebd_t_1)
    # cos sim
    t_ret = th.einsum('ij..., ij...->i', t_adj_embed_t_norm, t_adj_embed_t_1_norm)
    if use_cuda:
        t_adj_embed_t = t_adj_embed_t.to('cpu')
        t_adj_emebd_t_1 = t_adj_emebd_t_1.to('cpu')
    # cos dis
    t_ret = 1.0 - t_ret
    # low bound
    t_ret[t_ret < 0.0] = 0.0
    # up bound
    t_ret[t_ret > 2.0] = 2.0
    t_ret = th.divide(t_ret, 2.0)
    # t_ret = th.sqrt(th.mean(th.square(t_ret)))
    t_ret = th.mean(t_ret)
    return t_ret


def sparse_dense_element_mul(t_sp, t_de):
    t_de_typename = th.typename(t_de).split('.')[-1]
    sparse_tensortype = getattr(th.sparse, t_de_typename)

    i = t_sp._indices()
    v = t_sp._values()
    dv = t_de[i[0,:], i[1,:]]
    return sparse_tensortype(i, v * dv, t_sp.size())


def graph_cosine_deviation(t_gnu_sp_A, t_sym_gnu_A_one_mask, t_adj_embed_t_1, use_cuda=True):
    if use_cuda:
        t_adj_embed_t_1 = t_adj_embed_t_1.to('cuda')
        t_gnu_sp_A = t_gnu_sp_A.to('cuda')
    t_adj_embed_t_1_norm = th.nn.functional.normalize(t_adj_embed_t_1)
    t_edge_cos = th.matmul(t_adj_embed_t_1_norm, th.transpose(t_adj_embed_t_1_norm, 0, 1))

    # cos dev on neighbors => minimize
    # if use_cuda:
    #     t_gnu_sp_A = t_gnu_sp_A.to('cuda')

    t_cos_dev = sparse_dense_element_mul(t_gnu_sp_A, t_edge_cos)
    t_cos_dev = th.div(th.sub(th.sparse.sum(t_gnu_sp_A), th.sparse.sum(t_cos_dev)), 2.0)

    # t_edge_cos = th.sub(1.0, t_edge_cos)
    # t_edge_cos[t_edge_cos < 0.0] = 0.0
    # t_edge_cos[t_edge_cos > 2.0] = 2.0
    # t_edge_cos = th.div(t_edge_cos, 2.0)
    # t_edge_cos = th.divide(t_edge_cos, 2.0)
    # if use_cuda:
    #     t_edge_cos_for_neig[t_edge_cos_for_neig <= 0.0] = th.exp(th.tensor(-8.0)).to('cuda')
    # else:
    #     t_edge_cos_for_neig[t_edge_cos_for_neig <= 0.0] = th.exp(th.tensor(-8.0))
    # t_edge_cos_for_neig = th.log(t_edge_cos_for_neig)

    # if use_cuda:
        # t_edge_cos = t_edge_cos.to('cpu')
        # t_gnu_sp_A = t_gnu_sp_A.to('cuda:1')

    # t_cos_dev = sparse_dense_element_mul(t_gnu_sp_A, t_edge_cos)
    # t_cos_dev = th.sparse.sum(t_cos_dev)

    # cos dev beyond neighbors => maximize
    t_gnu_sp_A = t_gnu_sp_A.type(th.bool).type(th.int32)
    val_cnt = t_gnu_sp_A.shape[0]**2 - 2 * len(t_gnu_sp_A._values()) - t_gnu_sp_A.shape[0]
    t_cos_dev_bynd_neig = th.div(th.add(val_cnt,
                                        th.sub(th.div(th.sub(th.sum(t_edge_cos), th.sum(th.diag(t_edge_cos))), 2.0),
                                               th.sparse.sum(sparse_dense_element_mul(t_gnu_sp_A, t_edge_cos)))),
                                 2.0)

    if use_cuda:
        t_adj_embed_t_1 = t_adj_embed_t_1.to('cpu')
        t_gnu_sp_A = t_gnu_sp_A.to('cpu')

    # t_edge_cos_for_bynd = 1.0 - t_edge_cos
    # t_edge_cos_for_bynd[t_edge_cos_for_bynd < 0.0] = 0.0
    # t_edge_cos_for_bynd[t_edge_cos_for_bynd > 2.0] = 2.0
    # t_edge_cos_for_bynd = th.divide(t_edge_cos_for_bynd, 2.0)
    # if use_cuda:
    #     t_edge_cos_for_bynd[t_edge_cos_for_bynd <= 0.0] = th.exp(th.tensor(-8.0)).to('cuda')
    # else:
    #     t_edge_cos_for_bynd[t_edge_cos_for_bynd <= 0.0] = th.exp(th.tensor(-8.0))
    # t_edge_cos_for_bynd = th.log(t_edge_cos_for_bynd)
    #
    # t_cos_dev_bynd_neig_rm = sparse_dense_element_mul(t_sym_gnu_A_one_mask, t_edge_cos_for_bynd)
    # t_cos_dev_bynd_neig_rm = th.sparse.sum(t_cos_dev_bynd_neig_rm)
    # t_cos_dev_all = th.sum(t_edge_cos_for_bynd)
    # t_cos_dev_bynd_neig = - (t_cos_dev_all - t_cos_dev_bynd_neig_rm)

    # t_edge_cos = th.sub(1.0, t_edge_cos)
    # t_edge_cos[t_edge_cos < 0.0] = 0.0
    # t_edge_cos[t_edge_cos > 1.0] = 1.0
    # t_gnu_sp_A = t_gnu_sp_A.type(th.bool).type(th.int32)
    # t_cos_dev_bynd_neig = th.sum(t_edge_cos) \
    #                       - 2 * th.sparse.sum(sparse_dense_element_mul(t_gnu_sp_A, t_edge_cos)) \
    #                       - th.sum(th.diag(t_edge_cos))

    # val_cnt = t_sym_gnu_A_one_mask.shape[0]**2 - len(t_sym_gnu_A_one_mask._values())
    # val_cnt = t_gnu_sp_A.shape[0]**2 - 2 * len(t_gnu_sp_A._values()) - t_gnu_sp_A.shape[0]
    if val_cnt > 0:
        t_cos_dev_bynd_neig = t_cos_dev_bynd_neig / val_cnt
    else:
        t_cos_dev_bynd_neig = th.tensor(0.0)

    # if use_cuda:
    #     t_cos_dev = t_cos_dev.to('cuda:0')
    #     t_cos_dev_bynd_neig = t_cos_dev_bynd_neig.to('cuda:0')

    return t_cos_dev, t_cos_dev_bynd_neig


def embed_entropy(t_adj_embed_t_1):
    t_adj_embed_t_1_norm = th.nn.functional.normalize(t_adj_embed_t_1, p=1)
    t_adj_embed_t_1_norm = th.softmax(t_adj_embed_t_1_norm, dim=1)
    t_adj_embed_t_1_log = th.log(t_adj_embed_t_1_norm)
    H = - th.sum(t_adj_embed_t_1_norm * t_adj_embed_t_1_log, dim=1)
    H = th.mean(H)
    return H


def adjust_embed(np_measure_mat, np_orig_embed, max_epoch, term_threshold, use_cuda=False, trans_measure_mat=False,
                 num_fp=None, use_rec_cos=True, gnu_sp_A=None, sym_gnu_A_one_mask=None, cos_dev_bynd_neig_weight=2.0,
                 use_embed_entropy=False, np_squeeze_embed=None, l_squeeze=None):
    logging.debug('[adjust_embed] starts.')
    logging.debug('[adjust_embed] parameters: max_epoch=%s, term_threshold=%s, use_cuda=%s, trans_measure_mat=%s, '
    'num_fp=%s, use_rec_cos=%s, gnu_sp_A=%s, cos_dev_bynd_neig_weight=%s, use_embed_entropy=%s' %
                  (max_epoch, term_threshold, use_cuda, trans_measure_mat, num_fp, use_rec_cos, gnu_sp_A is not None,
                   cos_dev_bynd_neig_weight, use_embed_entropy))

    timer_start = time.time()
    th.autograd.set_detect_anomaly(True)

    if gnu_sp_A is not None:
        logging.debug('[adjust_embed] use gnu_sp_A.')
        gnu_sp_A = sparse.coo_matrix(gnu_sp_A)
        t_sym_gnu_A_one_mask = sym_gnu_A_one_mask
        # sym_gnu_A_one_mask = sparse.coo_matrix(sym_gnu_A_one_mask)
        # if use_cuda:
        #     t_gnu_sp_A = th.sparse.FloatTensor(th.LongTensor(np.vstack((gnu_sp_A.row, gnu_sp_A.col))),
        #                                        th.FloatTensor(gnu_sp_A.data), th.Size(gnu_sp_A.shape)).to('cuda')
        #     # t_sym_gnu_A_one_mask = th.sparse.FloatTensor(th.LongTensor(np.vstack((sym_gnu_A_one_mask.row, sym_gnu_A_one_mask.col))),
        #     #                                              th.FloatTensor(sym_gnu_A_one_mask.data), th.Size(sym_gnu_A_one_mask.shape)).to('cuda')
        # else:
        t_gnu_sp_A = th.sparse.FloatTensor(th.LongTensor(np.vstack((gnu_sp_A.row, gnu_sp_A.col))),
                                           th.FloatTensor(gnu_sp_A.data), th.Size(gnu_sp_A.shape))
        # t_sym_gnu_A_one_mask = th.sparse.FloatTensor(th.LongTensor(np.vstack((sym_gnu_A_one_mask.row, sym_gnu_A_one_mask.col))),
        #                                              th.FloatTensor(sym_gnu_A_one_mask.data), th.Size(sym_gnu_A_one_mask.shape))
        t_gnu_sp_A.requires_grad = False
        # t_sym_gnu_A_one_mask.requires_grad = False

    if trans_measure_mat:
        np_measure_mat = np.transpose(np_measure_mat)

    np_measure_mat = sparse.coo_matrix(np_measure_mat)
    np_measure_mat_values = np_measure_mat.data
    np_measure_mat_indices = np.vstack((np_measure_mat.row, np_measure_mat.col))
    t_measure_mat_indices = th.LongTensor(np_measure_mat_indices)
    t_measure_mat_values = th.FloatTensor(np_measure_mat_values)
    # if use_cuda:
    #     t_measure_mat = th.sparse.FloatTensor(t_measure_mat_indices, t_measure_mat_values, th.Size(np_measure_mat.shape)).to('cuda')
    # else:
    t_measure_mat = th.sparse.FloatTensor(t_measure_mat_indices, t_measure_mat_values, th.Size(np_measure_mat.shape))
    t_measure_mat.requires_grad = False

    if np_squeeze_embed is not None and l_squeeze is not None and len(l_squeeze) > 0:
        t_squeeze_embed = th.from_numpy(np_squeeze_embed)
        t_squeeze_embed = th.nn.functional.normalize(t_squeeze_embed)
        t_squeeze_embed.requires_grad = False

    # if use_cuda:
    #     t_adj_embed = th.from_numpy(np_orig_embed).to('cuda')
    # else:
    t_adj_embed = th.from_numpy(np_orig_embed)
    t_adj_embed = th.nn.functional.normalize(t_adj_embed)
    t_adj_embed.requires_grad = True

    optimizer = th.optim.Adagrad([t_adj_embed])
    # optimizer = th.optim.SparseAdam([t_adj_embed])

    if num_fp is not None:
        if use_cuda:
            t_adj_embed_fp = th.from_numpy(np_orig_embed[-num_fp:]).to('cuda')
        else:
            t_adj_embed_fp = th.from_numpy(np_orig_embed[-num_fp:])
        t_adj_embed_fp = th.nn.functional.normalize(t_adj_embed_fp)
        t_adj_embed_fp.requires_grad = False

    # we compute the values in t_adj_embed from the backpropagation.

    # Potential choices: Adagrad > Adamax > AdamW > Adam
    # optimizer = th.optim.Adagrad([t_adj_embed])
    # SGD may have much higher GPU memory consumption and need more epochs to converge
    # optimizer = th.optim.SGD([t_adj_embed], lr=0.1)

    for i in range(max_epoch):
        if num_fp is not None:
            with th.no_grad():
                t_adj_embed[-num_fp:] = t_adj_embed_fp
        optimizer.zero_grad()
        if use_cuda:
            t_measure_mat = t_measure_mat.to('cuda')
            t_adj_embed = t_adj_embed.to('cuda')
        t_adj_embed_t_1 = th.matmul(t_measure_mat, t_adj_embed)
        if use_cuda:
            t_measure_mat = t_measure_mat.to('cpu')
            t_adj_embed = t_adj_embed.to('cpu')
            t_adj_embed_t_1 = t_adj_embed_t_1.to('cpu')

        if num_fp is None:
            if use_rec_cos:
                cos_loss = custom_cosine_loss(t_adj_embed, t_adj_embed_t_1, use_cuda)
            else:
                cos_loss = th.tensor(0.0)
        else:
            if use_rec_cos:
                cos_loss = custom_cosine_loss(t_adj_embed[:-num_fp], t_adj_embed_t_1[:-num_fp])
            else:
                cos_loss = th.tensor(0.0)

        if gnu_sp_A is not None:
            t_cos_dev_on_neig, t_cos_dev_bynd_neig = graph_cosine_deviation(t_gnu_sp_A, t_sym_gnu_A_one_mask,
                                                                            t_adj_embed_t_1, use_cuda)
        else:
            t_cos_dev_on_neig = th.tensor(0.0)
            t_cos_dev_bynd_neig = th.tensor(0.0)

        if use_embed_entropy:
            em_ent = embed_entropy(t_adj_embed)
        else:
            em_ent = th.tensor(0.0)

        if np_squeeze_embed is not None and l_squeeze is not None and len(l_squeeze) > 0:
            cos_squeeze_loss = custom_cosine_loss(t_squeeze_embed, t_adj_embed_t_1[l_squeeze], use_cuda)
        else:
            if use_cuda:
                cos_squeeze_loss = th.tensor(0.0).to('cuda')
            else:
                cos_squeeze_loss = th.tensor(0.0)

        total_loss = cos_loss + t_cos_dev_on_neig + cos_dev_bynd_neig_weight * t_cos_dev_bynd_neig + em_ent + cos_squeeze_loss
        logging.debug('[adjust_embed] epoch %s: cos_loss=%s, cos_squeeze_loss=%s, cos_dev_on_neig=%s, cos_dev_bynd_neig=%s, em_ent=%s,'
                      'total loss = %s'
                      % (i, cos_loss, cos_squeeze_loss, t_cos_dev_on_neig, t_cos_dev_bynd_neig, em_ent, total_loss))

        # if cos_loss <= term_threshold:
        #     t_adj_embed = th.nn.functional.normalize(t_adj_embed)
        #     logging.debug('[adjust_embed] done at epoch %s in %s secs.' % (i, time.time() - timer_start))
        #     return t_adj_embed

        if use_cuda:
            t_adj_embed = t_adj_embed.to('cuda')
        total_loss.backward()
        optimizer.step()
    t_adj_embed = th.nn.functional.normalize(t_adj_embed)
    logging.debug('[adjust_embed] done when out of epoches in %s secs: %s'
                  % (time.time() - timer_start, t_adj_embed.shape))
    return t_adj_embed


def adjust_embed_wrapper_for_tkg(ds_name, tkg_min_size, l_tau, max_epoch, term_threshold, build_cckt=False,
                                 use_cuda=True):
    logging.debug('[adjust_embed_wrapper_for_tkg] starts.')
    timer_start = time.time()

    if build_cckt:
        ds_name = 'cckg#' + ds_name
    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper_for_tkg] load in tkg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg)))

    df_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper_for_tkg] load in df_token_embed with %s recs in %s secs.'
                  % (len(df_token_embed), time.time() - timer_start))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg, df_token_embed, tkg_min_size)
    logging.debug('[adjust_embed_wrapper_for_tkg] load in l_orig_token_embed %s' % str(len(l_orig_token_embed)))

    for tau in l_tau:
        logging.debug('[adjust_embed_wrapper_for_tkg] running with tau = %s' % str(tau))

        for orig_token_embed, sub_tkg in l_orig_token_embed:
            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            t_adj_embed = adjust_embed(np_measure_mat, orig_token_embed, max_epoch, term_threshold, use_cuda)
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            l_ready = []
            for idx, token in enumerate(sub_tkg.nodes()):
                l_ready.append((token, np_adj_embed[idx]))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[adjust_embed_wrapper_for_tkg] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[adjust_embed_wrapper_for_tkg] all done in %s secs.' % str(time.time() - timer_start))


def array_reorder_by_another_array(a_1, a_2):
    logging.debug('[array_reorder_by_another_array] starts.')
    # a_1 contains a_2,
    # a_1, a_2 have no repeated elements respectively

    a_1_copy = deepcopy(a_1)
    a_2_copy = deepcopy(a_2)

    d_ele_2_orig_idx_in_1 = dict()
    for ele_2_idx_in_a_2, ele_2 in enumerate(a_2_copy):
        ele_2_orig_idx_in_1 = a_1_copy.index(ele_2)
        d_ele_2_orig_idx_in_1[ele_2_idx_in_a_2] = ele_2_orig_idx_in_1

    # ele_1's orig idx in 1 should be [len(a_1) - len(a_2), len(a_1) - 1]
    # d_ele_1_val_orig_new = dict()
    d_ele_1_idx_orig = dict()
    l_cand_tail_in_1_for_2 = a_1_copy[-len(a_2_copy):]
    # we replace ele_1 from the tail
    cand_ele_1_idx = len(a_1_copy) - 1
    while len(a_2_copy) > 0:
        ele_2 = a_2_copy.pop()
        ele_2_idx_in_a_2 = len(a_2_copy)
        ele_2_orig_idx_in_1 = d_ele_2_orig_idx_in_1[ele_2_idx_in_a_2]

        # if ele_2 is already in the tail replacement region of a_1,
        # simply log the index mapping.
        if ele_2 in l_cand_tail_in_1_for_2:
            ele_1_orig_idx_in_1 = a_1_copy.index(ele_2)
            d_ele_1_idx_orig[ele_1_orig_idx_in_1] = ele_2_orig_idx_in_1
        # otherwise, find the next available vacancy in the tail replacement region of a_1 for ele_2
        else:
            while True:
                if cand_ele_1_idx in d_ele_1_idx_orig:
                    cand_ele_1_idx -= 1
                    continue
                else:
                    break
            cand_ele_1 = a_1_copy[cand_ele_1_idx]
            # do the value swap
            a_1_copy[cand_ele_1_idx] = ele_2
            a_1_copy[ele_2_orig_idx_in_1] = cand_ele_1
            # log the mapping
            d_ele_1_idx_orig[cand_ele_1_idx] = ele_2_orig_idx_in_1
    logging.debug('[array_reorder_by_another_array] all done.')
    return a_1_copy, d_ele_1_idx_orig


def reorder_matrix_by_idx_mapping(sp_coo_mat, d_cur_to_old):
    logging.debug('[reorder_matrix_by_idx_mapping] starts.')
    sp_coo_mat = sparse.coo_matrix(sp_coo_mat)
    d_swap_map = dict()
    for cur in d_cur_to_old:
        d_swap_map[cur] = d_cur_to_old[cur]
        d_swap_map[d_cur_to_old[cur]] = cur

    new_row = []
    for row_idx in sp_coo_mat.row:
        if row_idx in d_swap_map:
            new_row.append(d_swap_map[row_idx])
        else:
            new_row.append(row_idx)

    new_col = []
    for col_idx in sp_coo_mat.col:
        if col_idx in d_swap_map:
            new_col.append(d_swap_map[col_idx])
        else:
            new_col.append(col_idx)

    sp_coo_mat_re = sparse.coo_matrix((sp_coo_mat.data, (new_row, new_col)), shape=sp_coo_mat.shape)
    logging.debug('[reorder_matrix_by_idx_mapping] all done.')
    return sp_coo_mat_re


def swap_matrix_back_to_orig(np_mat, d_cur_to_old):
    logging.debug('[swap_matrix_back_to_orig] starts.')

    for cur in d_cur_to_old:
        cur_row = deepcopy(np_mat[cur])
        old_row = deepcopy(np_mat[d_cur_to_old[cur]])
        np_mat[cur] = old_row
        np_mat[d_cur_to_old[cur]] = cur_row

    logging.debug('[swap_matrix_back_to_orig] done.')
    return np_mat


def adjust_embed_with_fp_wrapper_for_tkg(ds_name, tkg_min_size, l_tau, max_epoch, term_threshold, build_cckt=False,
                                         use_cuda=True):
    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] starts.')
    timer_start = time.time()

    if build_cckt:
        ds_name = 'cckg#' + ds_name
    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] load in tkg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg)))

    df_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] load in df_token_embed with %s recs in %s secs.'
                  % (len(df_token_embed), time.time() - timer_start))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg, df_token_embed, tkg_min_size)
    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] load in l_orig_token_embed %s' % str(len(l_orig_token_embed)))

    l_shared_fp = []
    with open(global_settings.g_shared_fixed_points_by_deg_file, 'r') as in_fd:
        for ln in in_fd:
            l_shared_fp.append(ln.strip())
        in_fd.close()
    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] load in l_shared_fp %s' % str(len(l_shared_fp)))

    for tau in l_tau:
        logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] running with tau = %s' % str(tau))

        for orig_token_embed, sub_tkg in l_orig_token_embed:
            l_sub_tkg_node = list(sub_tkg.nodes())
            l_orig_tkg_node_idx = [l_sub_tkg_node.index(token) for token in l_sub_tkg_node]

            # reorder orig embeddings, put fp embeds at the end
            l_reordered_token, d_cur_idx_to_orig_idx = array_reorder_by_another_array(l_sub_tkg_node, l_shared_fp)
            l_reordered_token_idx = [l_sub_tkg_node.index(token) for token in l_reordered_token]
            reordered_orig_token_embed = orig_token_embed[l_reordered_token_idx]

            # reorder measure matrix as well
            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            np_measure_mat_rerodered = reorder_matrix_by_idx_mapping(np_measure_mat, d_cur_idx_to_orig_idx)

            t_adj_embed = adjust_embed(np_measure_mat_rerodered, reordered_orig_token_embed, max_epoch, term_threshold,
                                       use_cuda, len(l_shared_fp))
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            np_adj_embed = swap_matrix_back_to_orig(np_adj_embed, d_cur_idx_to_orig_idx)
            # np_adj_embed = np_adj_embed[l_orig_tkg_node_idx]
            l_ready = []
            for row_idx, adj_embed in enumerate(np_adj_embed):
                token = l_sub_tkg_node[l_orig_tkg_node_idx[row_idx]]
                l_ready.append((token, adj_embed))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = 'fp#' + ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[adjust_embed_with_fp_wrapper_for_tkg] all done in %s secs.' % str(time.time() - timer_start))


def adjust_embed_wrapper_for_tkg_with_regularization(ds_name, orig_embed_ds_name, tkg_min_size, l_tau, max_epoch, term_threshold,
                                                     build_cckt=False, use_cuda=True, trans_measure_mat=False,
                                                     use_rec_cos=True, use_graph_dev_reg=True,
                                                     cos_dev_bynd_neig_weight=2.0, use_embed_entropy=False,
                                                     tune_ds_name=None, tune_tau_str=None):
    logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] starts.')
    timer_start = time.time()

    if build_cckt:
        ds_name = 'cckg#' + ds_name
    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] load in tkg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg)))

    if tune_ds_name is None:
        df_token_embed = pd.read_pickle(global_settings.g_token_embed_combined_file_fmt.format(orig_embed_ds_name))
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] load in df_token_embed with %s recs in %s secs.'
                      % (len(df_token_embed), time.time() - timer_start))
    else:
        l_eval_token = []
        with open(global_settings.g_ordered_token_list_file, 'r') as in_fd:
            for ln in in_fd:
                l_eval_token.append(ln.strip())
            in_fd.close()

        ds_name = 'reg_cpu#' + ds_name + '#' + tune_tau_str
        df_alt_orig_train = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(ds_name))
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] load in df_alt_orig_train with %s recs in %s secs.'
                      % (len(df_alt_orig_train), time.time() - timer_start))
        if build_cckt:
            tune_ds_name = 'reg_cpu#cckg#' + tune_ds_name + '#' + tune_tau_str
        df_alt_orig_tune = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(tune_ds_name))
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] load in df_alt_orig_tune with %s recs in %s secs.'
                      % (len(df_alt_orig_tune), time.time() - timer_start))

        diff_tokens_train = set(df_alt_orig_train.index).difference(set(l_eval_token))
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] get diff_tokens_train with %s recs in %s secs.'
                      % (len(diff_tokens_train), time.time() - timer_start))
        df_token_embed = pd.concat([df_alt_orig_train.loc[diff_tokens_train], df_alt_orig_tune.loc[l_eval_token]])
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] get merged alt token embed with %s recs in %s secs.'
                      % (len(df_token_embed), time.time() - timer_start))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg, df_token_embed, tkg_min_size, tune_ds_name)
    logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] load in l_orig_token_embed %s'
                  % str(len(l_orig_token_embed)))

    for tau in l_tau:
        logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] running with tau = %s' % str(tau))

        for orig_token_embed, sub_tkg in l_orig_token_embed:
            if use_graph_dev_reg:
                gnu_sp_A, sym_gnu_A_one_mask = compute_globally_normalized_upper_adjacency(sub_tkg)
                logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] get gnu_sp_A and sym_gnu_A_one_mask '
                              'in %s secs: %s' % (time.time() - timer_start, gnu_sp_A.shape))
            else:
                gnu_sp_A = None
                sym_gnu_A_one_mask = None

            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            t_adj_embed = adjust_embed(np_measure_mat, orig_token_embed, max_epoch, term_threshold, use_cuda,
                                       trans_measure_mat=trans_measure_mat, num_fp=None, use_rec_cos=use_rec_cos,
                                       gnu_sp_A=gnu_sp_A, sym_gnu_A_one_mask=sym_gnu_A_one_mask,
                                       cos_dev_bynd_neig_weight=cos_dev_bynd_neig_weight,
                                       use_embed_entropy=use_embed_entropy, np_squeeze_embed=None, l_squeeze=None)
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            l_ready = []
            for idx, token in enumerate(sub_tkg.nodes()):
                l_ready.append((token, np_adj_embed[idx]))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = 'reg_cpu#' + ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[adjust_embed_wrapper_for_tkg_with_regularization] all done in %s secs.' % str(time.time() - timer_start))


def adjust_embed_wrapper_for_tkg_with_squeeze(ds_name, squeeze_ds_name, orig_embed_ds_name, tkg_min_size, l_tau,
                                              max_epoch, term_threshold, build_cckt=True, use_cuda=True,
                                              trans_measure_mat=False,
                                              use_rec_cos=True, use_graph_dev_reg=True,
                                              cos_dev_bynd_neig_weight=2.0, use_embed_entropy=False):
    logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] starts.')
    timer_start = time.time()

    if build_cckt:
        ds_name = 'cckg#' + ds_name
        squeeze_ds_name = 'reg_cpu#cckg#' + squeeze_ds_name

    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] load in tkg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg)))

    df_token_embed = pd.read_pickle(global_settings.g_token_embed_combined_file_fmt.format(orig_embed_ds_name))
    logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] load in df_token_embed with %s recs in %s secs.'
                  % (len(df_token_embed), time.time() - timer_start))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg, df_token_embed, tkg_min_size, None)
    logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] load in l_orig_token_embed %s'
                  % str(len(l_orig_token_embed)))

    for tau in l_tau:
        logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] running with tau = %s' % str(tau))

        squeeze_ds_name_per_tau = squeeze_ds_name + '#' + str(np.round(tau, decimals=1))
        df_squeeze_embed = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt.format(squeeze_ds_name_per_tau))
        l_squeeze_token = list(df_squeeze_embed.index)
        logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] load in df_squeeze_embed with %s recs in %s secs.'
                      % (len(df_squeeze_embed), time.time() - timer_start))

        l_squeeze_info = []
        for i in range(len(l_orig_token_embed)):
            orig_token_embed, sub_tkg = l_orig_token_embed[i]
            l_sub_tkg_node = list(sub_tkg.nodes())
            # l_squeeze_token_sub_tkg = [token for token in l_squeeze_token if token in l_sub_tkg_node]
            l_squeeze_token_sub_tkg = list(set(l_squeeze_token).intersection(set(l_sub_tkg_node)))
            np_squeeze_embed_sub_tkg = np.stack(df_squeeze_embed.loc[l_squeeze_token_sub_tkg]['adj_token_embed']).astype(np.float32)
            l_squeeze_token_sub_tkg_idx = [l_sub_tkg_node.index(token) for token in l_squeeze_token_sub_tkg]
            l_squeeze_info.append((np_squeeze_embed_sub_tkg, l_squeeze_token_sub_tkg_idx))

        for i in range(len(l_orig_token_embed)):
            orig_token_embed, sub_tkg = l_orig_token_embed[i]
            np_squeeze_embed_sub_tkg, l_squeeze_token_sub_tkg_idx = l_squeeze_info[i]

            if use_graph_dev_reg:
                gnu_sp_A, sym_gnu_A_one_mask = compute_globally_normalized_upper_adjacency(sub_tkg)
                logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] get gnu_sp_A and sym_gnu_A_one_mask '
                              'in %s secs: %s' % (time.time() - timer_start, gnu_sp_A.shape))
            else:
                gnu_sp_A = None
                sym_gnu_A_one_mask = None

            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            t_adj_embed = adjust_embed(np_measure_mat, orig_token_embed, max_epoch, term_threshold, use_cuda,
                                       trans_measure_mat=trans_measure_mat, num_fp=None, use_rec_cos=use_rec_cos,
                                       gnu_sp_A=gnu_sp_A, sym_gnu_A_one_mask=sym_gnu_A_one_mask,
                                       cos_dev_bynd_neig_weight=cos_dev_bynd_neig_weight,
                                       use_embed_entropy=use_embed_entropy, np_squeeze_embed=np_squeeze_embed_sub_tkg,
                                       l_squeeze=l_squeeze_token_sub_tkg_idx)
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            l_ready = []
            for idx, token in enumerate(sub_tkg.nodes()):
                l_ready.append((token, np_adj_embed[idx]))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = 'squeeze_reg_cpu#' + ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[adjust_embed_wrapper_for_tkg_with_squeeze] all done in %s secs.' % str(time.time() - timer_start))


def get_tkg_intersection(ds_name_1, ds_name_2):
    logging.debug('[get_tkg_intersection] starts.')
    tkg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1))
    tkg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_2))

    tkg_int = nx.Graph()
    node_int = set(tkg_1.nodes()).intersection(set(tkg_2.nodes()))
    tkg_int.add_nodes_from(node_int)

    for edge in tkg_1.edges(data=True):
        node_1 = edge[0]
        node_2 = edge[1]
        weight = edge[2]['weight']

        if tkg_2.has_edge(node_1, node_2):
            weight = np.min([weight, tkg_2.edges[node_1, node_2]['weight']])
            tkg_int.add_edge(node_1, node_2, weight=weight)
    nx.write_gpickle(tkg_int, global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1 + '#INT#' + ds_name_2))
    logging.debug('[get_tkg_intersection] all done: %s' % nx.info(tkg_int))


def get_tkg_difference(ds_name_1, ds_name_2, ds_name_int):
    logging.debug('[get_tkg_difference] starts.')

    tkg_1 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_1))
    tkg_2 = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_2))
    tkg_int = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name_int))

    for edge in tkg_int.edges(data=True):
        node_1 = edge[0]
        node_2 = edge[1]
        weight = edge[2]['weight']

        weight_1 = tkg_1.edges[node_1, node_2]['weight']
        weight_1 -= weight
        if weight_1 == 0:
            tkg_1.remove_edge(node_1, node_2)
        else:
            tkg_1.edges[node_1, node_2]['weight'] = weight_1

        weight_2 = tkg_2.edges[node_1, node_2]['weight']
        weight_2 -= weight
        if weight_2 == 0:
            tkg_2.remove_edge(node_1, node_2)
        else:
            tkg_2.edges[node_1, node_2]['weight'] = weight_2

    nx.write_gpickle(tkg_1, global_settings.g_merged_tw_tkg_file_fmt.format('diff#' + ds_name_1))
    nx.write_gpickle(tkg_2, global_settings.g_merged_tw_tkg_file_fmt.format('diff#' + ds_name_2))
    logging.debug('[get_tkg_difference] all done. diff tkg_1: %s, diff tkg_2: %s' % (nx.info(tkg_1), nx.info(tkg_2)))



def update_adjust_embed_wrapper_for_tkg(ds_name, rel_ds_name, tkg_min_size, l_tau, max_epoch, term_threshold,
                                        use_cuda=True):
    logging.debug('[update_adjust_embed_wrapper_for_tkg] starts.')
    timer_start = time.time()

    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    logging.debug('[update_adjust_embed_wrapper_for_tkg] load in tkg for %s in %s secs: %s'
                  % (ds_name, time.time() - timer_start, nx.info(tkg)))

    for tau in l_tau:
        logging.debug('[update_adjust_embed_wrapper_for_tkg] running with tau = %s' % str(tau))

        df_adj_token_embed = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt
                                            .format(ds_name + '#' + str(np.round(tau, decimals=1))))
        logging.debug('[update_adjust_embed_wrapper_for_tkg] load in df_adj_token_embed with %s recs in %s secs.'
                      % (len(df_adj_token_embed), time.time() - timer_start))
        df_rel_adj_token_embed = pd.read_pickle(global_settings.g_adj_token_embed_file_fmt
                                                .format(rel_ds_name + '#' + str(np.round(tau, decimals=1))))
        logging.debug('[update_adjust_embed_wrapper_for_tkg] load in df_rel_adj_token_embed with %s recs in %s secs.'
                      % (len(df_rel_adj_token_embed), time.time() - timer_start))

        l_merged_adj_token_embed = update_adj_embed_with_relative_embed(tkg, df_adj_token_embed,
                                                                        df_rel_adj_token_embed, tkg_min_size)
        logging.debug('[update_adjust_embed_wrapper_for_tkg] load in l_merged_adj_token_embed for %s sub_tkg'
                          % str(len(l_merged_adj_token_embed)))

        for orig_token_embed, sub_tkg in l_merged_adj_token_embed:
            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            t_adj_embed = adjust_embed(np_measure_mat, orig_token_embed, max_epoch, term_threshold, use_cuda)
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            l_ready = []
            for idx, token in enumerate(sub_tkg.nodes()):
                l_ready.append((token, np_adj_embed[idx]))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = ds_name + '#rel#' + rel_ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[update_adjust_embed_wrapper_for_tkg] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[update_adjust_embed_wrapper_for_tkg] all done in %s secs.' % str(time.time() - timer_start))


def adjust_embed_wrapper_for_pksg(ds_name, l_tau, max_epoch, term_threshold, use_cuda=True):
    logging.debug('[adjust_embed_wrapper] starts.')
    timer_start = time.time()

    pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[adjust_embed_wrapper] load in pksg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(pksg)))

    np_orig_ph_embed = retrieve_orig_embed_for_pksg(pksg, ds_name)

    for tau in l_tau:
        logging.debug('[adjust_embed_wrapper] running with tau = %s' % str(tau))
        np_measure_mat = compute_measure_matrix_from_pksg(pksg, tau)
        t_adj_embed = adjust_embed(np_measure_mat, np_orig_ph_embed, max_epoch, term_threshold, use_cuda)
        save_name = ds_name + '#' + str(tau)
        th.save(t_adj_embed, global_settings.g_adj_embed_file_fmt.format(save_name))
        logging.debug('[adjust_embed_wrapper] done with tau = %s in %s secs.' % (tau, time.time() - timer_start))

    logging.debug('[adjust_embed_wrapper] all done in %s secs.' % str(time.time() - timer_start))


def compute_adjusted_embedding_distributions(ds_name):
    logging.debug('[compute_adjusted_embedding_distributions] starts.')
    timer_start = time.time()

    l_adj_embed = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_adj_embed_folder):
        for filename in filenames:
            if filename[-3:] != '.pt' or filename[:13] != 'adj_ph_embed_':
                continue
            t_adj_embed = th.load(dirpath + filename)
            logging.debug('[compute_adjusted_embedding_distributions] load in %s: %s in %s secs.'
                          % (filename, t_adj_embed.shape, time.time() - timer_start))
            l_adj_embed.append(t_adj_embed.cpu().detach().numpy())
    logging.debug('[compute_adjusted_embedding_distributions] load in %s adjusted embeddings.' % str(len(l_adj_embed)))

    np_adj_embed_mean = np.mean(l_adj_embed, axis=0)
    np_adj_embed_std = np.std(l_adj_embed, axis=0)
    np_adj_embed_dist = np.stack([np_adj_embed_mean, np_adj_embed_std], axis=2)
    logging.debug('[compute_adjusted_embedding_distributions] done with np_adj_embed_dist: %s in %s secs.'
                  % (np_adj_embed_dist.shape, time.time() - timer_start))
    np.save(global_settings.g_adj_embed_dist_file_fmt.format(ds_name), np_adj_embed_dist)
    logging.debug('[compute_adjusted_embedding_distributions] all done in %s secs.' % str(time.time() - timer_start))


def extend_adj_embed_dist_to_samples(ds_name, sample_size, num_batch):
    logging.debug('[extend_adj_embed_dist_to_samples] starts.')
    timer_start = time.time()

    np_adj_embed_dist = np.load(global_settings.g_adj_embed_dist_file_fmt.format(ds_name))
    logging.debug('[extend_adj_embed_dist_to_samples] load in np_adj_embed_dist: %s in %s secs.'
                  % (np_adj_embed_dist.shape, time.time() - timer_start))

    pksg = nx.read_gpickle(global_settings.g_merged_tw_pksg_file_fmt.format(ds_name))
    logging.debug('[extend_adj_embed_dist_to_samples] load in pksg: %s in %s secs.'
                  % (nx.info(pksg), time.time() - timer_start))

    if pksg.number_of_nodes() != np_adj_embed_dist.shape[0]:
        raise Exception('[extend_adj_embed_dist_to_samples] np_adj_embed_dist does not match pksg.')

    l_pksg_nodes = list(pksg.nodes)

    rng = np.random.default_rng()
    batch_size = math.ceil(np_adj_embed_dist.shape[0] / num_batch)
    cnt = 0
    batch_id = 0
    l_np_dim_sample = []
    for ph_row_id, ph_dist in enumerate(np_adj_embed_dist):
        # ph_dist should be (300, 2)
        l_ph_dim_sample = []
        for ph_dim_dist in ph_dist:
            # ph_dim_dist should be (2,)
            ph_dim_sample = rng.normal(loc=ph_dim_dist[0], scale=ph_dim_dist[1], size=sample_size)
            ph_dim_sample = ph_dim_sample.astype(np.float32)
            l_ph_dim_sample.append(ph_dim_sample)
        np_dim_sample = np.stack(l_ph_dim_sample)
        l_np_dim_sample.append((l_pksg_nodes[ph_row_id], np_dim_sample))
        cnt += 1
        if cnt % batch_size == 0 and cnt >= batch_size:
            df_np_dim_sample = pd.DataFrame(l_np_dim_sample, columns=['pksg_node_id', 'np_dim_sample'])
            df_np_dim_sample.to_pickle(global_settings.g_adj_embed_samples_file_fmt
                                       .format(ds_name + '#' + str(sample_size) + '_' + str(batch_id)))

            logging.debug('[extend_adj_embed_dist_to_samples] batch %s df_np_dim_sample: %s done in %s secs.'
                          % (batch_id, len(df_np_dim_sample), time.time() - timer_start))
            batch_id += 1
            l_np_dim_sample = []
            df_np_dim_sample = None
        # if cnt % 10000 == 0 and cnt >= 10000:
        #     logging.debug('[extend_adj_embed_dist_to_samples] %s np_dim_sample done in %s secs.'
        #                   % (cnt, time.time() - timer_start))

    if len(l_np_dim_sample) > 0:
        df_np_dim_sample = pd.DataFrame(l_np_dim_sample, columns=['pksg_node_id', 'np_dim_sample'])
        df_np_dim_sample.to_pickle(global_settings.g_adj_embed_samples_file_fmt
                                   .format(ds_name + '#' + str(sample_size) + '_' + str(batch_id)))
        logging.debug('[extend_adj_embed_dist_to_samples] batch %s df_np_dim_sample: %s done in %s secs.'
                      % (batch_id, len(df_np_dim_sample), time.time() - timer_start))

    # np_adj_embed_sample = np.stack(l_np_dim_sample)
    # logging.debug('[extend_adj_embed_dist_to_samples] done with np_adj_embed_sample: %s in %s secs.'
    #               % (np_adj_embed_sample.shape, time.time() - timer_start))
    # np.save(global_settings.g_adj_embed_samples_file_fmt(ds_name + '#' + str(sample_size)))
    logging.debug('[extend_adj_embed_dist_to_samples] all done in %s secs.' % str(time.time() - timer_start))


def adj_embed_by_merged_corpus(l_ds_name, l_tau, max_epoch, term_threshold, build_cckt=True, use_cuda=True):
    logging.debug('[adj_embed_by_merged_corpus] starts.')
    timer_start = time.time()

    l_tkg = []
    l_df_token_embed = []
    for ds_name in l_ds_name:
        if build_cckt:
            ds_name = 'cckg#' + ds_name
        tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
        l_tkg.append(tkg)
        logging.debug('[adj_embed_by_merged_corpus] load in tkg in %s secs: %s'
                      % (time.time() - timer_start, nx.info(tkg)))

        df_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
        l_df_token_embed.append(df_token_embed)
        logging.debug('[adj_embed_by_merged_corpus] load in df_token_embed with %s recs in %s secs.'
                      % (len(df_token_embed), time.time() - timer_start))

    tkg_merge = l_tkg[0]
    for tkg in l_tkg[1:]:
        for edge in tkg.edges(data=True):
            node_1 = edge[0]
            node_2 = edge[1]
            weight = edge[2]['weight']
            if not tkg_merge.has_edge(node_1, node_2):
                tkg_merge.add_edge(node_1, node_2, weight=weight)
            else:
                tkg_merge.edges[node_1, node_2]['weight'] += weight
    logging.debug('[adj_embed_by_merged_corpus] done tkg_merge in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg_merge)))

    df_token_embed_merge = l_df_token_embed[0]
    l_ready = []
    for df_token_embed in l_df_token_embed[1:]:
        for token, token_embed_rec in df_token_embed.iterrows():
            if token not in df_token_embed_merge.index:
                l_ready.append((token, token_embed_rec['token_embed']))
        df_new = pd.DataFrame(l_ready, columns=['token', 'token_embed'])
        df_new = df_new.set_index('token')
        df_token_embed_merge = df_token_embed_merge.append(df_new)
    logging.debug('[adj_embed_by_merged_corpus] done df_token_embed_merge in %s secs: %s'
                  % (time.time() - timer_start, len(df_token_embed_merge)))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg_merge, df_token_embed_merge, 1000)
    logging.debug('[adj_embed_by_merged_corpus] load in l_orig_token_embed %s' % str(len(l_orig_token_embed)))

    for tau in l_tau:
        logging.debug('[adj_embed_by_merged_corpus] running with tau = %s' % str(tau))

        for orig_token_embed, sub_tkg in l_orig_token_embed:
            l_sub_tkg_node = list(sub_tkg.nodes())
            l_orig_tkg_node_idx = [l_sub_tkg_node.index(token) for token in l_sub_tkg_node]

            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            t_adj_embed = adjust_embed(np_measure_mat, orig_token_embed, max_epoch, term_threshold, use_cuda, None)
            np_adj_embed = t_adj_embed.cpu().detach().numpy()

            l_ready = []
            for row_idx, adj_embed in enumerate(np_adj_embed):
                token = l_sub_tkg_node[l_orig_tkg_node_idx[row_idx]]
                l_ready.append((token, adj_embed))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = 'merge#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[adj_embed_by_merged_corpus] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

####################################################################################################
#   Test Only Start
####################################################################################################
def gen_tkg_subset():
    logging.debug('[get_tkg_subset] starts.')
    ds_name = 'ccoha1'
    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))

    l_sub_nodes = list(tkg.nodes)[:100]
    sub_tkg = tkg.subgraph(l_sub_nodes)
    l_sorted_comp = sorted(list(nx.connected_components(sub_tkg)), key=lambda k: len(k), reverse=True)

    l_sub_nodes = l_sorted_comp[0]
    sub_tkg = tkg.subgraph(l_sub_nodes)
    l_sub_nodes = list(l_sub_nodes)
    l_rand_fp_idx = np.random.choice([i for i in range(len(l_sub_nodes))], 10, replace=False)
    l_rand_fp = [l_sub_nodes[i] for i in l_rand_fp_idx]

    nx.write_gpickle(sub_tkg, global_settings.g_test_tkg)
    logging.debug('[get_tkg_subset] sub_tkg: %s' % nx.info(sub_tkg))

    with open(global_settings.g_test_fp, 'w+') as out_fd:
        out_str = '\n'.join(l_rand_fp)
        out_fd.write(out_str)
        out_fd.close()

    df_orig_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
    df_orig_embed_sub_node = df_orig_token_embed.loc[l_sub_nodes]
    pd.to_pickle(df_orig_embed_sub_node, global_settings.g_test_orig_token_embed)

    logging.debug('[get_tkg_subset] all done.')


def test_adjust_embed_with_fp_wrapper_for_tkg(tkg_min_size, l_tau, max_epoch, term_threshold, use_cuda=True):
    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] starts.')
    timer_start = time.time()

    ds_name = 'ccoha1'
    tkg = nx.read_gpickle(global_settings.g_test_tkg)
    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] load in tkg in %s secs: %s'
                  % (time.time() - timer_start, nx.info(tkg)))

    df_token_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format('cckg#' + ds_name))
    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] load in df_token_embed with %s recs in %s secs.'
                  % (len(df_token_embed), time.time() - timer_start))

    l_orig_token_embed = retrieve_orig_embed_for_tkg(tkg, df_token_embed, tkg_min_size)
    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] load in l_orig_token_embed %s' % str(len(l_orig_token_embed)))

    l_shared_fp = []
    with open(global_settings.g_test_fp, 'r') as in_fd:
        for ln in in_fd:
            l_shared_fp.append(ln.strip())
        in_fd.close()
    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] load in l_shared_fp %s' % str(len(l_shared_fp)))

    for tau in l_tau:
        logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] running with tau = %s' % str(tau))

        for orig_token_embed, sub_tkg in l_orig_token_embed:
            l_sub_tkg_node = list(sub_tkg.nodes())
            l_orig_tkg_node_idx = [l_sub_tkg_node.index(token) for token in l_sub_tkg_node]

            # reorder orig embeddings, put fp embeds at the end
            l_reordered_token, d_cur_idx_to_orig_idx = array_reorder_by_another_array(l_sub_tkg_node, l_shared_fp)
            l_reordered_token_idx = [l_sub_tkg_node.index(token) for token in l_reordered_token]
            reordered_orig_token_embed = orig_token_embed[l_reordered_token_idx]

            # reorder measure matrix as well
            np_measure_mat = compute_measure_matrix_from_pksg(sub_tkg, tau)
            np_measure_mat_rerodered = reorder_matrix_by_idx_mapping(np_measure_mat, d_cur_idx_to_orig_idx)

            t_adj_embed = adjust_embed(np_measure_mat_rerodered, reordered_orig_token_embed, max_epoch, term_threshold,
                                       use_cuda, len(l_shared_fp))
            np_adj_embed = t_adj_embed.cpu().detach().numpy()
            # np_adj_embed = swap_matrix_back_to_orig(np_adj_embed, d_cur_idx_to_orig_idx)
            np_adj_embed = np_adj_embed[l_orig_tkg_node_idx]
            l_ready = []
            for row_idx, adj_embed in enumerate(np_adj_embed):
                token = l_sub_tkg_node[l_orig_tkg_node_idx[row_idx]]
                l_ready.append((token, adj_embed))
            df_ready = pd.DataFrame(l_ready, columns=['token', 'adj_token_embed'])
            save_name = 'test#fp#' + ds_name + '#' + str(np.round(tau, decimals=1))
            # np.save(global_settings.g_adj_token_embed_file_fmt.format(save_name), np_adj_embed)
            df_ready = df_ready.set_index('token')
            pd.to_pickle(df_ready, global_settings.g_adj_token_embed_file_fmt.format(save_name))
            logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] done with tau = %s for sub_tkg %s in %s secs.'
                          % (tau, nx.info(sub_tkg), time.time() - timer_start))

    logging.debug('[test_adjust_embed_with_fp_wrapper_for_tkg] all done in %s secs.' % str(time.time() - timer_start))


####################################################################################################
#   Test Only End
####################################################################################################



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'adj_embed':
        # this procedure is memory consuming. if GPU is in use, it may ran out of GPU memory.
        # running this with pure CPU is somewhat slower than GPU, but it can run through.
        ds_name = sys.argv[2]
        tau_stride = float(sys.argv[3])
        max_epoch = int(sys.argv[4])
        term_threshold = float(sys.argv[5])
        use_cuda = bool(sys.argv[6])
        tau = sys.argv[7]
        l_tau = [np.round(float(tau), 2)]
        # l_tau = np.round(np.arange(0.0, 1.0, tau_stride), 2)
        adjust_embed_wrapper_for_pksg(ds_name, l_tau, max_epoch, term_threshold, use_cuda)
    elif cmd == 'adj_embed_dist':
        ds_name = sys.argv[2]
        compute_adjusted_embedding_distributions(ds_name)
    elif cmd == 'adj_embed_samples':
        ds_name = sys.argv[2]
        sample_size = int(sys.argv[3])
        num_batch = int(sys.argv[4])
        extend_adj_embed_dist_to_samples(ds_name, sample_size, num_batch)
    elif cmd == 'adj_token_embed':
        ds_name = sys.argv[2].strip()
        # tau_stride = float(sys.argv[3])
        max_epoch = int(sys.argv[3])
        term_threshold = float(sys.argv[4])
        build_cckt = bool(sys.argv[5])
        use_cuda = bool(sys.argv[6])
        # tau = sys.argv[7]
        # l_tau = [np.round(float(tau), 2)]
        l_tau = np.arange(0.0, 1.0, 0.1)
        tkg_min_size = 1000
        adjust_embed_wrapper_for_tkg(ds_name, tkg_min_size, l_tau, max_epoch, term_threshold, build_cckt, use_cuda)
    elif cmd == 'adj_token_embed_with_fp':
        ds_name = sys.argv[2].strip()
        max_epoch = int(sys.argv[3])
        term_threshold = float(sys.argv[4])
        build_cckt = bool(sys.argv[5])
        use_cuda = bool(sys.argv[6])
        l_tau = np.arange(0.0, 1.0, 0.1)
        tkg_min_size = 1000
        adjust_embed_with_fp_wrapper_for_tkg(ds_name, tkg_min_size, l_tau, max_epoch, term_threshold, build_cckt,
                                             use_cuda)
    elif cmd == 'adj_token_embed_with_merged_corpus':
        l_ds_name = ['ccoha1', 'ccoha2']
        max_epoch = int(sys.argv[2])
        term_threshold = float(sys.argv[3])
        build_cckt = bool(sys.argv[4])
        use_cuda = bool(sys.argv[5])
        l_tau = np.arange(0.0, 1.0, 0.1)
        adj_embed_by_merged_corpus(l_ds_name, l_tau, max_epoch, term_threshold, build_cckt, use_cuda)
    elif cmd == 'update_adj_token_embed':
        ds_name = sys.argv[2].strip()
        rel_ds_name = sys.argv[3].strip()
        max_epoch = int(sys.argv[4])
        term_threshold = float(sys.argv[5])
        use_cuda = bool(sys.argv[6])
        l_tau = np.arange(0.0, 1.0, 0.1)
        tkg_min_size = 1000
        update_adjust_embed_wrapper_for_tkg(ds_name, rel_ds_name, tkg_min_size, l_tau, max_epoch, term_threshold,
                                            use_cuda)
    elif cmd == 'adj_token_embed_with_reg':
        ds_name = sys.argv[2].strip()
        max_epoch = int(sys.argv[3])
        term_threshold = float(sys.argv[4])

        if len(sys.argv) >= 7:
            tune_ds_name = bool(sys.argv[5])
            tune_tau_str = sys.argv[6].strip()
        else:
            tune_ds_name = None
            tune_tau_str = None

        l_tau = np.arange(0.0, 1.0, 0.1)
        tkg_min_size = 1000
        adjust_embed_wrapper_for_tkg_with_regularization(ds_name, tkg_min_size, l_tau, max_epoch, term_threshold,
                                                         build_cckt=False, use_cuda=False, trans_measure_mat=False,
                                                         use_rec_cos=True, use_graph_dev_reg=True,
                                                         cos_dev_bynd_neig_weight=2.0, use_embed_entropy=False,
                                                         tune_ds_name=tune_ds_name, tune_tau_str=tune_tau_str)
    elif cmd == 'intersect_tkg':
        ds_name_1 = 'cckg#ccoha1'
        ds_name_2 = 'cckg#ccoha2'
        get_tkg_intersection(ds_name_1, ds_name_2)

    elif cmd == 'diff_tkg':
        ds_name_1 = 'cckg#ccoha1'
        ds_name_2 = 'cckg#ccoha2'
        ds_name_int = 'cckg#ccoha1#INT#cckg#ccoha2'
        get_tkg_difference(ds_name_1, ds_name_2, ds_name_int)

    elif cmd == 'adj_token_embed_with_squeeze':
        ds_name = sys.argv[2].strip()
        squeeze_ds_name = sys.argv[3].strip()
        orig_embed_ds_name = sys.argv[4].strip()
        max_epoch = int(sys.argv[5])
        term_threshold = float(sys.argv[6])
        l_tau = np.arange(0.1, 1.0, 0.1)
        tkg_min_size = 1000
        adjust_embed_wrapper_for_tkg_with_squeeze(ds_name, squeeze_ds_name, orig_embed_ds_name, tkg_min_size, l_tau,
                                                  max_epoch, term_threshold, build_cckt=True, use_cuda=False,
                                                  trans_measure_mat=False,
                                                  use_rec_cos=True, use_graph_dev_reg=True,
                                                  cos_dev_bynd_neig_weight=2.0, use_embed_entropy=False)

    elif cmd == 'test':
        # gen_tkg_subset()
        tkg_min_size = 10
        l_tau = np.arange(0.0, 1.0, 0.1)
        max_epoch = 200
        term_threshold = 0.001
        use_cuda = True
        test_adjust_embed_with_fp_wrapper_for_tkg(tkg_min_size, l_tau, max_epoch, term_threshold, use_cuda)
