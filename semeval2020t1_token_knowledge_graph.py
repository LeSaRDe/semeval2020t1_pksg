import json
import logging
import math
import multiprocessing
# import threading
import sys
import time
# import traceback
from os import walk
from itertools import combinations, product

import networkx as nx
# from networkx.algorithms.approximation.clique import clique_removal
# from networkx.drawing.nx_agraph import graphviz_layout
import pandas as pd

import semeval2020t1_global_settings as global_settings


def build_cckg_for_one_tw(tw_id, df_tw_to_phid, df_phid_to_phstr):
    '''
    build pure co-occurrence graph on tokens
    '''
    l_phid_phpos_in_tw = [item for item in df_tw_to_phid.loc[tw_id]['l_phrase_id']]

    l_ph_to_tokens = []
    for phid, _ in l_phid_phpos_in_tw:
        phstr_i = df_phid_to_phstr.loc[phid]['phrase_str']
        ph_tokens_i = [token.strip() for token in phstr_i.split(' ')]
        l_ph_to_tokens.append(ph_tokens_i)

    tw_cckg_graph = nx.Graph()
    l_all_tokens = set([item for sublist in l_ph_to_tokens for item in sublist])
    tw_cckg_graph.add_nodes_from(l_all_tokens)
    l_all_edges = combinations(l_all_tokens, 2)
    tw_cckg_graph.add_edges_from(l_all_edges, weight=1)

    return tw_cckg_graph


def build_tkg_for_one_tw(tw_id, df_tw_to_phid, df_phid_to_phstr, sim_phid=None):
    '''
    if 'sim_phid' is not specified, then construct a clique upon all phrases of the tweet given by 'tw_id'.
    Otherwise, construct a star graph centered at 'sim_phrase_id' which links to every other phrase of the tweet.
    '''
    l_phid_phpos_in_tw = [item for item in df_tw_to_phid.loc[tw_id]['l_phrase_id']]

    l_ph_to_tokens = []
    for phid, _ in l_phid_phpos_in_tw:
        phstr_i = df_phid_to_phstr.loc[phid]['phrase_str']
        ph_tokens_i = [token.strip() for token in phstr_i.split(' ')]
        l_ph_to_tokens.append(ph_tokens_i)

    tw_pksg_graph = nx.Graph()
    l_all_tokens = set([item for sublist in l_ph_to_tokens for item in sublist])
    tw_pksg_graph.add_nodes_from(l_all_tokens)

    # add edges from phrases:
    for ph_tokens in l_ph_to_tokens:
        l_token_edge = [item for item in list(combinations(ph_tokens, 2)) if item[0] != item[1]]
        for token_edge in l_token_edge:
            if not tw_pksg_graph.has_edge(token_edge[0], token_edge[1]):
                tw_pksg_graph.add_edge(token_edge[0], token_edge[1], weight=1)
            else:
                tw_pksg_graph.edges[token_edge[0], token_edge[1]]['weight'] += 1

    # add edges between phrases
    if sim_phid is None:
        l_ph_combs = list(combinations(l_ph_to_tokens, 2))
        for ph_pair in l_ph_combs:
            l_token_edge = [item for item in product(ph_pair[0], ph_pair[1]) if item[0] != item[1]]
            for token_edge in l_token_edge:
                if not tw_pksg_graph.has_edge(token_edge[0], token_edge[1]):
                    tw_pksg_graph.add_edge(token_edge[0], token_edge[1], weight=1)
                else:
                    tw_pksg_graph.edges[token_edge[0], token_edge[1]]['weight'] += 1
    else:
        pass

    return tw_pksg_graph


def build_tw_tkg_single_proc(task_id, ds_name, build_cckg=False, sim_phid=None):
    logging.debug('[build_tw_tkg_single_proc] Proc %s: starts.' % str(task_id))
    timer_start = time.time()

    l_tw_id = []
    with open(global_settings.g_tw_tkg_task_file_fmt.format(str(task_id)), 'r') as in_fd:
        for ln in in_fd:
            l_tw_id.append(ln.strip())
        in_fd.close()
    logging.debug('[build_tw_tkg_single_proc] Proc %s: load in %s tw ids for tasks.' % (task_id, len(l_tw_id)))

    df_tw_to_phid = pd.read_pickle(global_settings.g_tw_to_phrase_id_file_fmt.format(ds_name))
    logging.debug('[build_tw_tkg_single_proc] Proc %s: load in df_tw_to_phid with %s recs in %s secs.'
                  % (task_id, len(df_tw_to_phid), time.time() - timer_start))
    df_phid_to_phstr = pd.read_pickle(global_settings.g_phrase_id_to_phrase_str_file_fmt.format(ds_name))
    logging.debug('[build_tw_tkg_single_proc] Proc %s: load in df_phid_to_phstr with %s recs in %s secs.'
                  % (task_id, len(df_phid_to_phstr), time.time() - timer_start))

    l_tw_pksg_rec = []

    if build_cckg:
        logging.debug('[build_tw_tkg_single_proc] Proc %s: build cckg.')

    for tw_id in l_tw_id:
        if not build_cckg:
            tw_pksg = build_tkg_for_one_tw(tw_id, df_tw_to_phid, df_phid_to_phstr, sim_phid)
        else:
            tw_pksg = build_cckg_for_one_tw(tw_id, df_tw_to_phid, df_phid_to_phstr)
        l_tw_pksg_rec.append((tw_id, tw_pksg))
        if len(l_tw_pksg_rec) % 500 == 0 and len(l_tw_pksg_rec) >= 500:
            logging.debug('[build_tw_tkg_single_proc] Proc %s: %s tw pksgs done in %s secs.'
                          % (task_id, len(l_tw_pksg_rec), time.time() - timer_start))
    logging.debug('[build_tw_tkg_single_proc] Proc %s: %s tw pksgs done in %s secs.'
                  % (task_id, len(l_tw_pksg_rec), time.time() - timer_start))

    df_tw_pksg = pd.DataFrame(l_tw_pksg_rec, columns=['tw_id', 'tw_tkg'])
    df_tw_pksg = df_tw_pksg.set_index('tw_id')
    df_tw_pksg.to_pickle(global_settings.g_tw_tkg_int_file_fmt.format(task_id))
    logging.debug('[build_tw_tkg_single_proc] Proc %s: all done with %s tw pksgs in %s secs.'
                  % (task_id, len(df_tw_pksg), time.time() - timer_start))


def build_tw_tkg_multiproc(num_proc, ds_name, job_id, build_cckg=False):
    logging.debug('[build_tw_tkg_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=build_tw_tkg_single_proc,
                                    args=(task_id, ds_name, build_cckg),
                                    name='Proc ' + str(task_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[build_tw_tkg_multiproc] %s is finished.' % p.name)
    logging.debug('[build_tw_tkg_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_tw_tkg_tasks(num_proc, ds_name, job_id):
    logging.debug('[gen_tw_tkg_tasks] Starts.')
    timer_start = time.time()

    df_tw_phrase = pd.read_pickle(global_settings.g_tw_phrase_file_fmt.format(ds_name))
    logging.debug('[gen_tw_tkg_tasks] load in df_tw_phrase with %s recs in %s secs.' %
                  (len(df_tw_phrase), time.time() - timer_start))

    l_tw_id = df_tw_phrase.index.to_list()
    num_tw_id = len(l_tw_id)
    num_tasks = int(num_proc)
    batch_size = math.ceil(num_tw_id / num_tasks)
    l_tasks = []
    for i in range(0, num_tw_id, batch_size):
        if i + batch_size < num_tw_id:
            l_tasks.append(l_tw_id[i:i + batch_size])
        else:
            l_tasks.append(l_tw_id[i:])
    logging.debug('[gen_tw_tkg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        with open(global_settings.g_tw_tkg_task_file_fmt.format(task_name), 'w+') as out_fd:
            out_str = '\n'.join([item.strip() for item in task])
            out_fd.write(out_str)
            out_fd.close()
    logging.debug('[gen_tw_tkg_tasks] All done with %s tw pksg tasks generated.' % str(len(l_tasks)))


def merge_two_tw_tkg(tw_pksg_1, tw_pksg_2):
    if tw_pksg_1.number_of_nodes() == 0 or tw_pksg_2.number_of_nodes() == 0:
        raise Exception('[merge_two_tw_tkg] empty pksg occurs.')

    tw_pksg_2.add_nodes_from(tw_pksg_1.nodes())

    for edge_1 in tw_pksg_1.edges(data=True):
        if not tw_pksg_2.has_edge(edge_1[0], edge_1[1]):
            tw_pksg_2.add_edge(edge_1[0], edge_1[1], weight=edge_1[2]['weight'])
        else:
            tw_pksg_2.edges()[edge_1[0], edge_1[1]]['weight'] += edge_1[2]['weight']

    return tw_pksg_2


def divide_and_conquer_merge_tw_tkg(l_tw_pksg):
    if len(l_tw_pksg) < 1:
        raise Exception('[divide_and_conquer_merge_tw_tkg] invalid l_tw_pksg')
    if len(l_tw_pksg) == 1:
        return l_tw_pksg[0]
    if len(l_tw_pksg) == 2:
        return merge_two_tw_tkg(l_tw_pksg[0], l_tw_pksg[1])

    batch_size = math.ceil(len(l_tw_pksg) / 2)
    ret_graph = merge_two_tw_tkg(divide_and_conquer_merge_tw_tkg(l_tw_pksg[:batch_size]),
                                 divide_and_conquer_merge_tw_tkg(l_tw_pksg[batch_size:]))
    return ret_graph


def merge_tw_tkg_single_proc(task_id):
    logging.debug('[merge_tw_tkg_single_proc] Proc %s: starts.' % str(task_id))
    timer_start = time.time()

    df_tw_pksg_task = pd.read_pickle(global_settings.g_merged_tw_tkg_task_file_fmt.format(task_id))
    logging.debug('[merge_tw_tkg_single_proc] Proc %s: load in %s tw pksg in %s secs.'
                  % (task_id, len(df_tw_pksg_task), time.time() - timer_start))

    l_tw_pksg = df_tw_pksg_task['tw_tkg'].to_list()
    batch_size = 10
    cnt = 0
    while True:
        l_tasks = []
        num_tasks = len(l_tw_pksg)
        for i in range(0, num_tasks, batch_size):
            if i + batch_size < num_tasks:
                l_tasks.append(l_tw_pksg[i:i + batch_size])
            else:
                l_tasks.append(l_tw_pksg[i:])

        l_rets = []
        for task in l_tasks:
            merged_tw_pksg = divide_and_conquer_merge_tw_tkg(task)
            l_rets.append(merged_tw_pksg)

        if len(l_rets) <= 0:
            raise Exception('[merge_tw_tkg_single_proc] Proc %s: invalid l_rets')
        elif len(l_rets) == 1:
            cnt += 1
            nx.write_gpickle(l_rets[0], global_settings.g_merged_tw_tkg_int_file_fmt.format(task_id))
            logging.debug('[merge_tw_tkg_single_proc] Proc %s: All done in %s secs: %s'
                          % (task_id, time.time() - timer_start, nx.info(l_rets[0])))
            return
        else:
            cnt += 1
            logging.debug('[merge_tw_tkg_single_proc] Proc %s: %s interation done in %s secs. %s tw pksg left to merge.'
                          % (task_id, cnt, time.time() - timer_start, len(l_rets)))
            l_tw_pksg = l_rets


def merge_tw_tkg_multiproc(num_proc, ds_name, job_id, build_cckg=False):
    logging.debug('[merge_tw_tkg_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=merge_tw_tkg_single_proc,
                                    args=(task_id,),
                                    name='Proc ' + str(task_id))
        p.start()
        l_proc.append(p)

    while len(l_proc) > 0:
        for p in l_proc:
            if p.is_alive():
                p.join(1)
            else:
                l_proc.remove(p)
                logging.debug('[merge_tw_tkg_multiproc] %s is finished.' % p.name)

    l_merged_tw_pksg_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:18] != 'merged_tw_tkg_int_':
                continue
            merge_tw_pksg_int = nx.read_gpickle(dirpath + filename)
            l_merged_tw_pksg_int.append(merge_tw_pksg_int)
    merged_tw_pksg = divide_and_conquer_merge_tw_tkg(l_merged_tw_pksg_int)
    if build_cckg:
        save_name = 'cckg#' + ds_name
    else:
        save_name = ds_name
    nx.write_gpickle(merged_tw_pksg, global_settings.g_merged_tw_tkg_file_fmt.format(save_name))
    logging.debug('[merge_tw_tkg_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_merged_tw_tkg_tasks(num_proc, job_id):
    logging.debug('[gen_merged_tw_tkg_tasks] starts.')
    timer_start = time.time()

    l_tw_pksg = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_ks_graph_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:11] != 'tw_tkg_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_tw_pksg += df_int['tw_tkg'].to_list()
            logging.debug('[gen_merged_tw_tkg_tasks] read in %s tw pksg.' % str(len(df_int)))
    logging.debug('[gen_merged_tw_tkg_tasks] all %s tw pksg loaded in %s secs.'
                  % (len(l_tw_pksg), time.time() - timer_start))

    l_tw_pksg = [(item,) for item in l_tw_pksg]
    num_tw_pksg = len(l_tw_pksg)
    num_tasks = int(num_proc)
    batch_size = math.ceil(num_tw_pksg / num_tasks)
    l_tasks = []
    for i in range(0, num_tw_pksg, batch_size):
        if i + batch_size < num_tw_pksg:
            l_tasks.append(l_tw_pksg[i:i + batch_size])
        else:
            l_tasks.append(l_tw_pksg[i:])
    logging.debug('[gen_merged_tw_pksg_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for task_id, task in enumerate(l_tasks):
        task_name = str(job_id) + '#' + str(task_id)
        df_task = pd.DataFrame(task, columns=['tw_tkg'])
        pd.to_pickle(df_task, global_settings.g_merged_tw_tkg_task_file_fmt.format(task_name))
    logging.debug('[gen_merged_tw_pksg_tasks] All done with %s tw pksg tasks generated.' % str(len(l_tasks)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_tw_tkg_tasks':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        gen_tw_tkg_tasks(num_proc, ds_name, job_id)
    elif cmd == 'tw_tkg':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        build_cckg = bool(sys.argv[5])
        build_tw_tkg_multiproc(num_proc, ds_name, job_id, build_cckg)
    elif cmd == 'gen_merged_tw_tkg_tasks':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        gen_merged_tw_tkg_tasks(num_proc, job_id)
    elif cmd == 'merge_tw_tkg':
        num_proc = sys.argv[2]
        ds_name = sys.argv[3]
        job_id = sys.argv[4]
        build_cckg = bool(sys.argv[5])
        merge_tw_tkg_multiproc(num_proc, ds_name, job_id, build_cckg)