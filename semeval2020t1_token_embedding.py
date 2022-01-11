import json
import logging
import math
import multiprocessing
import sys
import time
from os import walk

import networkx as nx
import pandas as pd
import numpy as np
from sklearn import preprocessing

import semeval2020t1_global_settings as global_settings
sys.path.insert(1, global_settings.g_lexvec_model_folder)
import model as lexvec


def load_lexvec_model():
    lexvec_model = lexvec.Model(global_settings.g_lexvec_vect_file_path)
    embedding_len = len(lexvec_model.word_rep('the'))
    logging.debug('[load_lexvec_model] The length of embeddings is %s' % embedding_len)
    return lexvec_model, embedding_len


def token_embedding(lexvec_model, embedding_len, token):
    if lexvec_model is None:
        raise Exception('lexvec_model is not loaded!')
    word_vec = lexvec_model.word_rep(token)
    if not np.isfinite(word_vec).all():
        logging.error('Invalid embedding for %s!' % str(token))
        word_vec = np.zeros(embedding_len)
    word_vec = np.asarray(word_vec, dtype=np.float32)
    word_vec = preprocessing.normalize(word_vec.reshape(1, -1))
    word_vec = word_vec[0]
    return word_vec


def token_embedding_single_proc(task_id):
    logging.debug('[token_embedding_single_proc] Proc %s: Starts.' % str(task_id))
    timer_start = time.time()

    df_task = pd.read_pickle(global_settings.g_token_embed_task_file_fmt.format(task_id))
    logging.debug('[token_embedding_single_proc] Proc %s: Load in %s phrase recs.' % (task_id, len(df_task)))

    lexvec_model, embedding_len = load_lexvec_model()

    l_token_embed_rec = []
    token_cnt = 0
    for _, token_rec in df_task.iterrows():
        token = token_rec['token'].strip()
        phrase_embed = token_embedding(lexvec_model, embedding_len, token)
        l_token_embed_rec.append((token, phrase_embed))
        token_cnt += 1
        if token_cnt % 5000 == 0 and token_cnt >= 5000:
            logging.debug('[token_embedding_single_proc] Proc %s: %s token embeds done in %s secs.'
                          % (task_id, token_cnt, time.time() - timer_start))
    logging.debug('[token_embedding_single_proc] Proc %s: %s token embeds done in %s secs.'
                  % (task_id, token_cnt, time.time() - timer_start))

    df_token_embed = pd.DataFrame(l_token_embed_rec, columns=['token', 'token_embed'])
    pd.to_pickle(df_token_embed, global_settings.g_token_embed_int_file_fmt.format(task_id))
    df_token_embed = df_token_embed.set_index('token')
    logging.debug('[token_embedding_single_proc] Proc %s: All done with %s token embeds in %s secs.'
                  % (task_id, len(df_token_embed), time.time() - timer_start))


def token_embedding_multiproc(num_proc, job_id, build_cckt=False):
    logging.debug('[token_embedding_multiproc] Starts.')
    timer_start = time.time()

    l_task_ids = [str(job_id) + '#' + str(idx) for idx in range(int(num_proc))]
    l_proc = []
    for task_id in l_task_ids:
        p = multiprocessing.Process(target=token_embedding_single_proc,
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
                logging.debug('[token_embedding_multiproc] %s is finished.' % p.name)
    logging.debug('[token_embedding_multiproc] All done in %s secs.' % str(time.time() - timer_start))


def gen_token_embedding_tasks(ds_name, num_tasks, job_id, build_cckt=False):
    logging.debug('[gen_token_embedding_tasks] Starts.')
    timer_start = time.time()

    if build_cckt:
        ds_name = 'cckg#' + ds_name
    tkg = nx.read_gpickle(global_settings.g_merged_tw_tkg_file_fmt.format(ds_name))
    l_tokens = list(tkg.nodes())
    num_tokens = len(l_tokens)
    logging.debug('[gen_token_embedding_tasks] Load in %s tokens.' % str(num_tokens))

    num_tasks = int(num_tasks)
    batch_size = math.ceil(num_tokens / num_tasks)
    l_tasks = []
    for i in range(0, num_tokens, batch_size):
        if i + batch_size < num_tokens:
            l_tasks.append(l_tokens[i:i + batch_size])
        else:
            l_tasks.append(l_tokens[i:])
    logging.debug('[gen_token_embedding_tasks] Need to generate %s tasks.' % str(len(l_tasks)))

    for idx, tokens in enumerate(l_tasks):
        l_ready = []
        for token in tokens:
            l_ready.append((token, ))
            df_ready = pd.DataFrame(l_ready, columns=['token'])
            pd.to_pickle(df_ready, global_settings.g_token_embed_task_file_fmt.format(str(job_id) + '#' + str(idx)))
    logging.debug('[gen_token_embedding_tasks] All done with %s tasks in %s secs.'
                  % (len(l_tasks), time.time() - timer_start))


def merge_token_embedding_int(ds_name, build_cckg=False):
    logging.debug('[merge_token_embedding_int]')
    timer_start = time.time()

    l_int = []
    for (dirpath, dirname, filenames) in walk(global_settings.g_phrase_embed_int_folder):
        for filename in filenames:
            if filename[-7:] != '.pickle' or filename[:16] != 'token_embed_int_':
                continue
            df_int = pd.read_pickle(dirpath + filename)
            l_int.append(df_int)
    logging.debug('[merge_token_embedding_int] Read in %s int dfs.' % str(len(l_int)))
    df_merge = pd.concat(l_int)
    df_merge = df_merge.set_index('token')
    if build_cckg:
        ds_name = 'cckg#' + ds_name
    pd.to_pickle(df_merge, global_settings.g_token_embed_file_fmt.format(ds_name))
    logging.debug('[merge_token_embedding_int] All done with %s recs in %s secs.'
                  % (len(df_merge), time.time() - timer_start))


def combine_embeddings(l_ds_name):
    logging.debug('[combine_embeddings] starts.')

    l_df_embed = []
    for ds_name in l_ds_name:
        df_embed = pd.read_pickle(global_settings.g_token_embed_file_fmt.format(ds_name))
        logging.debug('[combine_embeddings] load in df_embed for %s with %s recs.' % (ds_name, len(df_embed)))
        l_df_embed.append(df_embed)

    df_embed_combined = pd.concat(l_df_embed).reset_index().drop_duplicates(['token']).set_index('token')
    pd.to_pickle(df_embed_combined, global_settings.g_token_embed_combined_file_fmt.format('#'.join(l_ds_name)))
    logging.debug('[combine_embeddings] all done with %s recs.' % str(len(df_embed_combined)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'gen_token_embed_tasks':
        ds_name = sys.argv[2]
        num_tasks = sys.argv[3]
        job_id = sys.argv[4]
        build_cckg = bool(sys.argv[5])
        gen_token_embedding_tasks(ds_name, num_tasks, job_id, build_cckg)
    elif cmd == 'token_embed':
        num_proc = sys.argv[2]
        job_id = sys.argv[3]
        token_embedding_multiproc(num_proc, job_id)
    elif cmd == 'merge_token_embed_int':
        ds_name = sys.argv[2]
        build_cckg = bool(sys.argv[3])
        merge_token_embedding_int(ds_name, build_cckg)
    elif cmd == 'combine_token_embed':
        l_ds_name = ['cckg#ccoha1', 'cckg#ccoha2']
        combine_embeddings(l_ds_name)