import logging
from os import walk
import time
import json
import math
import multiprocessing
import sys

import pandas as pd

import semeval2020t1_global_settings
from semantic_units_extractor import SemUnitsExtractor


def tw_clean_for_one_ds(ds_name):
    logging.debug('[tw_clean_for_one_ds] DS %s: Starts...' % str(ds_name))
    timer_start = time.time()

    df_tw_clean_task = pd.read_pickle(semeval2020t1_global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    df_tw_clean_task['tw_clean_txt'] = None
    logging.debug('[tw_clean_for_one_ds] DS %s: Load in %s tasks.' % (ds_name, len(df_tw_clean_task)))
    sem_unit_ext_ins = SemUnitsExtractor(semeval2020t1_global_settings.g_sem_units_extractor_config_file)

    for tw_id, tw_clean_task in df_tw_clean_task.iterrows():
        tw_raw_txt = tw_clean_task['tw_raw_txt']
        tw_clean_txt = sem_unit_ext_ins.text_clean(tw_raw_txt)
        if not tw_clean_txt is None and tw_clean_txt != '':
            df_tw_clean_task.at[tw_id, 'tw_clean_txt'] = tw_clean_txt
    pd.to_pickle(df_tw_clean_task, semeval2020t1_global_settings.g_tw_clean_file_fmt.format(ds_name))
    logging.debug('[tw_clean_for_one_ds] Proc %s: All done with %s recs in %s secs.'
                  % (ds_name, len(df_tw_clean_task), time.time() - timer_start))


def tw_clean(l_ds_name):
    logging.debug('[tw_clean] Starts')
    timer_start = time.time()

    for ds_name in l_ds_name:
        tw_clean_for_one_ds(ds_name)

    logging.debug('[tw_clean] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'txt_clean':
        l_ds_name = ['ccoha1', 'ccoha2']
        tw_clean(l_ds_name)
