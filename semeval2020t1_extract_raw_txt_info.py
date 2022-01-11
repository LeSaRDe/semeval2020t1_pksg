import logging
import time
import sys

import pandas as pd

import semeval2020t1_global_settings


def extract_txt_info_for_one_ds(ds_name):
    logging.debug('[extract_txt_info_single_proc] DS %s: Starts.' % str(ds_name))
    timer_start = time.time()

    l_tw_json = []
    with open(semeval2020t1_global_settings.g_tw_raw_data_input_folder + ds_name, 'r') as in_fd:
        for ln in in_fd:
            tw_str = ln.strip()
            l_tw_json.append(tw_str)
        in_fd.close()
    logging.debug('[extract_txt_info_single_proc] DS %s: %s tw objs in total.' % (ds_name, len(l_tw_json)))

    total_cnt = 0
    ready_cnt = 0
    l_ready = []
    for tw_id, tw_json in enumerate(l_tw_json):
        tw_id = ds_name + '#' + str(tw_id)
        tw_type = 'n'
        tw_datetime = None
        tw_lang = 'en'
        tw_usr_id = None
        tw_src_id = None
        tw_src_usr_id = None
        tw_raw_txt = tw_json
        full_txt_flag = False

        l_ready.append((tw_id, tw_type, tw_datetime, tw_lang, tw_usr_id, tw_src_id, tw_src_usr_id,
                        tw_raw_txt, full_txt_flag))
        ready_cnt += 1
        total_cnt += 1
        if total_cnt % 5000 == 0 and total_cnt >= 5000:
            logging.debug('[extract_txt_info_single_proc] DS %s: total_cnt = %s ready_cnt = %s done in %s secs.'
                          % (ds_name, total_cnt, ready_cnt, time.time() - timer_start))
    df_ready = pd.DataFrame(l_ready, columns=['tw_id', 'tw_type', 'tw_datetime', 'tw_lang', 'tw_usr_id', 'tw_src_id',
                                              'tw_src_usr_id', 'tw_raw_txt', 'full_txt_flag'])
    df_ready = df_ready.set_index('tw_id')
    pd.to_pickle(df_ready, semeval2020t1_global_settings.g_raw_tw_info_file_fmt.format(ds_name))
    logging.debug('[extract_txt_info_single_proc] Proc %s: All done with %s recs in %s secs.'
                  % (ds_name, len(df_ready), time.time() - timer_start))


def extract_txt_info(l_ds_name):
    logging.debug('[extract_txt_info] Starts.')
    timer_start = time.time()

    for ds_name in l_ds_name:
        extract_txt_info_for_one_ds(ds_name)

    logging.debug('[extract_txt_info] All done in %s secs.' % str(time.time() - timer_start))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cmd = sys.argv[1]

    if cmd == 'extract_txt_info':
        l_ds_name = ['ccoha1.txt', 'ccoha2.txt']
        extract_txt_info(l_ds_name)
