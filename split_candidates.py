import pandas as pd
import pathlib
import json
import os
import time
import pickle
from tqdm import trange, tqdm
import numpy as np
from parmap import starmap, map

df = pd.read_csv('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates_merged_split.csv')
root = pathlib.Path('/data/dk/LUNA16_crops')

def worker(indexes, pid, split, cls=None):
    samples = []
    if cls is not None:
        target_df = df[(df['Split']==split) & (df['class']==cls)]
    else:  
        target_df = df[df['Split']==split]
    tbar = tqdm(indexes, desc=f"PID {pid} / {split} / {cls}", position=pid+1, leave=False)
    for i in tbar:
        series_dir = root / str(target_df['class'][i]) / target_df['seriesuid'][i]
        for path in series_dir.glob('*'):
            samples.append((str(path), int(target_df['class'][i])))
    
    file_name = f'{split}_{cls}_{pid}.json' if cls is not None else f'{split}_{pid}.json'
    with open(file_name, 'w') as file:
        json.dump(samples, file)
    

def main():
    core_count = 20 # Should be multiple of gpu_n_list's length
    pids = list(range(core_count))
    df = pd.read_csv('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates_merged_split.csv')
    root = pathlib.Path('/data/dk/LUNA16_crops')

    ''' Training dataset '''
    # train_samples = []
    # train_df = df[df['Split']=='train']
    # split_indexes = [x.tolist() for x in np.array_split(np.array(list(train_df.index)), core_count)]
    # starmap(worker, list(zip(split_indexes, pids)), pm_pbar=True, pm_processes=core_count)
    # for pid in pids:
    #     cur_t_samples = json.load(open(f'train_{pid}.json', 'r'))
    #     os.remove(f'train_{pid}.json')
    #     train_samples += cur_t_samples
    # with open('train.json', 'w') as file:
    #     json.dump(train_samples, file)
    
    ''' Validation dataset '''
    # valid_pos_samples, valid_neg_samples = [], []
    # valid_pos_df = df[(df['Split']=='valid') & (df['class']==1)]
    # valid_pos_df.reset_index(drop=True)
    # valid_neg_df = df[(df['Split']=='valid') & (df['class']==0)]
    # valid_neg_df.reset_index(drop=True)
    # for i in tqdm(valid_pos_df.index):
    #     series_dir = root / '1' / valid_pos_df['seriesuid'][i]
    #     for path in series_dir.glob('*'):
    #         valid_pos_samples.append((str(path), 1))
    # with open('valid_pos.json', 'w') as file:
    #     json.dump(valid_pos_samples, file)
    
    # split_indexes = [x.tolist() for x in np.array_split(np.array(list(valid_neg_df.index)), core_count)]
    # starmap(worker, list(zip(split_indexes, pids)), 'valid', 0, pm_pbar=True, pm_processes=core_count)
    # for pid in tqdm(pids):
    #     cur_vn_samples = json.load(open(f'valid_0_{pid}.json', 'r'))
    #     os.remove(f'valid_0_{pid}.json')
    #     valid_neg_samples += cur_vn_samples
    # with open('valid_neg.json', 'w') as file:
    #     json.dump(valid_neg_samples, file)
    
    ''' Test dataset '''
    test_pos_samples, test_neg_samples = [], []
    test_pos_df = df[(df['Split']=='test') & (df['class']==1)]
    test_pos_df.reset_index(drop=True)
    test_neg_df = df[(df['Split']=='test') & (df['class']==0)]
    test_neg_df.reset_index(drop=True)
    for i in tqdm(test_pos_df.index):
        series_dir = root / '1' / test_pos_df['seriesuid'][i]
        for path in series_dir.glob('*'):
            test_pos_samples.append((str(path), 1))
    with open('test_pos.json', 'w') as file:
        json.dump(test_pos_samples, file)
    
    split_indexes = [x.tolist() for x in np.array_split(np.array(list(test_neg_df.index)), core_count)]
    starmap(worker, list(zip(split_indexes, pids)), 'test', 0, pm_pbar=True, pm_processes=core_count)
    for pid in tqdm(pids):
        cur_tn_samples = json.load(open(f'test_0_{pid}.json', 'r'))
        os.remove(f'test_0_{pid}.json')
        test_neg_samples += cur_tn_samples
    with open('test_neg.json', 'w') as file:
        json.dump(test_neg_samples, file)

def json_to_pickle():
    init_train_json_load = time.time()
    with open('train.json', 'r') as file:
        train_samples = json.load(file)
    print(f"train json load: {time.time()-init_train_json_load}")
    with open('train.pkl', 'wb') as file:
        pickle.dump(train_samples, file)
    del train_samples
    init_train_pickle_load = time.time()
    with open('train.json', 'rb') as file:
        train_samples = pickle.load(file)
    print(f"train pickle load: {time.time()-init_train_pickle_load}")
    
    with open('valid_pos.json', 'r') as file:
        valid_pos_samples = json.load(file)
    with open('valid_pos.pkl', 'wb') as file:
        pickle.dump(valid_pos_samples, file)
    init_valid_neg_json = time.time()
    with open('valid_neg.json', 'r') as file:
        valid_neg_samples = json.load(file)
    print(f"valid neg json load: {time.time()-init_valid_neg_json}")
    with open('valid_neg.pkl', 'wb') as file:
        pickle.dump(valid_neg_samples, file)
    del valid_neg_samples
    init_valid_neg_pickle = time.time()
    with open('valid_neg.pkl', 'rb') as file:
        valid_neg_samples = pickle.load(file)
    print(f"valid neg pickle load: {time.time()-init_valid_neg_pickle}")
    
    with open('test_pos.json', 'r') as file:
        test_pos_samples = json.load(file)
    with open('test_pos.pkl', 'wb') as file:
        pickle.dump(test_pos_samples, file)
    init_test_neg_json = time.time()
    with open('test_neg.json', 'r') as file:
        test_neg_samples = json.load(file)
    print(f"test neg json load: {time.time()-init_test_neg_json}")
    with open('test_neg.pkl', 'wb') as file:
        pickle.dump(test_neg_samples, file)
    del test_neg_samples
    init_test_neg_pickle = time.time()
    with open('test_neg.pkl', 'rb') as file:
        test_neg_samples = pickle.load(file)
    print(f"test neg pickle load: {time.time()-init_test_neg_pickle}")

if __name__ == '__main__':
    # main()
    json_to_pickle()