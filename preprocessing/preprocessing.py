import cupy
import pydicom
import json
import parmap
import tqdm
import pathlib
import argparse
from setproctitle import setproctitle
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils import interpolation_cupy, fig_3views

parser = argparse.ArgumentParser()
parser.add_argument('--num_process', default=1, type=int, help="Number of processes for multiprocess")
parser.add_argument('--gpu', default='0', type=str, help="numbers of gpu e.g. 0,1,2")
args = parser.parse_args()

def _normalize(nparray, max_hu=400., min_hu=-1000.): 
    nparray = (nparray - min_hu) / (max_hu - min_hu)
    nparray[nparray>1] = 1.
    nparray[nparray<0] = 0.
    return (nparray*255).astype(np.uint8)

def preprocessing(series_list, gpu, p_num, spacings=[0.8,0.9,1.0,1.1,1.2], cs=72,
                  img_save_dir = '/data/dk/luna16_crops',
                  fig_save_dir = '/data/dk/images/luna16/crops',
                  csv_path = '/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates_merged_split.csv',
                  data_path = '/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/all_series'):
    setproctitle(f"preprocessing_{p_num}")
    t_bar = tqdm.tqdm(series_list, position=p_num+1, leave=False)
    cs_half = int(cs/2)
    df_candidates = pd.read_csv(csv_path)
    for series_uid in t_bar:
        cur_df = df_candidates[df_candidates['seriesuid']==series_uid]
        dcm_dir = pathlib.Path(data_path)
        mhd_path = dcm_dir / f"{series_uid}.mhd"
        ct_image = sitk.ReadImage(mhd_path)
        img_arr = sitk.GetArrayViewFromImage(ct_image)
        origin = list(reversed(ct_image.GetOrigin()))
        spacing = list(reversed(ct_image.GetSpacing()))
        img_arr = _normalize(img_arr)
        for s in spacings:
            new_spacing = [s]*3
            img_rescaled = interpolation_cupy(img_arr, spacing, new_spacing, gpu)
            pad_arr = np.pad(img_rescaled, ((cs_half,cs_half),(cs_half,cs_half),(cs_half,cs_half)), mode='constant', constant_values=0)
            for i in range(len(cur_df)):
                # t_bar.set_description(f"P_IDX: {p_num}, {series_uid}, {s}, {i}/{len(cur_df)}")
                t_bar.set_description(f"P_IDX: {p_num}")
                row = cur_df.iloc[i]
                coords = [int(row['coordZ']),int(row['coordY']),int(row['coordX'])]
                cls = str(row['class'])
                img_save_path = pathlib.Path(img_save_dir) / cls / series_uid
                img_save_path.mkdir(parents=True, exist_ok=True)
                fig_save_path = pathlib.Path(fig_save_dir) / cls 
                fig_save_path.mkdir(parents=True, exist_ok=True)
                
                # if (fig_save_path / f"{series_uid}_{'_'.join([str(x) for x in coords])}_{s}.png").exists():
                #     continue
                
                voxel_coord = np.absolute(np.array(coords)-origin)
                voxel_coord /= np.array(new_spacing)
                voxel_coord = voxel_coord.astype(int)
                patch = pad_arr[voxel_coord[0]:voxel_coord[0]+cs,
                        voxel_coord[1]:voxel_coord[1]+cs,
                        voxel_coord[2]:voxel_coord[2]+cs]
                np.save(img_save_path / f"{'_'.join([str(x) for x in coords])}_{s}.npy", patch)
                if cls == '1':
                    fig_3views(patch)
                    plt.savefig(fig_save_path / f"{series_uid}_{'_'.join([str(x) for x in coords])}_{s}.png")
                    plt.close()

def preprocessing_index(series_list, gpu, p_num, spacings=[0.8,0.9,1.0,1.1,1.2], cs=72):
    setproctitle(f"preprocessing_{p_num}")
    t_bar = tqdm.tqdm(series_list, position=p_num+1, leave=False, desc=f"PID: {p_num}")
    cs_half = int(cs/2)
    df_candidates = pd.read_csv('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates_merged_split.csv')
    missed_dict = json.load(open('missed_indexes.json', 'r'))
    for series_uid in t_bar:
        indexes = missed_dict[series_uid]
        dcm_dir = pathlib.Path('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/all_series')
        mhd_path = dcm_dir / f"{series_uid}.mhd"
        ct_image = sitk.ReadImage(mhd_path)
        img_arr = sitk.GetArrayViewFromImage(ct_image)
        origin = list(reversed(ct_image.GetOrigin()))
        spacing = list(reversed(ct_image.GetSpacing()))
        img_arr = _normalize(img_arr)
        for s in spacings:
            new_spacing = [s]*3
            img_rescaled = interpolation_cupy(img_arr, spacing, new_spacing, gpu)
            pad_arr = np.pad(img_rescaled, ((cs_half,cs_half),(cs_half,cs_half),(cs_half,cs_half)), mode='constant', constant_values=0)
            for i in indexes:
                # t_bar.set_description(f"P_IDX: {p_num}, {series_uid}, {s}, {i}/{len(cur_df)}")
                
                row = df_candidates.iloc[i]
                coords = [int(row['coordZ']),int(row['coordY']),int(row['coordX'])]
                cls = str(row['class'])
                img_save_path = pathlib.Path('/data/dk/LUNA16_crops') / cls / series_uid
                img_save_path.mkdir(parents=True, exist_ok=True)
                fig_save_path = pathlib.Path('/data/dk/images/LUNA16/crops') / cls 
                fig_save_path.mkdir(parents=True, exist_ok=True)
                
                # if (fig_save_path / f"{series_uid}_{'_'.join([str(x) for x in coords])}_{s}.png").exists():
                #     continue
                
                voxel_coord = np.absolute(np.array(coords)-origin)
                voxel_coord /= np.array(new_spacing)
                voxel_coord = voxel_coord.astype(int)
                patch = pad_arr[voxel_coord[0]:voxel_coord[0]+cs,
                        voxel_coord[1]:voxel_coord[1]+cs,
                        voxel_coord[2]:voxel_coord[2]+cs]
                np.save(img_save_path / f"{'_'.join([str(x) for x in coords])}_{s}.npy", patch)
                if cls == '1':
                    fig_3views(patch)
                    plt.savefig(fig_save_path / f"{series_uid}_{'_'.join([str(x) for x in coords])}_{s}.png")
                    plt.close()
    
def main():
    core_count = args.num_process # Should be multiple of gpu_n_list's length
    gpu_n_list = (args.gpu).replace(' ','').split(',')
    pids = list(range(core_count))
    # df_candidates = pd.read_csv('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates.csv')
    df_candidates = pd.read_csv('/mnt/NAS/datasets/LuCAS-Plus/LungSegmentation/LUNA16/candidates_merged_split.csv')
    series_uids = list(set(df_candidates['seriesuid']))
    # with open('/data/dk/LUNA16_crops/remain_series_uids_2.json', 'r') as file:
    #     series_uids = json.load(file)
    print(f"Total series number: {len(series_uids)}")
    split_series_uids = [x.tolist() for x in np.array_split(np.array(series_uids), core_count)]
    gpus = []
    for i, g in enumerate(gpu_n_list,1):
        if i == len(gpu_n_list):
            gpus += [g]*(core_count-len(gpus))
        else:
            gpus += [g]*int((core_count/len(gpu_n_list)))
    parmap.starmap(preprocessing, list(zip(split_series_uids, gpus, pids)), pm_pbar=True, pm_processes=core_count)

def main_missed():
    core_count = args.num_process # Should be multiple of gpu_n_list's length
    gpu_n_list = (args.gpu).replace(' ','').split(',')
    pids = list(range(core_count))
    missed_dict = json.load(open('missed_indexes.json', 'r'))
    series_uids = list(missed_dict.keys())
    print(f"Total series number: {len(series_uids)}")
    split_series_uids = [x.tolist() for x in np.array_split(np.array(series_uids), core_count)]
    gpus = []
    for i, g in enumerate(gpu_n_list,1):
        if i == len(gpu_n_list):
            gpus += [g]*(core_count-len(gpus))
        else:
            gpus += [g]*int((core_count/len(gpu_n_list)))
    parmap.starmap(preprocessing_index, list(zip(split_series_uids, gpus, pids)), pm_pbar=True, pm_processes=core_count)

if __name__ == '__main__':
    main()
    # main_missed()