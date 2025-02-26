from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle, os, glob, sys
import pandas as pd, numpy as np
from utils.data_utils import *


class dcase23_t2_dataset(Dataset):
    def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.tsv"
        assert os.path.exists(metadata_path), "Error - Metadata is missing!"
        df_all = pd.read_csv(metadata_path, sep='\t')
        df_train = df_all[df_all.label == 'normal']
        if not train:
            df_train = df_all[df_all['train/test']=='train']

        classes = df_train[['mtype', 'attributes']].drop_duplicates()
        self.cls2lab = dict([["_".join([c.iloc[0], *eval(c.iloc[1])]), i] for i, (_, c) in enumerate(classes.iterrows())])
        self.lab2cls = dict([[self.cls2lab[c], c] for c in self.cls2lab])

        df_train = df_all[df_all['train/test']=='train']
        if mtype:
            df_train = df_train[df_train.mtype == mtype]
        self.mtypes = df_train.mtype.tolist()
        self.classes = ["_".join([c.iloc[0], *eval(c.iloc[1])]) for _, c in df_train[['mtype', 'attributes']].iterrows()]
        self.labels = [self.cls2lab[c] for c in self.classes]
        self.fpaths = df_train['wave_path'].tolist()
        self.domain = df_train['domain'].tolist()
        self.wave_length = wave_length
        self.mixup = mixup

    def __len__(self) -> int:
        return self.classes.__len__()
    
    def machine_types(self) -> list:
        return set(self.mtypes)

    def __getitem__(self, index: int) -> dict:
        fpath = self.fpaths[index]
        lab = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(self.cls2lab.keys()))
        cls = self.classes[index]
        spectrogram, waveform = file2spectrogram(fpath, self.wave_length)
        spectrum, waveform = file2spectrum(fpath, self.wave_length)
        mixup_flag = np.random.choice([0,1], 1)
        if self.mixup and mixup_flag: #  
            # mixup_coeff = np.random.beta(0.2, 0.2)
            mixup_coeff = np.random.uniform()*0.5
            # mixup on the wave level
            # random_index = np.random.choice([i for i in range(len(self.classes)) if self.classes[i] != cls])
            random_index = np.random.choice([i for i in range(len(self.classes)) if i != index])
            random_fpath = self.fpaths[random_index]
            random_lab = F.one_hot(torch.tensor(self.labels[random_index]), num_classes=len(self.cls2lab.keys()))
            random_cls = self.classes[random_index]
            random_waveform, sr = file2rawsig(random_fpath, self.wave_length, normalize=False)
            assert sr == 16000, "Error - Sample rate for file {} is not 16k.".format(random_fpath)
            mixup_waveform = mixup_coeff*waveform + (1-mixup_coeff)*random_waveform
            lab = mixup_coeff*lab + (1-mixup_coeff)*random_lab
            cls = cls + '_' + random_cls
            spectrogram = extract_spectrogram(mixup_waveform).unsqueeze(0)
            spectrum    = extract_spectrum(mixup_waveform).unsqueeze(0)
        else:
            lab = label_smoothing(lab, factor=np.random.uniform(0., .5))
            # lab = lab.float()
        return {
            'filepath'   : fpath,
            'spectrogram': spectrogram,
            'spectrum'   : spectrum,
            'label'      : lab,
            'class'      : cls,
            'mixup'      : 1 if self.mixup and mixup_flag else 0,
            'mixup_coeff': mixup_coeff if self.mixup and mixup_flag else -1,
            # 'mixup'      : 1 if self.mixup else 0,
            # 'mixup_coeff': mixup_coeff if self.mixup else -1,
            'domain'     : self.domain[index],
            'mtype'      : self.mtypes[index],
        }

class dcase2023_t2_testset(Dataset):
    def __init__(self, mtype, wave_length=10) -> None:
        super().__init__()
        data_dir = "./data/dcase2023_t2"
        dev_metadata_path= f"{data_dir}/metadata.tsv"
        eval_metadata_path=f"{data_dir}/test_metadata.tsv"
        assert os.path.exists(dev_metadata_path) and os.path.exists(eval_metadata_path), "metadata is missing!"
        df_dev_all = pd.read_csv(dev_metadata_path, sep='\t')
        df_dev_test = df_dev_all[(df_dev_all['train/test']=='test') & (df_dev_all.mtype == mtype)]
        df_eval_all = pd.read_csv(eval_metadata_path, sep='\t')
        df_eval_test = df_eval_all[df_eval_all.mtype == mtype]

        self.fpaths = df_dev_test['wave_path'].tolist() + df_eval_test['wave_path'].tolist()
        self.filenames = df_dev_test['filename'].tolist() + df_eval_test['filename'].tolist()
        self.wave_length = wave_length
        self.mtype = mtype

    def __len__(self) -> int:
        return self.fpaths.__len__()

    def __getitem__(self, index: int) -> dict:
        fpath = self.fpaths[index]
        spectrogram, _ = file2spectrogram(fpath, self.wave_length)
        spectrum, _ = file2spectrum(fpath, self.wave_length)
        return {
            'filepath'   : fpath,
            'filename'   : self.filenames[index],
            'spectrogram': spectrogram,
            'spectrum'   : spectrum,
            'mtype'      : self.mtype,
        }


