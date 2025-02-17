from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pickle, os, glob, sys
import pandas as pd, numpy as np
sys.path.append("/DKUdata/2020/zhangyc/workspace/ASD/test/FTE-Net")
from utils.data_utils import *


# ICASSP 2024 version #
class dcase23_t2_dataset(Dataset):
    def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
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

# ICASSP 2024 version, beta low, label smoothing low #
class dcase23_t2_dataset_v2(Dataset):
    def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
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
            mixup_coeff = np.random.beta(0.2, 0.2)
            # mixup_coeff = np.random.uniform()*0.5
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
            lab = label_smoothing(lab, factor=np.random.uniform(0.01, .2))
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

# ICASSP 2024 version, beta low, label smoothing high #
class dcase23_t2_dataset_v3(Dataset):
    def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
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
            mixup_coeff = np.random.beta(0.2, 0.2)
            # mixup_coeff = np.random.uniform()*0.5
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

# testing area #
class dcase23_t2_dataset_raw(Dataset):
    def __init__(self, wave_length=10, mtype=None):
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
        assert os.path.exists(metadata_path), "Error - Metadata is missing!"
        df_all = pd.read_csv(metadata_path, sep='\t')
        df_train = df_all[df_all.label == 'normal']
        # if not train:
        # df_train = df_all[df_all['train/test']=='train']

        classes = df_train[['mtype', 'attributes']].drop_duplicates()
        self.cls2lab = dict([["_".join([c[0], *eval(c[1])]), i] for i, (_, c) in enumerate(classes.iterrows())])
        self.lab2cls = dict([[self.cls2lab[c], c] for c in self.cls2lab])

        df_train = df_all[df_all['train/test']=='train']
        if mtype is not None:
            df_train = df_train[df_train.mtype == mtype]
        self.mtypes = df_train.mtype.tolist()
        self.classes = ["_".join([c[0], *eval(c[1])]) for _, c in df_train[['mtype', 'attributes']].iterrows()]
        self.labels = [self.cls2lab[c] for c in self.classes]
        self.fpaths = df_train['wave_path'].tolist()
        self.domain = df_train['domain'].tolist()
        self.wave_length = wave_length

    def __len__(self) -> int:
        return self.classes.__len__()

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        wf, _ = file2rawsig(fpath, wav_length=self.wave_length)
        return {
            'filepath'   : fpath,
            'waveform'   : wf,
            'label'      : F.one_hot(torch.tensor(self.labels[index]), num_classes=len(self.cls2lab.keys())),
            'class'      : self.classes[index],
            'domain'     : self.domain[index],
            'mtype'      : self.mtypes[index],
        }

def collate_fn(batch):
    batch_size = len(batch)

    target_length = random.randint(8, 16) * 16000  # Convert seconds to samples
    filepaths = []
    waveforms = []
    labels = []
    classes = []
    domains = []
    mtypes = []
    spectrograms = []
    spectrums = []

    lmbs = []; mixup = np.random.randint(0, 2, size=batch_size)
    if not np.any(mixup): mixup[np.random.randint(0, batch_size)] = 1 # make sure at least one is mix-uped
    mixup = mixup.astype(bool)

    for i, item in enumerate(batch):
        filepaths.append(item['filepath'])
        labels.append(item['label'])
        classes.append(item['class'])
        domains.append(item['domain'])
        mtypes.append(item['mtype'])
        
        lmbs.append(np.random.uniform()*0.5)
        adjusted_waveform = adjust_size(item['waveform'], new_size=target_length).view(-1).float()
        waveforms.append(adjusted_waveform)

    waveforms = torch.stack(waveforms)
    lmbs = torch.tensor(lmbs)
    ori_labels = labels

    # lmb = np.random.beta(.2, .2)
    
    for i, each_wav in enumerate(waveforms):
        if mixup[i]:
            mixup_waveform = lmbs[i]*each_wav + (1-lmbs[i])*torch.flip(each_wav.clone(), dims=[0])
            labels[i] = lmbs[i]*labels[i] + (1-lmbs[i])*torch.flip(labels[i].clone(), dims=[0])
            spectrograms.append(extract_spectrogram(mixup_waveform))
            spectrums.append(extract_spectrum(mixup_waveform))
        else:
            spectrograms.append(extract_spectrogram(each_wav))
            spectrums.append(extract_spectrum(each_wav))
            # label_smoothing
            labels[i] = label_smoothing(labels[i], factor=np.random.uniform(0., .5))

    labels = torch.stack(labels)
    ori_labels = torch.stack(ori_labels)
    spectrograms = torch.stack(spectrograms).unsqueeze(1)
    spectrums = torch.stack(spectrums).unsqueeze(1)

    return {
        'filepaths': filepaths,
        'waveforms': waveforms,
        'labels': labels,
        'classes': classes,
        'domains': domains,
        'mtypes': mtypes,
        'spectrogram': spectrograms,
        'spectrum': spectrums,
        'mixup'   : torch.from_numpy(mixup).int(),
        'mixup_coeff': lmbs, 
        'ori_labels': ori_labels,
    }

class dcase2023_t2_testset_raw(Dataset):
    def __init__(self, mtype, wave_length=10) -> None:
        super().__init__()
        data_dir = "./data/dcase2023_t2"
        dev_metadata_path= f"{data_dir}/metadata2.csv"
        eval_metadata_path=f"{data_dir}/test_metadata.csv"
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
        wf, _ = file2rawsig(fpath, wav_length=self.wave_length)
        return {
            'filepath'   : fpath,
            'filename'   : self.filenames[index],
            'waveform'   : wf,
            'mtype'      : self.mtype,
        }

# class dcase23_t2_dataset(Dataset):
#     def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
#         super().__init__()
#         data_dir = "./data"
#         task = "dcase2023_t2"
#         metadata_path= f"{data_dir}/{task}/metadata.csv"
#         assert os.path.exists(metadata_path), "Error - Metadata is missing!"
#         df_all = pd.read_csv(metadata_path, sep='\t')
#         df_train = df_all[df_all.label == 'normal']
#         if not train:
#             df_train = df_all[df_all['train/test']=='train']

#         classes = df_train[['mtype', 'attributes']].drop_duplicates()
#         self.cls2lab = dict([["_".join([c[0], *eval(c[1])]), i] for i, (_, c) in enumerate(classes.iterrows())])
#         self.lab2cls = dict([[self.cls2lab[c], c] for c in self.cls2lab])

#         df_train = df_all[df_all['train/test']=='train']
#         if mtype:
#             df_train = df_train[df_train.mtype == mtype]
#         self.mtypes = df_train.mtype.tolist()
#         self.classes = ["_".join([c[0], *eval(c[1])]) for _, c in df_train[['mtype', 'attributes']].iterrows()]
#         self.labels = [self.cls2lab[c] for c in self.classes]
#         self.fpaths = df_train['wave_path'].tolist()
#         self.domain = df_train['domain'].tolist()
#         self.wave_length = wave_length
#         # self.mixup = mixup

#     def __len__(self) -> int:
#         return self.classes.__len__()

#     def __getitem__(self, index: int) -> dict:
#         fpath = self.fpaths[index]
#         lab = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(self.cls2lab.keys()))
#         cls = self.classes[index]
#         spectrogram, waveform = file2spectrogram(fpath, self.wave_length)
#         spectrum, waveform = file2spectrum(fpath, self.wave_length)
#         return {
#             'filepath'   : fpath,
#             'spectrogram': spectrogram,
#             'spectrum'   : spectrum,
#             'label'      : lab,
#             'class'      : cls,
#             # 'mixup'      : 1 if self.mixup and mixup_flag else 0,
#             # 'mixup_coeff': mixup_coeff if self.mixup and mixup_flag else -1,
#             'domain'     : self.domain[index],
#             'mtype'      : self.mtypes[index],
#         }




# Not working versions #
class dcase23_t2_dataset_mini(Dataset):
    def __init__(self, wave_length=10, mixup=True, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
        assert os.path.exists(metadata_path), "Error - Metadata is missing!"
        df_all = pd.read_csv(metadata_path, sep='\t')
        df_train = df_all[df_all.label == 'normal']
        if not train:
            df_train = df_all[df_all['train/test']=='train']

        classes = df_train[['mtype', 'attributes']].drop_duplicates()
        self.cls2lab = dict([["_".join([c[0], *eval(c[1])]), i] for i, (_, c) in enumerate(classes.iterrows())])
        self.lab2cls = dict([[self.cls2lab[c], c] for c in self.cls2lab])

        df_train = df_all[df_all['train/test']=='train']
        if mtype:
            df_train = df_train[df_train.mtype == mtype]
        self.mtypes = df_train.mtype.tolist()
        self.classes = ["_".join([c[0], *eval(c[1])]) for _, c in df_train[['mtype', 'attributes']].iterrows()]
        self.labels = [self.cls2lab[c] for c in self.classes]
        self.fpaths = df_train['wave_path'].tolist()
        self.domain = df_train['domain'].tolist()
        self.wave_length = wave_length
        self.evalmode = (not train)
        self.mixup = mixup

    def __len__(self) -> int:
        return self.classes.__len__()

    def __getitem__(self, index: int) -> dict:
        fpath = self.fpaths[index]
        lab = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(self.cls2lab.keys()))
        cls = self.classes[index]
        spectrogram, waveform = file2spectrogram2(fpath, max_frames=200, evalmode=self.evalmode)
        spectrum, waveform = file2spectrum2(fpath, max_frames=200, evalmode=self.evalmode)
        mixup_flag = np.random.choice([0,1], 1)
        if self.mixup and mixup_flag:
            # mixup_coeff = np.random.beta(0.2, 0.2)
            mixup_coeff = np.random.uniform()*0.5
            # mixup on the wave level
            # random_index = np.random.choice([i for i in range(len(self.classes)) if self.classes[i] != cls])
            random_index = np.random.choice([i for i in range(len(self.classes)) if i != index])
            random_fpath = self.fpaths[random_index]
            random_lab = F.one_hot(torch.tensor(self.labels[random_index]), num_classes=len(self.cls2lab.keys()))
            random_cls = self.classes[random_index]
            random_waveform, sr = loadWAV(random_fpath, max_frames=200, evalmode=False)
            assert sr == 16000, "Error - Sample rate for file {} is not 16k.".format(random_fpath)
            mixup_waveform = mixup_coeff*waveform + (1-mixup_coeff)*random_waveform
            lab = mixup_coeff*lab + (1-mixup_coeff)*random_lab
            cls = cls + '_' + random_cls
            spectrogram = extract_spectrogram(mixup_waveform).unsqueeze(0)
            spectrum    = extract_spectrum(mixup_waveform).unsqueeze(0)
        else:
            lab = label_smoothing(lab, factor=np.random.uniform(0., .5))
        return {
            'filepath'   : fpath,
            'spectrogram': spectrogram,
            'spectrum'   : spectrum,
            'label'      : lab,
            'class'      : cls,
            'mixup'      : 1 if self.mixup and mixup_flag else 0,
            'mixup_coeff': mixup_coeff if self.mixup and mixup_flag else -1,
            'domain'     : self.domain[index],
            'mtype'      : self.mtypes[index],
        }

class dcase23_t2_dataset_rnn(Dataset):
    def __init__(self, wave_length=10, mixup=False, train=True, mtype=None) -> None:
        super().__init__()
        data_dir = "./data"
        task = "dcase2023_t2"
        metadata_path= f"{data_dir}/{task}/metadata.csv"
        assert os.path.exists(metadata_path), "Error - Metadata is missing!"
        df_all = pd.read_csv(metadata_path, sep='\t')
        df_train = df_all[df_all.label == 'normal']
        if not train:
            df_train = df_all[df_all['train/test']=='train']

        classes = df_train[['mtype', 'attributes']].drop_duplicates()
        self.cls2lab = dict([["_".join([c[0], *eval(c[1])]), i] for i, (_, c) in enumerate(classes.iterrows())])
        self.lab2cls = dict([[self.cls2lab[c], c] for c in self.cls2lab])

        df_train = df_all[df_all['train/test']=='train']
        if mtype:
            df_train = df_train[df_train.mtype == mtype]
        self.mtypes = df_train.mtype.tolist()
        self.classes = ["_".join([c[0], *eval(c[1])]) for _, c in df_train[['mtype', 'attributes']].iterrows()]
        self.labels = [self.cls2lab[c] for c in self.classes]
        self.fpaths = df_train['wave_path'].tolist()
        self.domain = df_train['domain'].tolist()
        self.wave_length = wave_length
        self.mixup = mixup

    def __len__(self) -> int:
        return self.classes.__len__()

    def __getitem__(self, index: int) -> dict:
        fpath = self.fpaths[index]
        lab = F.one_hot(torch.tensor(self.labels[index]), num_classes=len(self.cls2lab.keys()))
        cls = self.classes[index]
        spectrogram, waveform = file2spectrogram3(fpath, max_frames=200, evalmode=True)
        spectrum, waveform = file2spectrum3(fpath, max_frames=200, evalmode=True)
        mixup_flag = np.random.choice([0,1], 1)
        if self.mixup and mixup_flag:
            # mixup_coeff = np.random.beta(0.2, 0.2)
            mixup_coeff = np.random.uniform()*0.5
            # mixup on the wave level
            # random_index = np.random.choice([i for i in range(len(self.classes)) if self.classes[i] != cls])
            random_index = np.random.choice([i for i in range(len(self.classes)) if i != index])
            random_fpath = self.fpaths[random_index]
            random_lab = F.one_hot(torch.tensor(self.labels[random_index]), num_classes=len(self.cls2lab.keys()))
            random_cls = self.classes[random_index]
            random_waveform, sr = loadWAV(random_fpath, max_frames=200, evalmode=False)
            assert sr == 16000, "Error - Sample rate for file {} is not 16k.".format(random_fpath)
            mixup_waveform = mixup_coeff*waveform + (1-mixup_coeff)*random_waveform
            lab = mixup_coeff*lab + (1-mixup_coeff)*random_lab
            cls = cls + '_' + random_cls
            spectrogram = extract_spectrogram(mixup_waveform).unsqueeze(0)
            spectrum    = extract_spectrum(mixup_waveform).unsqueeze(0)
        else:
            lab = label_smoothing(lab, factor=np.random.uniform(0., .5))
        return {
            'filepath'   : fpath,
            'spectrogram': spectrogram,
            'spectrum'   : spectrum,
            'label'      : lab,
            'class'      : cls,
            'mixup'      : 1 if self.mixup and mixup_flag else 0,
            'mixup_coeff': mixup_coeff if self.mixup and mixup_flag else -1,
            'domain'     : self.domain[index],
            'mtype'      : self.mtypes[index],
        }

class dcase2023_t2_testset(Dataset):
    def __init__(self, mtype, wave_length=10) -> None:
        super().__init__()
        data_dir = "./data/dcase2023_t2"
        dev_metadata_path= f"{data_dir}/metadata2.csv"
        eval_metadata_path=f"{data_dir}/test_metadata.csv"
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

class dcase2023_t2_testset_mini(Dataset):
    def __init__(self, mtype, wave_length=10) -> None:
        super().__init__()
        data_dir = "./data/dcase2023_t2"
        dev_metadata_path= f"{data_dir}/metadata2.csv"
        eval_metadata_path=f"{data_dir}/test_metadata.csv"
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
        spectrogram, _ = file2spectrogram2(fpath, max_frames=200, evalmode=True)
        spectrum, _ = file2spectrum2(fpath, max_frames=200, evalmode=True)
        return {
            'filepath'   : fpath, 
            'filename'   : self.filenames[index], 
            'spectrogram': spectrogram, 
            'spectrum'   : spectrum, 
            'mtype'      : self.mtype, 
        }

class dcase2023_t2_testset_rnn(Dataset):
    def __init__(self, mtype, wave_length=10) -> None:
        super().__init__()
        data_dir = "./data/dcase2023_t2"
        dev_metadata_path= f"{data_dir}/metadata2.csv"
        eval_metadata_path=f"{data_dir}/test_metadata.csv"
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
        spectrogram, _ = file2spectrogram3(fpath, max_frames=200, evalmode=True)
        spectrum, _ = file2spectrum3(fpath, max_frames=200, evalmode=True)
        return {
            'filepath'   : fpath, 
            'filename'   : self.filenames[index], 
            'spectrogram': spectrogram, 
            'spectrum'   : spectrum, 
            'mtype'      : self.mtype, 
        }





if __name__ == "__main__":
    seed_everything(3407)
    # dataset = dcase23_t2_dataset_raw("slider")
    # for each in dataset:
    #     spectrogram, spectrum, label, domain, attribute = each.values()
    #     print(spectrogram.size(), spectrum.size(), attribute.size(), label.item(), domain.item())
        # break

    # dataset = dcase2023_t2_testset('ToyCar', wave_length=18)
    dataset = dcase23_t2_dataset_raw(wave_length=None)
    # for i, data in enumerate(dataset):
    #     print(data['waveform'].shape)
        # for k in data:
        #     print(f"{k}: {data[k]}")
        # break

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=16, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        _, waveforms, labels, _, domain, _, spectrogram, spectrum, mixup = batch.values()
        # _, filename, spectrogram, spectrum, _ = batch.values()
        print(mixup)
        print(waveforms.shape, spectrogram.size(), spectrum.size(), labels.shape)
        # print(filename)
        # break

