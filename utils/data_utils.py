import torchaudio, torch
import argparse, random, os, sys
import numpy as np
from scipy.io import wavfile

extract_spectrogram = torchaudio.transforms.Spectrogram(
                                                        n_fft=1024,
                                                        win_length=1024,
                                                        hop_length=512,
                                                        normalized=False,
                                                        power=1,
                                                        )
## if batch input, need to unsqueeze dim 1
extract_spectrum = lambda x: torch.abs(torch.fft.fft(x, n=x.shape[-1]*2)[...,:x.shape[-1]//2])


def adjust_size(wav, new_size):
    reps = int(np.ceil(new_size/wav.shape[1]))
    offset = np.random.randint(low=0, high=int(reps*wav.shape[1]-new_size+1))
    return torch.tile(wav, dims=(1,reps))[:, offset:offset+new_size]

def adjust_size_batch(wav_list, new_size):
    adjust_wav_list = []
    for wav in wav_list:
        adjust_wav_list.append(adjust_size(wav, new_size).view(-1).float())
    return torch.stack(adjust_wav_list)

def file2rawsig(fpath, wav_length=10, normalize=False):
    wf, sr = torchaudio.load(fpath, normalize=normalize)
    if wav_length is not None:
        return adjust_size(wf, wav_length*sr).view(-1).float(), sr
    else:
        return wf, sr

def file2spectrogram(fpath, wav_length=10):
    wf, sr = file2rawsig(fpath, wav_length, normalize=False)
    spectrogram = extract_spectrogram(wf)
    return spectrogram.unsqueeze(0), wf

def file2spectrum(fpath, wav_length=10):
    wf, sr = file2rawsig(fpath, wav_length, normalize=False)
    spectrum = extract_spectrum(wf)
    return spectrum.unsqueeze(0), wf

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def label_smoothing(labels: torch.tensor, factor=0.1):
    num_labels = labels.size()[-1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / (np.math.sqrt(sum(np.power(line, 2))) + sys.float_info.epsilon)
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


