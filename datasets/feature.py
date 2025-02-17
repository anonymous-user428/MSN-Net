import librosa, torchaudio
import numpy as np, random, sys
import torch, torch.nn as nn, torch.nn.functional as F

## same as dcase2020 baseline system
def file_load(file_name, mono=True, sr=None, **kwargs):
    try:
        return librosa.load(file_name, sr=sr, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(file_name))

def file_to_mel(
        file_name,
        frames=1,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        power=2.0, 
        device='cpu',
        random_clip=False,
        scale=False,
        **kwargs
    ):
    y, sr = file_load(file_name, **kwargs)
    if y.shape[0] == 0: return y, 0
    if len(y) != 160000 and random_clip:
        ## random select 160000 samples
        rand_st = random.randint(0, len(y)-160000-1)
        y = y[rand_st:rand_st+160000]
    # y = (y - y.mean()) / (y.std()+sys.float_info.epsilon)
    # assert sum(np.isnan(y)) == 0, "File loading issue! file_name: {}".format(file_name)
    dims = n_mels * frames
    spectrogram_fn = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power).to(device)
    mel_scale_fn = torchaudio.transforms.MelScale(n_mels=n_mels, n_stft=n_fft//2+1).to(device)
    spectrogram = spectrogram_fn(torch.tensor(y, dtype=torch.float32).to(device))
    mel_spectrogram = mel_scale_fn(spectrogram)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims))
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
    if scale:
        vector_array = 2*(vector_array - vector_array.min())/(vector_array.max()-vector_array.min())-1
    return vector_array.astype(np.float32), y

def file_to_linear(
        file_name,
        frames=1,
        n_fft=1024,
        hop_length=512,
        power=2.0, 
        device='cpu',
        **kwargs
    ):
    y, sr = file_load(file_name, **kwargs)
    y = (y - y.mean()) / y.std()
    # assert sum(np.isnan(y)) == 0, "File loading issue!"
    dims = (n_fft//2+1)*frames
    spectrogram_fn = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power).to(device)
    spectrogram = spectrogram_fn(torch.tensor(y, dtype=torch.float32).to(device))
    log_spectrogram = 20.0 / power * np.log10(spectrogram + sys.float_info.epsilon)
    vector_array_size = len(log_spectrogram[0, :]) - frames + 1
    if vector_array_size < 1:
        return np.empty((0, dims))
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, (n_fft//2+1) * t: (n_fft//2+1) * (t + 1)] = log_spectrogram[:, t: t + vector_array_size].T
    return vector_array.astype(np.float32)

def file_to_sig(
        file_name,
        **kwargs,
    ):
    y, sr = file_load(file_name, **kwargs)
    return y

class Pipeline(nn.Module):
    r'''
    Pitch shift
        Randomly increase or decrease the pitch of the audio. 

    Time stretch
        Change the speed of the audio by a pre-defined rate. [0.5, 2]
        Re-sample the resulting audio to make audio length consistent. 

    white noise injection
        Choose a suitable SNR, e.g. [-6, 6] 

    Fade in / Fade out
        At the begining and in the end of the audio. 
        Linear/logarithmic/exponential/quarter sinusoidal/half sinusoidal 

    Time shifting
        Shifts the audio signal forward or backward by randomly picked shift length, e.g. [0, L/2] 

    Time masking
        Randomly selects a segment of the signal and set it equal to zero or another constant value.  
        The size is randomly chosen to a value less than 1/10 of the signal’s length. 

    Frequency masking 
        randomly removes a segment of frequencies of the audio. 
        The length is randomly chosen to a value less than 1/10 of the signal’s frequency length. 
    '''
    def __init__(
        self, 
        n_mels=128,
        mel = True,
        n_fft=1024,
        hop_length=512,
        power=2.0, 
        device='cpu',
        frames=1,
        scale=False, 
        **kwargs
    ) -> None:
        super().__init__()
        self.device = device
        # augmentation types
        self.sig_aug_types = ["pitch_shift", "time_stretch", "white_noise_injection", "fade", "time_shift"]
        # self.sig_aug_types = ["white_noise_injection", "fade", "time_shift"]
        self.spec_aug_types = ["time_masking", "freq_masking"]
        # self.spec_aug_types = []
        # mel-spectrogram
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=power)
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, n_stft=n_fft//2+1)
        self.spectrogram = self.spectrogram.to(device); self.spectrogram.eval()
        self.mel_scale = self.mel_scale.to(device); self.mel_scale.eval()
        self.mel = mel
        self.frames = frames
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.scale = scale

    def _pitch_shift(self, signal, sr=16000, **kwargs) -> torch.tensor:
        shifted_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=random.randint(-10, 10), bins_per_octave=12)
        return shifted_signal

    def _time_stretch(self, signal, sr=16000, **kwargs) -> torch.tensor:
        stretched_signal = librosa.effects.time_stretch(signal, rate=random.uniform(0.5, 2))
        _sr = sr*(len(stretched_signal) / len(signal))
        signal = librosa.resample(stretched_signal, orig_sr=_sr, target_sr=sr)  # recovery
        return signal

    def _white_noise_injection(self, signal, sr=16000, **kwargs) -> torch.tensor:
        white = np.random.randn(signal.shape[0])
        snr = random.uniform(-6, 6)
        sigma_n = np.sqrt(10 ** (- snr / 10))
        return signal + white * sigma_n

    def _fade(self, signal, sr=16000, **kwargs) -> torch.tensor:
        fade_type = ["linear", "logarithmic", "exponential", "quarter_sine", "half_sine"]
        selected_type = random.choice(fade_type)
        length = random.randint(0, round(len(signal)/2))
        transform = torchaudio.transforms.Fade(length, length, selected_type)
        return transform(torch.tensor(signal, dtype=torch.float32)).to(self.device)

    def _time_shift(self, signal, sr=16000, **kwargs) -> torch.tensor:
        length = len(signal)
        signal = np.concatenate([signal, signal, signal])
        start_pt = np.int64(random.random()*(2*length))
        return signal[start_pt:start_pt+length]

    def _time_masking(self, spectrogram, **kwargs) -> torch.tensor:
        masking = torchaudio.transforms.TimeMasking(time_mask_param=spectrogram.shape[1]/10)
        return masking(spectrogram)

    def _freq_masking(self, spectrogram, **kwargs) -> torch.tensor:
        masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=spectrogram.shape[1]/10)
        return masking(spectrogram)

    def forward(self, signal):
        total_aug = self.sig_aug_types+self.spec_aug_types
        aug_lab = random.choice([x for x in range(len(total_aug))])
        # if aug_lab >= len(total_aug):
        #     aug_type = "raw"
        # else:
        aug_type = total_aug[aug_lab]
        if aug_type in self.sig_aug_types:
            aug_func = eval("self._"+aug_type)
            signal = aug_func(signal)
        if not torch.is_tensor(signal):
            signal = torch.tensor(signal, dtype=torch.float32).to(self.device)
        spectrogram = self.spectrogram(signal).unsqueeze(0)
        if self.mel: 
            step_sz = self.n_mels
            spectrogram = self.mel_scale(spectrogram)
        else:
            step_sz = (self.n_fft//2+1)
        log_spectrogram = 20.0 / 2 * np.log10(spectrogram.squeeze(0) + sys.float_info.epsilon)
        if aug_type in self.spec_aug_types:
            aug_func = eval("self._"+aug_type)
            spectrogram = aug_func(spectrogram)
        # print("after", spectrogram.shape)
        dims = step_sz * self.frames; vector_array_size = len(log_spectrogram[0, :]) - self.frames + 1
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(self.frames):
            vector_array[:, step_sz * t: step_sz * (t + 1)] = log_spectrogram[:, t: t + vector_array_size].T
        if self.scale:
            vector_array = 2*(vector_array - vector_array.min())/(vector_array.max()-vector_array.min())-1
        return vector_array.astype(np.float32), aug_lab

