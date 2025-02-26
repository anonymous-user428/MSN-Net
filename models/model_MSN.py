import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np, pandas as pd
import os, re, glob
from tqdm import tqdm
from sklearn.cluster import KMeans
# ----------------------------------------
from models.modules.MSN_net import *
from datasets import dcase2023_t2
from utils.data_utils import *
from utils.model_utils import *

class NLL_loss_onehot(nn.Module):
    def __init__(self):
        super(NLL_loss_onehot, self).__init__()

    def forward(self, output, target):
        loss = (-(output+1e-7).log() * target).sum(dim=1).mean()
        return loss

class dcase2023_task2:
    def __init__(self, 
                 n_epochs=10, 
                 batch_size=32,
                 device=torch.device('cpu'),
                 exp_dir="./exp",
                 wave_length=18,
                 SEED=3407,
                 logfile='./logs.tmp', 
                 **kwargs,
                 ) -> None:

        self.main_logger, self.file_logger, self.console_logger = setup_logging(logfile)

        seed_everything(SEED)
        self.seed = SEED
        self.n_epochs = n_epochs
        self.start_epoch = 0
        self.batch_size = batch_size
        self.device = device
        self.exp_dir = exp_dir
        self.wave_length = wave_length
        self.curr_epoch = self.start_epoch

        self.encoder = emb_model_MSN(spectrogram_size=(1, 513, 563), spectrum_size=(1, 144000)).to(device)


    def load_checkpoint(self, model):
        self.main_logger.info(f"loading model from {model}...")
        chkpts = torch.load(model, map_location=self.device)
        self.encoder.load_state_dict(chkpts['encoder'])
        

    def extract_embs(self, mtype, out_path, test=True) -> None:
        self.encoder.eval()
        torch.set_grad_enabled(False)
        train_dataset = dcase2023_t2.dcase23_t2_dataset(mtype=mtype, wave_length=self.wave_length, mixup=False)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        embs_info = []
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Extracting {mtype} 4 normal", ncols=100)):
            _, spectrogram, spectrum, _, _, _, _, domain, _ = batch.values()
            input = {
                "spectrogram": spectrogram.to(self.device),
                "spectrum"   : spectrum.to(self.device),
            }
            emb = self.encoder(input)
            emb = emb['combo_emb']
            # emb = self.encoder(spectrogram.to(self.device))
            embs_info.append([domain, emb.cpu().detach().numpy()])
        domain, embs = zip(*embs_info)
        embs = length_norm(np.vstack(embs))
        torch.save({'domain': np.hstack(domain), 'embs':embs}, f"{out_path}/normal_{mtype}.emb")

        if test:
            test_dataset = dcase2023_t2.dcase2023_t2_testset(mtype, wave_length=self.wave_length)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            embs_info = []
            for i, batch in enumerate(tqdm(test_dataloader, desc=f"Extracting {mtype} 4 test", ncols=100)):
                _, filename, spectrogram, spectrum, _ = batch.values()
                input = {
                    "spectrogram": spectrogram.to(self.device),
                    "spectrum"   : spectrum.to(self.device),
                }
                emb = self.encoder(input)
                emb = emb['combo_emb']
                # emb = self.encoder(spectrogram.to(self.device))
                assert torch.sum(torch.isnan(emb.cpu())).item() == 0, f"{filename}, {torch.sum(torch.isnan(emb.cpu())).item()}"
                embs_info.append([filename, emb.cpu().detach().numpy()])
            filenames, embs = zip(*embs_info)
            torch.save({'filenames':np.hstack(filenames), 'embs':length_norm(np.vstack(embs))}, f"{out_path}/test_{mtype}.emb")

        return

    @staticmethod
    def scoring(exp_dir, num_epoch, n_clusters=16, output_folder=None, seed=7932):
        mtypes = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve', "bandsaw", "grinder", "shaker", "ToyDrone", "ToyNscale", "ToyTank", "Vacuum"]
        for mtype in mtypes:
            subsys_exps = glob.glob(os.path.join(exp_dir, f"seed_*"))
            table1 = {}; table2 = {}
            table1 = {
                'filename': None,
                'score'   : None,
            }
            table2 = table1.copy()
            for subsys_exp in subsys_exps:
                emb_dict = torch.load(f"{subsys_exp}/{mtype}/embs_{num_epoch}/normal_{mtype}.emb")
                domains, embs = emb_dict.values()
                source_index = [i for i, d in enumerate(domains) if d == 'source']
                target_index = [i for i, d in enumerate(domains) if d == 'target']
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(embs[source_index, :])
                c_embs = np.zeros((len(target_index)+n_clusters, embs.shape[1]))
                c_embs[:n_clusters, :] = kmeans.cluster_centers_
                c_embs[n_clusters:, :] = embs[target_index, :]

                # Generate anomaly scores
                emb_dict = torch.load(f"{subsys_exp}/{mtype}/embs_{num_epoch}/test_{mtype}.emb")
                filenames, embs = emb_dict.values()
                y_pred = np.min(1-np.dot(embs, c_embs.T), axis=-1)
                if table1['filename'] is None: table1['filename'] = filenames.tolist()
                table1['score'] = y_pred if table1['score'] is None else table1['score'] + y_pred

                # Generate decision results
                if table2['filename'] is None: table2['filename'] = filenames.tolist()
                table2['score'] = np.ones(len(filenames)) if table2['score'] is None else table2['score'] + np.ones(len(filenames))

            table1['score'] /= len(subsys_exps)
            table2['score'] /= len(subsys_exps)
            df1 = pd.DataFrame(table1); df2 = pd.DataFrame(table2)
            if output_folder is not None:
                df1.to_csv(f"{output_folder}/anomaly_score_{mtype}_section_00_test.csv.csv")
                df2.to_csv(f"{output_folder}/decision_result_{mtype}_section_00_test.csv.csv")
            df1.to_csv(f"{exp_dir}/anomaly_scores/anomaly_score_{mtype}_section_00_test.csv", index=False, header=False)
            df2.to_csv(f"{exp_dir}/anomaly_scores/decision_result_{mtype}_section_00_test.csv", index=False, header=False)


