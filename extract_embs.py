from utils.data_utils import *
from utils.model_utils import *
import argparse, importlib, glob, re
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--exp_dir",    type=str, default='./exp')
parser.add_argument("--wave_length",type=int, default=10)
parser.add_argument("--gpu", type=str, default='6')
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--model_name", type=str, default='model_rnn')
parser.add_argument("--use_epoch", type=int, default=100)
args = parser.parse_args()

model = importlib.import_module(f"models.{args.model_name}")

SEED=args.seed
seed_everything(SEED)
np.seterr(divide='ignore',invalid='ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# print("Use GPU: %s" % args.gpu)
device = torch.device('cuda')

if __name__ == "__main__":
    feature_extractor = model.dcase2023_task2(batch_size=args.batch_size,
                                              exp_dir=args.exp_dir,
                                              wave_length=args.wave_length,
                                              device=device,
                                              SEED=SEED)
    # Load checkpoint
    model_path = f"{args.exp_dir}/{SEED}_model_epoch_{args.use_epoch}.pth"
    if os.path.exists(model_path):
        feature_extractor.load_checkpoint(model_path)
    else:
        raise ValueError

    exp_dir = "/".join(args.exp_dir.split('/')[:-1] + ['embs'])
    for mtype in ["ToyCar", "ToyTrain", "fan", "valve", "slider", "gearbox", "bearing", "bandsaw", "grinder", "shaker", "ToyDrone", "ToyNscale", "ToyTank", "Vacuum"]:
        postfix = args.use_epoch if args.use_epoch != -1 else 'average'
        emb_folder = f"{exp_dir}/seed_{SEED}/{mtype}/embs_{postfix}"
        os.system(f"mkdir -p {emb_folder}")
        feature_extractor.extract_embs(mtype, out_path=emb_folder)


