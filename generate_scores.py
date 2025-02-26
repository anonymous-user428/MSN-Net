from utils.data_utils import *
import argparse, importlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Enter the model name")
parser.add_argument("--exp_dir", type=str, default="./exp")
parser.add_argument("--n_clusters", type=int, default=16)
parser.add_argument("--use_epoch", type=int, default=100)
parser.add_argument("--seed", type=int, default=7932)
args = parser.parse_args()

model = importlib.import_module(f"models.{args.model_name}")

seed_everything(args.seed)
np.seterr(divide='ignore',invalid='ignore')

if __name__ == "__main__":
    # Evaluate
    model.dcase2023_task2.scoring(args.exp_dir, n_clusters=args.n_clusters, num_epoch=args.use_epoch)


