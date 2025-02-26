import argparse, os, glob
import pandas as pd
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--task',       help="the task that dataset belongs to")
args = parser.parse_args()


corpus_dir= f"<dir>/{args.task}"
os.system(f"mkdir -p ground_truth_data")
os.system(f"mkdir -p ground_truth_attributes")
os.system(f"mkdir -p ground_truth_domain")

# compute files for dev
files = sorted(glob.glob(os.path.join(corpus_dir, 'development/*/*/*.wav')), reverse=False)
metadata = []
for fp in tqdm(files, ncols=100, desc="DEV"):
    filename = fp.split('/')[-1]; mtype = fp.split('/')[-3]
    attribute = '.'.join(fp.split('/')[-1].split('.')[:-1])
    _metadata = attribute.split('_')
    if _metadata[3] == 'train': continue
    domain = 0 if _metadata[2] == 'source' else 1
    label = 0 if _metadata[4] == 'normal' else 1
    metadata.append([mtype, filename, domain, attribute, label])
df = pd.DataFrame(metadata, columns=['mtype', 'filename', 'domain', 'attributes', 'labels'])


for mtype in df.mtype.unique():
    print(mtype)
    df[df.mtype == mtype][['filename', 'domain']]\
        .to_csv(f"ground_truth_domain/ground_truth_{mtype}_section_00_test.csv", 
                index=False, 
                header=False)

    df[df.mtype == mtype][['filename', 'attributes']]\
        .to_csv(f"ground_truth_attributes/ground_truth_{mtype}_section_00_test.csv", 
                index=False, 
                header=False)

    df[df.mtype == mtype][['filename', 'labels']]\
        .to_csv(f"ground_truth_data/ground_truth_{mtype}_section_00_test.csv", 
                index=False, 
                header=False)


