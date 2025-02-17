#!/usr/bin/bash

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo ${timestamp}

task=dcase2023_t2
data_dir=./data
gpu=$1

n_epochs=100
lr=1e-3
weight_decay=0
batch_size=64
use_epoch=$n_epochs
use_mixup=1

wave_length=18
model=model_MSN
exp_name=experiment_${model}

cat_dir=./exp/${exp_name}
mkdir -p $cat_dir

echo "Stage 1. train embeddings"
declare -a seeds=(3407 7932 12388)
for SEED in ${seeds[@]}; do
    python extract_embs.py --batch_size ${batch_size} \
                            --gpu ${gpu} \
                            --exp_dir ${cat_dir}/chkpts \
                            --wave_length ${wave_length} \
                            --model_name $model \
                            --use_epoch $use_epoch \
                            --seed ${SEED}

done

echo "Stage 2. generate the scores"
mkdir -p ${cat_dir}/anomaly_scores
python generate_scores.py --exp_dir ${cat_dir}/embs --model_name $model --n_clusters 16 --use_epoch $use_epoch


echo "Stage 3. evaluate"
evaluator_path=./official_evaluator
mkdir -p ${evaluator_path}/teams/${exp_name}
system_path=${evaluator_path}/teams/${exp_name}/${model}
mkdir -p $system_path

declare -a MachineTypes=("ToyCar" "ToyTrain" "fan" "valve" "slider" "gearbox" "bearing" "bandsaw" "grinder" "shaker" "ToyDrone" "ToyNscale" "ToyTank" "Vacuum")
for mtype in ${MachineTypes[@]}; do
    echo $mtype
    cp ${cat_dir}/anomaly_scores/anomaly_score_${mtype}_section_00_test.csv ${system_path}
    cp ${cat_dir}/anomaly_scores/decision_result_${mtype}_section_00_test.csv ${system_path}
done

cd official_evaluator
python dcase2023_task2_evaluator.py --teams_root_dir=./teams/${exp_name} --dir_depth=1
cd ..
cp ${evaluator_path}/teams_result/${model}_result.csv ${cat_dir}/results.csv


