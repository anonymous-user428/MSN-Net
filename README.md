# Multi-scale Network (MSN-Net)
This is the implementation of the MSN-Net.

### Data Preparation

Prepare the metadata.csv as follows, where the wave_path should be the path to the wave files.
```
section_id	domain	train/test	label	wave_id	attributes	filename	mtype	wave_path
00	source	test	anomaly	0001	['car', 'B1', 'spd', '31V', 'mic', '1']	section_00_source_test_anomaly_0001_car_B1_spd_31V_mic_1.wav	ToyCar	<dir>/dcase2023_t2/development/ToyCar/test/section_00_source_test_anomaly_0001_car_B1_spd_31V_mic_1.wav
00	source	test	anomaly	0002	['car', 'B1', 'spd', '31V', 'mic', '1']	section_00_source_test_anomaly_0002_car_B1_spd_31V_mic_1.wav	ToyCar	<dir>/dcase2023_t2/development/ToyCar/test/section_00_source_test_anomaly_0002_car_B1_spd_31V_mic_1.wav
```

Prepare the test_metadata.csv as follows, where the wave_path should be the path to the wave files.
```
filename	mtype	wave_path
section_00_0000.wav	ToyDrone	<dir>/dcase2023_t2/evaluation/ToyDrone/test/section_00_0000.wav
section_00_0001.wav	ToyDrone	<dir>/dcase2023_t2/evaluation/ToyDrone/test/section_00_0001.wav
```

### Download Model Checkpoints

Please download the all the models from [huggingface](https://huggingface.co/anonymous-user428/MSE-Net), and put them under `exp/experiment_model_MSN/chkpts`.

### Run

```
bash evaluate.sh <gpu_id>
```

### Citation
The codes in the `official_evaluator` originated from the officials of DCASE Challenge Task 2, which can be found [here](https://github.com/nttcslab/dcase2023_task2_evaluator). We also borrow some codes from [Sub-Cluster Adacos](https://github.com/wilkinghoff/sub-cluster-AdaCos). Thanks for their kindly open-sourced codes.

- Kota Dohi, Keisuke Imoto, Noboru Harada, Daisuke Niizumi, Yuma Koizumi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, and Yohei Kawaguchi, "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring," in arXiv-eprints: 2305.07828, 2023.
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, Shoichiro Saito, "ToyADMOS2: Another Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection under Domain Shift Conditions," in Proc. DCASE 2022 Workshop, 2022.
- Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi, "MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task," in Proc. DCASE 2022 Workshop, 2022.
- Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, "First-Shot Anomaly Detection for Machine Condition Monitoring: A Domain Generalization Baseline," in arXiv e-prints: 2303.00455, 2023.
- Wilkinghoff, Kevin. "Sub-cluster AdaCos: Learning representations for anomalous sound detection." in Proc. IJCNN, IEEE, 2021.
