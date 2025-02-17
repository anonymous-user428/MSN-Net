# Multi-scale Network (MSN-Net)
This is the implementation of the MSN-Net.

The codes in the `official_evaluator` originated from the officials of DCASE Challenge Task 2, which can be found [here](https://github.com/nttcslab/dcase2023_task2_evaluator).

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

### Run

```
bash evaluate.sh <gpu_id>
```


