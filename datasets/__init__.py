# import os
# # ---------------------------------------
# from datasets.feature import *
# from datasets import dcase_dataset, MIMII_dataset, dcase22_dataset, dcase23_dataset
# from datasets import audioset, audioset2, dcase2023_t2

# # dataset & dataloader settings
# # print("## preparing datasets for training..")
# MIMII_CATEGORIES = ['fan', 'pump', 'slider', 'valve']
# ToyADMOS_CATEGORIES = ['ToyCar', 'ToyConveyor']
# MTYPE2ID = {
#     "fan":         [0, 1, 2, 3, 4, 5, 6],
#     "pump":        [0, 1, 2, 3, 4, 5, 6],
#     "slider":      [0, 1, 2, 3, 4, 5, 6],
#     "valve":       [0, 1, 2, 3, 4, 5, 6],
#     "ToyCar":      [1, 2, 3, 4, 5, 6, 7],
#     "ToyConveyor": [1, 2, 3, 4, 5, 6],
# }

# MTYPE2TRAIN = {
#     "fan":         [0, 2, 4, 6],
#     "pump":        [0, 2, 4, 6],
#     "slider":      [0, 2, 4, 6],
#     "valve":       [0, 2, 4, 6],
#     "ToyCar":      [1, 2, 3, 4],
#     "ToyConveyor": [1, 2, 3],
# }

# MTYPE2EVAL = {
#     "fan":         [1, 3, 5],
#     "pump":        [1, 3, 5],
#     "slider":      [1, 3, 5],
#     "valve":       [1, 3, 5],
#     "ToyCar":      [5, 6, 7],
#     "ToyConveyor": [4, 5, 6],
# }
