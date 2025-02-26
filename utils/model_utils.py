import os, logging, re
from tqdm import tqdm
import torch, torch.nn as nn
from collections import defaultdict

def average_checkpoints(checkpoint_dir):
    """Average model parameters across all checkpoints in the given directory."""
    checkpoints = sorted([
        os.path.join(checkpoint_dir, ckpt) 
        for ckpt in os.listdir(checkpoint_dir) 
        if ckpt.endswith(".pth")
    ])[-5:]
    
    if not checkpoints:
        raise ValueError("No checkpoints found in the directory.")
    
    # Initialize an accumulator for model weights
    avg_state_dict = defaultdict(float)
    # cls_state_dict = defaultdict()
    num_checkpoints = len(checkpoints)
    
    for ckpt_path in tqdm(checkpoints):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_state_dict = checkpoint['encoder']
        for key, param in model_state_dict.items():
            avg_state_dict[key] += param / num_checkpoints
    
    # Convert defaultdict back to a standard dictionary
    avg_state_dict = {k: v for k, v in avg_state_dict.items()}

    return avg_state_dict


def get_last_checkpoint(checkpoint_dir):
    """Find the last checkpoint based on the epoch number in the filename."""
    # Regular expression to match filenames like model_epoch_xx.pth
    pattern = r"model_epoch_(\d+)\.pth"
    
    checkpoints = []
    
    for filename in os.listdir(checkpoint_dir):
        match = re.match(pattern, filename)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, filename))
    
    if not checkpoints:
        raise ValueError("No checkpoint files found in the directory.")
    
    # Sort by epoch number and get the last one
    checkpoints.sort(key=lambda x: x[0])  # Sort by epoch number
    last_checkpoint = checkpoints[-1][1]  # Get filename of the highest epoch
    
    return os.path.join(checkpoint_dir, last_checkpoint)


def setup_logging(log_file="training.log"):
    """
    Sets up logging for DDP training.
    Logs to:
    - File (all logs)
    - Console (only for rank 0) with selective logging
    """

    # Main logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (logs all messages)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (only for rank 0)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Additional loggers for console/file specific logging
    file_logger = logging.getLogger("FileLogger")
    file_logger.setLevel(logging.DEBUG)
    file_logger.addHandler(file_handler)  # File only, no console
    file_logger.propagate = False

    console_logger = logging.getLogger("ConsoleLogger")
    console_logger.setLevel(logging.DEBUG)
    console_logger.addHandler(console_handler)  # Console only, no file
    console_logger.propagate = False

    return logger, file_logger, console_logger

def kaiming_init(m):
    if isinstance(m, nn.Conv2d):  # For Conv2D layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):  # For Linear layers
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # Handle the initialization of other layer types if necessary

# Xavier normal initialization function
def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


