"""Core configuration and launch handling for distributed training."""

import argparse
import datetime
import os
import re
import shutil
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utility_funcs import prompt_interaction, convert_string_to_boolean, integer_to_string_representation
import configuration_template as config_template

# Initialize parser for command line arguments
parser = argparse.ArgumentParser(description='Dynamic Scene Deblurring Configuration')

# Group for Device settings
device_opts = parser.add_argument_group('Device Options')
device_opts.add_argument('--seed', type=int, default=-1, help='Random seed for initialization')
device_opts.add_argument('--workers', type=int, default=7, help='Number of worker threads for data loading')
device_opts.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device for computation')
device_opts.add_argument('--gpu_id', type=int, default=0, help='GPU ID for computation')
device_opts.add_argument('--gpus', type=int, default=1, help='Number of GPUs to utilize')
device_opts.add_argument('--use_ddp', type=convert_string_to_boolean, default=False, help='Use DistributedDataParallel for training')
device_opts.add_argument('--init_from_launch', type=convert_string_to_boolean, default=False, help='Set if launched from a cluster script')

# Specifications for distributed computation
device_opts.add_argument('--master_ip', type=str, default='localhost', help='IP address for master node')
device_opts.add_argument('--master_port', type=integer_to_string_representation, default='29500', help='Port for master node communication')
device_opts.add_argument('--backend', type=str, default='nccl', help='Backend for distributed training')
device_opts.add_argument('--init_method', type=str, default='env://', help='Initialization method for distributed training')
device_opts.add_argument('--process_rank', type=int, default=0, help='Rank of this process in the distributed setup')
device_opts.add_argument('--total_gpus', type=int, default=1, help='Total number of GPUs across all nodes')

# Data handling arguments
data_group = parser.add_argument_group('Data Handling')
data_group.add_argument('--data_dir', type=str, default='~/datasets', help='Root directory for datasets')
data_group.add_argument('--train_dataset', type=str, default='GOPRO_Large', help='Dataset for training')
data_group.add_argument('--validation_dataset', type=str, default=None, help='Dataset for validation')
data_group.add_argument('--test_dataset', type=str, default='GOPRO_Large', help='Dataset for testing')
data_group.add_argument('--blur_type', type=str, default='blur_gamma', choices=['blur', 'blur_gamma'], help='Type of blur effect')
data_group.add_argument('--value_range', type=int, default=255, help='Range of pixel values (usually 0-255)')

# Model specifications
model_group = parser.add_argument_group('Model Configurations')
model_group.add_argument('--architecture', type=str, default='MSResNet', help='Model architecture to use')
model_group.add_argument('--pretrained_path', type=str, default='', help='Path to the pretrained model')

# Parse arguments from command line
args = parser.parse_args()

# Expand user and environment variables in data paths
args.data_dir = os.path.expanduser(args.data_dir)

# Setup logging and model save directory based on current date and time
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if not args.save_dir:
    args.save_dir = os.path.join('../experiments', current_time)
os.makedirs(args.save_dir, exist_ok=True)

# Additional configurations based on runtime conditions
if args.process_rank == 0:
    config_template.apply(args)

# Device configuration and setup
def configure_device(args):
    cudnn.benchmark = True
    if args.use_ddp:
        os.environ['MASTER_ADDR'] = args.master_ip
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group(backend=args.backend, init_method=args.init_method, rank=args.process_rank, world_size=args.total_gpus)
        args.device = torch.device('cuda', args.gpu_id) if args.device == 'gpu' else torch.device('cpu')
    else:
        args.device = torch.device('cuda', args.gpu_id) if args.device == 'gpu' else torch.device('cpu')

    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
        torch.cuda.manual_seed_all(args.seed)

def cleanup_environment():
    if args.use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    configure_device(args)
    # Execute the training, validation, or testing phase

