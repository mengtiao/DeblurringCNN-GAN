import sys
import subprocess
from argparse import ArgumentParser

from utils import convert_str_to_bool, convert_int_to_str

def parse_launch_arguments():
    parser = ArgumentParser(description="Utility to facilitate the launching of PyTorch distributed training across multiple GPUs.")

    parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs to use for training')

    parser.add_argument("script_path", type=str,
                        help="Path to the single-GPU training script, "
                             "including any arguments that should be passed to it.")

    parser.add_argument('script_args', nargs='*')
    return parser.parse_args()

def launch_processes():
    args = parse_launch_arguments()

    jobs = []
    for gpu_index in range(args.gpu_count):
        command_line = [sys.executable]  # sys.executable points to the Python executable

        command_line.append(args.script_path)
        command_line.extend(args.script_args)

        command_line += ['--distributed', 'True']
        command_line += ['--initialized', 'True']
        command_line += ['--gpu_count', convert_int_to_str(args.gpu_count)]
        command_line += ['--gpu_index', convert_int_to_str(gpu_index)]

        job = subprocess.Popen(command_line)
        jobs.append(job)

    for job in jobs:
        job.wait()
        if job.returncode != 0:
            raise subprocess.CalledProcessError(returncode=job.returncode, cmd=command_line)

if __name__ == "__main__":
    launch_processes()
