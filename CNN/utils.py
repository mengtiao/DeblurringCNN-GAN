import readline
import rlcompleter
readline.parse_and_bind("tab: complete")
import code
import pdb

import time
import argparse
import os
import imageio
import torch
import torch.multiprocessing as mp

# Debugging tools
def interact(local=None):
    """Interactive console with autocomplete function."""
    if local is None:
        local = dict(globals(), **locals())

    readline.set_completer(rlcompleter.Completer(local).complete)
    code.interact(local=local)

def set_trace(local=None):
    """Debugging with pdb"""
    if local is None:
        local = dict(globals(), **locals())

    pdb.Pdb.complete = rlcompleter.Completer(local).complete
    pdb.set_trace()

# Timer
class Timer:
    """Brought from EDSR-PyTorch"""
    def __init__(self):
        self.acc = 0
        self.start()

    def start(self):
        self.t0 = time.time()

    def stop(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.stop()

    def release(self):
        duration = self.acc
        self.acc = 0
        return duration

    def reset(self):
        self.acc = 0

# Argument parser type casting functions
def str2bool(val):
    """Enable default constant true arguments"""
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', 'yes'):
        return True
    if val.lower() in ('false', 'no'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected')

def int2str(val):
    """Convert int to str for environment variable related arguments"""
    return str(val) if isinstance(val, int) else val

# 使用多进程队列的图像保存器

class MultiSaver:
    def __init__(self, output_dir=None):
        self.queue = None
        self.processes = None
        self.output_dir = output_dir

    def start_background(self):
        self.queue = mp.Queue()

        def worker(queue):
            while True:
                if queue.empty():
                    continue
                img, name = queue.get()
                if name:
                    try:
                        if not name.endswith('.png'):
                            name += '.png'
                        imageio.imwrite(name, img)
                    except Exception as e:
                        print(e)
                else:
                    return

        cpu_count = min(8, mp.cpu_count() - 1)
        self.processes = [mp.Process(target=worker, args=(self.queue,), daemon=False) for _ in range(cpu_count)]
        for proc in self.processes:
            proc.start()

    def stop_background(self):
        if self.queue:
            for _ in self.processes:
                self.queue.put((None, None))

    def wait_background(self):
        if self.queue:
            while not self.queue.empty():
                time.sleep(0.5)
            for proc in self.processes:
                proc.join()
            self.queue = None

    def save_image(self, img_batch, names, output_dir=None):
        output_dir = output_dir if output_dir else self.output_dir
        if not output_dir:
            raise Exception('No result directory specified')

        if not self.queue:
            try:
                self.start_background()
            except Exception as e:
                print(e)
                return

        if img_batch.ndim == 2:
            img_batch = img_batch.unsqueeze(0).unsqueeze(0)
        elif img_batch.ndim == 3:
            img_batch = img_batch.unsqueeze(0)

        for img, name in zip(img_batch, names):
            img = img.add(0.5).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
            save_path = os.path.join(output_dir, name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.queue.put((img, save_path))

class Map(dict):
    """Support dot notation access to dictionary attributes"""
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                self.update(arg)
        if kwargs:
            self.update(kwargs)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def to_dict(self):
        return dict(self)
