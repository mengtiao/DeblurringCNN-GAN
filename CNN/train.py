import os
from tqdm import tqdm
import torch
import data.common
from utils import interact, MultiSaver
import torch.cuda.amp as amp

class TrainingManager:

    def __init__(self, config, model, loss_fn, optimizer, data_loaders):
        print('===> Initializing Training Manager')
        self.config = config
        self.mode = 'train'  # 'val', 'test'
        self.epoch = config.start_epoch
        self.save_path = config.save_dir

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_loaders = data_loaders

        self.device = config.device
        self.dtype = config.dtype
        self.dtype_eval = torch.float32 if config.precision == 'single' else torch.float16

        self.result_dir = config.demo_output_dir if config.demo and config.demo_output_dir else os.path.join(self.save_path, 'result')
        os.makedirs(self.result_dir, exist_ok=True)
        print(f'Results saved in {self.result_dir}')

        self.image_saver = MultiSaver(self.result_dir)

        self.is_worker_node = config.launched and config.rank != 0
        self.scaler = amp.GradScaler(init_scale=config.init_scale, enabled=config.amp)

    def save_state(self, epoch=None):
        current_epoch = self.epoch if epoch is None else epoch
        if current_epoch % self.config.save_every == 0:
            if self.mode == 'train':
                self.model.save(current_epoch)
                self.optimizer.save(current_epoch)
            self.loss_fn.save()

    def load_state(self, epoch=None, pretrained=None):
        target_epoch = self.config.load_epoch if epoch is None else epoch
        self.epoch = target_epoch
        self.model.load(target_epoch, pretrained)
        self.optimizer.load(target_epoch)
        self.loss_fn.load(target_epoch)

    def execute_train(self, epoch):
        self.mode = 'train'
        self.epoch = epoch
        self.model.train()
        self.model.to(dtype=self.dtype)
        self.loss_fn.train()
        self.loss_fn.epoch = epoch

        # 初始化训练循环

        if not self.is_worker_node:
            print(f'[Epoch {epoch} / lr {self.optimizer.get_lr():.2e}]')

        if self.config.distributed:
            self.data_loaders[self.mode].sampler.set_epoch(epoch)

        data_loader = tqdm(self.data_loaders[self.mode], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}') if not self.is_worker_node else self.data_loaders[self.mode]

        torch.set_grad_enabled(True)
        for _, batch in enumerate(data_loader):
            self.optimizer.zero_grad()

            inputs, targets = data.common.to(batch[0], batch[1], device=self.device, dtype=self.dtype)
            with amp.autocast(self.config.amp):
                predictions = self.model(inputs)
                loss = self.loss_fn(predictions, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer.G)
            self.scaler.update()

            if isinstance(data_loader, tqdm):
                data_loader.set_description(self.loss_fn.get_loss_desc())

        self.loss_fn.normalize()
        if isinstance(data_loader, tqdm):
            data_loader.set_description(self.loss_fn.get_loss_desc())
            data_loader.display(pos=-1)

        self.loss_fn.step()
        self.optimizer.schedule(self.loss_fn.get_last_loss())

        if self.config.rank == 0:
            self.save_state(epoch)

    def execute_evaluation(self, epoch, mode='val'):
        self.mode = mode
        self.epoch = epoch
        self.model.eval()
        self.model.to(dtype=self.dtype_eval)

        self.loss_fn.validate() if mode == 'val' else self.loss_fn.test()
        self.loss_fn.epoch = epoch
        self.image_saver.join_background()

        #禁用梯度计算

        data_loader = tqdm(self.data_loaders[self.mode], ncols=80, smoothing=0, bar_format='{desc}|{bar}{r_bar}') if not self.is_worker_node else self.data_loaders[self.mode]

        compute_loss = True
        torch.set_grad_enabled(False)
        for _, batch in enumerate(data_loader):
            inputs, targets = data.common.to(batch[0], batch[1], device=self.device, dtype=self.dtype_eval)
            with amp.autocast(self.config.amp):
                predictions = self.model(inputs)

            if mode == 'demo':
                pad_width = batch[2]
                predictions[0], _ = data.common.pad(predictions[0], pad_width=pad_width, negative=True)

            if isinstance(targets, torch.BoolTensor):
                compute_loss = False

            if compute_loss:
                self.loss_fn(predictions, targets)
                if isinstance(data_loader, tqdm):
                    data_loader.set_description(self.loss_fn.get_loss_desc())

            if self.config.save_results != 'none':
                result = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
                file_names = batch[-1]

                if self.config.save_results == 'part' and compute_loss:
                    indices = batch[-2]
                    save_ids = [idx for idx, i in enumerate(indices) if i % 10 == 0]
                    result = result[save_ids]
                    file_names = [file_names[i] for i in save_ids]

                self.image_saver.save_image(result, file_names)

        if compute_loss:
            self.loss_fn.normalize()
            if isinstance(data_loader, tqdm):
                data_loader.set_description(self.loss_fn.get_loss_desc())
                data_loader.display(pos=-1)

            self.loss_fn.step()
            if self.config.rank == 0:
                self.save_state()

        self.image_saver.end_background()

    def validate(self, epoch):
        self.execute_evaluation(epoch, 'val')

    def test(self, epoch):
        self.execute_evaluation(epoch, 'test')

    def fill_evaluation_gaps(self, epoch, mode=None, force=False):
        if epoch <= 0:
            return

        if mode:
            self.mode = mode

        evaluation_needed = force or not all(epoch in self.loss_fn.metric_stat[self.mode][metric] for metric in self.loss_fn.metric)
        if evaluation_needed:
            try:
                self.load_state(epoch)
                self.execute_evaluation(epoch, self.mode)
            except:
                pass
