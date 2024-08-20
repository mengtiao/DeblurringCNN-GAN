import os
import cv2
import numpy as np
import torch
import yaml
from glob import glob
from typing import Optional
from tqdm import tqdm
from fire import Fire

from aug import get_normalize
from models.networks import get_generator


class ImagePredictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config.yaml', encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)
        self.model = get_generator(model_name or config['model'])
        self.model.load_state_dict(torch.load(weights_path, map_location='cuda')['model'])
        self.model.cuda()
        self.model.eval()  # 对批量规范图层进行评估
        self.normalize_fn = get_normalize()

    def preprocess_image(self, img: np.ndarray, mask: Optional[np.ndarray]):
        img, _ = self.normalize_fn(img, img)
        if mask is None:
            mask = np.ones_like(img, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        pad_height = ((img.shape[0] + 31) // 32) * 32
        pad_width = ((img.shape[1] + 31) // 32) * 32

        img = np.pad(img, ((0, pad_height - img.shape[0]), (0, pad_width - img.shape[1]), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_height - img.shape[0]), (0, pad_width - img.shape[1]), (0, 0)), mode='constant')

        return img, mask

    def predict(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        img, mask = self.preprocess_image(img, mask)
        img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        
        with torch.no_grad():
            outputs = self.model(img_tensor, mask_tensor if mask is not None else None)
        output_img = outputs[0].cpu().numpy().transpose(1, 2, 0)
        output_img = np.clip((output_img + 1) * 127.5, 0, 255).astype(np.uint8)
        return output_img

def process_video(input_path, predictor, output_dir):
    video_cap = cv2.VideoCapture(input_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '_deblurred.mp4')
    video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    for _ in tqdm(range(total_frames), desc=f'Processing {input_path}'):
        success, frame = video_cap.read()
        if not success:
            break
        processed_frame = predictor.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video_out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

def main(image_path: str, mask_path: Optional[str] = None, weights_path: str = 'model.h5', output_dir: str = 'output/'):
    predictor = ImagePredictor(weights_path)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(image_path) and image_path.endswith('.mp4'):
        process_video(image_path, predictor, output_dir)
    else:
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
        prediction = predictor.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    Fire(main)
