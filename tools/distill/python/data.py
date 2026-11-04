import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VIDEOS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def prepare(inputs, output, fps, roi):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = output / 'frames'
    frames.mkdir(exist_ok=True)
    paths = []
    for value in inputs:
        path = Path(value).resolve()
        paths.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    paths = sorted(set(paths))
    rows = []
    for path in paths:
        ext = path.suffix.lower()
        if ext in IMAGES:
            rows.append({'image': str(path), 'roi': roi})
        elif ext in VIDEOS:
            stamp = f'{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{fps}'
            prefix = hashlib.sha256(stamp.encode()).hexdigest()[:20]
            video = cv2.VideoCapture(str(path))
            rate = video.get(cv2.CAP_PROP_FPS)
            if not video.isOpened() or rate <= 0:
                raise RuntimeError('Cannot read video: ' + str(path))
            index, next_time = 0, 0.0
            try:
                while True:
                    ok, frame = video.read()
                    if not ok:
                        break
                    timestamp = index / rate
                    if timestamp + 1e-8 >= next_time:
                        dest = frames / f'{prefix}-{index:09d}.jpg'
                        if not dest.exists() and not cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                            raise RuntimeError('Cannot save frame: ' + str(dest))
                        rows.append({'image': str(dest), 'roi': roi, 'source': str(path), 'seconds': timestamp})
                        next_time += 1 / fps
                    index += 1
            finally:
                video.release()
        print(f'Prepared {len(rows)} images: {path.name}', flush=True)
    if not rows:
        raise RuntimeError('No supported image or video files')
    manifest = output / 'images.json'
    manifest.write_text(json.dumps({'format': 'distill37-images-v1', 'rows': rows}, ensure_ascii=False), encoding='utf-8')
    print('Manifest: ' + str(manifest), flush=True)


def read_image(row):
    with Image.open(row['image']) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        # ROI is the person box, normalized to the original image; all limbs must fit.
        x, y, w, h = row.get('roi', [0, 0, 1, 1])
        x, y, w, h = x * width, y * height, w * width, h * height
        ew, eh = w * 1.25, h * 1.25
        if ew / eh > .75:
            eh = ew / .75
        else:
            ew = eh * .75
        left, top = x + w / 2 - ew / 2, y + h / 2 - eh / 2
        crop = image.transform((288, 384), Image.Transform.AFFINE,
                               (ew / 288, 0, left, 0, eh / 384, top), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(crop, dtype=np.float32).copy()).permute(2, 0, 1) / 255
