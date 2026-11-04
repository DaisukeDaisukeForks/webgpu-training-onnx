"""Load the official MMPose model and original checkpoint, without an ONNX teacher."""
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request
import runpy

REVISION = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
FILENAME = 'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth'
URL = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/' + FILENAME
SHA256 = 'b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.part')
        print('Downloading official RTMW3D-X: ' + URL, flush=True)
        with urllib.request.urlopen(URL, timeout=180) as source, temporary.open('wb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if digest(temporary) != SHA256:
            raise RuntimeError('Official checkpoint SHA-256 mismatch: ' + str(temporary))
        temporary.replace(path)
    if digest(path) != SHA256:
        raise RuntimeError('This is not the pinned official RTMW3D-X checkpoint')
    return path


def load_teacher(upstream, weight_path, device):
    import torch
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    upstream = Path(upstream).resolve()
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise RuntimeError('MMPose source revision mismatch: ' + actual)
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / 'projects/rtmpose3d'))
    import mmpose.models  # noqa: F401
    import mmpose.codecs  # noqa: F401
    import rtmpose3d  # noqa: F401
    from mmpose.registry import MODELS
    init_default_scope('mmpose')
    # This official config defines its complete model locally. Its _base_ only
    # supplies runner/logging defaults, which a direct inference load does not use.
    cfg = runpy.run_path(str(upstream / 'projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py'))['model']
    cfg['backbone']['init_cfg'] = None
    model = revert_sync_batchnorm(MODELS.build(cfg)).cpu().eval()
    # Only the hash-verified original distribution is unpickled here.
    saved = torch.load(checkpoint(weight_path), map_location='cpu', weights_only=False)
    weights = {k.removeprefix('module.'): v for k, v in saved['state_dict'].items()}
    loaded = model.load_state_dict(weights, strict=False)
    missing = [k for k in loaded.missing_keys if not k.startswith('data_preprocessor.') and not k.endswith('num_batches_tracked')]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(f'Official weight mismatch: {missing}; {loaded.unexpected_keys}')
    model.requires_grad_(False)
    model.to(device)
    print(f'Official RTMW3D-X loaded: {len(weights)} tensors, SHA256={SHA256}', flush=True)
    return model


def predict_teacher(model, images):
    import torch
    import torch.nn.functional as F
    x = F.interpolate(images, (384, 288), mode='bilinear', align_corners=False) * 255
    mean = x.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = x.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    with torch.no_grad():
        return model.head(model.neck(model.backbone((x - mean) / std)))
