"""Download official RTMW3D weights and export raw SimCC for WebGPU distillation.

No training runs in Python. Run from the environment described in README-ja.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

REVISION = '759b39c13fea6ba094afc1fa932f51dc1b11cbf9'
OFFICIAL = 'https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/'
MODELS = {
    'x': ('rtmw3d-x_8xb32_cocktail14-384x288.py',
          'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth',
          'b0a0eab7ca501e2c88028fe1cfd78b50c7130e220d39541da7f669e5de25e6ca'),
    'l': ('rtmw3d-l_8xb64_cocktail14-384x288.py',
          'rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth', '794dbc78'),
}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='x')
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('teacher'))
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Official source revision must be {REVISION}; got {revision}')
    config_name, filename, expected = MODELS[args.model]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or args.output / filename
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.part')
        print(f'Downloading official checkpoint: {OFFICIAL}{filename}', flush=True)
        with urllib.request.urlopen(OFFICIAL + filename, timeout=180) as src, temporary.open('wb') as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        temporary.replace(checkpoint)
    checkpoint_sha = digest(checkpoint)
    if not checkpoint_sha.startswith(expected):
        raise RuntimeError('Checkpoint SHA-256 does not match the official weight hash')

    import torch
    import onnx
    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    sys.path.insert(0, str(upstream / 'projects/rtmpose3d'))
    import mmpose.models  # noqa: F401
    import mmpose.codecs  # noqa: F401
    import rtmpose3d  # noqa: F401
    from mmpose.registry import MODELS as REGISTRY

    init_default_scope('mmpose')
    torch.set_num_threads(4)
    config_path = upstream / 'projects/rtmpose3d/configs' / config_name
    config = Config.fromfile(str(config_path))
    config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = revert_sync_batchnorm(REGISTRY.build(config.model)).cpu().eval()
    # Trusted, hash-checked official checkpoint. Never load arbitrary uploaded pickle files.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('module.'): v for k, v in saved.get('state_dict', saved).items()}
    loaded = model.load_state_dict(weights, strict=False)
    missing = [k for k in loaded.missing_keys if not k.endswith('num_batches_tracked') and not k.startswith('data_preprocessor.')]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(f'Weight mismatch: {missing}; {loaded.unexpected_keys}')

    class RawSimCC(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.backbone, self.neck, self.head = source.backbone, source.neck, source.head

        def forward(self, image):
            return self.head(self.neck(self.backbone(image)))

    network = RawSimCC(model).eval()
    out = args.output / f'rtmw3d-{args.model}-raw-fp32.onnx'
    with torch.inference_mode():
        torch.onnx.export(network, torch.zeros(1, 3, 384, 288), str(out),
                          input_names=['input'], output_names=['simcc_x', 'simcc_y', 'simcc_z'],
                          opset_version=17, do_constant_folding=True)
    onnx.checker.check_model(str(out))
    provenance = {'upstream': 'https://github.com/open-mmlab/mmpose', 'revision': revision,
                  'config': str(config_path.relative_to(upstream)), 'checkpointUrl': OFFICIAL + filename,
                  'checkpointSha256': checkpoint_sha, 'onnxSha256': digest(out),
                  'parameters': sum(p.numel() for p in network.parameters()),
                  'input': [1, 3, 384, 288], 'outputs': [[1, 133, 576], [1, 133, 768], [1, 133, 576]],
                  'license': 'Apache-2.0', 'purpose': 'teacher only; student training runs in WebGPU'}
    out.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Teacher ready: {out}', flush=True)


if __name__ == '__main__':
    main()
