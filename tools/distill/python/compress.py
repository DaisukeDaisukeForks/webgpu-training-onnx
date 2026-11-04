"""Physically remove channels from pretrained RTMW3D; no new random student."""
import argparse
import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torch_pruning as tp
from official import FILENAME, REVISION, SHA256, URL, load_teacher

IDS = list(range(23)) + [95, 96, 99, 103, 107, 108, 111, 116, 117, 120, 124, 128, 129, 132]
FORMAT = 'rtmw3d-pruned37-v1'


class Pruned37(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.backbone, self.neck, self.head = source.backbone, source.neck, source.head

    def forward(self, image):
        mean = image.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
        std = image.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
        raw = self.head(self.neck(self.backbone((image * 255 - mean) / std)))
        return tuple(v[:, IDS] for v in raw)


class Decode37(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, image):
        x, y, z = self.source(image)
        points = torch.stack([x.argmax(-1).float() / 2, y.argmax(-1).float() / 2,
                              (z.argmax(-1).float() / 576 * 2 - 1) * 2.1744869], -1)
        scores = torch.minimum(x.amax(-1), y.amax(-1))
        return points, scores


def export(model, output):
    import onnx
    network = Decode37(copy.deepcopy(model).cpu().eval())
    with torch.no_grad():
        torch.onnx.export(network, torch.zeros(1, 3, 384, 288), str(output / 'pruned37.onnx'),
                          opset_version=17, input_names=['image'], output_names=['keypoints', 'scores'])
    onnx.checker.check_model(str(output / 'pruned37.onnx'))
    (output / 'pruned37.schema.json').write_text(json.dumps({
        'format': FORMAT, 'coco133_ids': IDS, 'input': {'name': 'image', 'shape': [1, 3, 384, 288], 'RGB_range': [0, 1]},
        'keypoints': [1, 37, 3], 'units': ['288-wide crop pixels', '384-high crop pixels', 'pelvis-relative metres'],
        'scores': 'original RTMW3D XY peak score',
    }, indent=2), encoding='utf-8')


def factor_head(head, fraction):
    """Low-rank factors computed from learned weights; preserves every head dimension."""
    records = []
    for name in ('conv_dec', 'final_layer', 'final_layer2'):
        owner = getattr(head, name)
        original = owner.conv
        weight = original.weight.detach().float().flatten(1)
        rank = min(original.out_channels, max(8, math.ceil(original.out_channels * fraction / 8) * 8))
        if rank >= original.out_channels:
            continue
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        spatial = nn.Conv2d(original.in_channels, rank, original.kernel_size,
                            original.stride, original.padding, original.dilation, bias=False)
        mixing = nn.Conv2d(rank, original.out_channels, 1, bias=original.bias is not None)
        with torch.no_grad():
            spatial.weight.copy_(vh[:rank].reshape_as(spatial.weight))
            mixing.weight.copy_((u[:, :rank] * s[:rank]).reshape_as(mixing.weight))
            if original.bias is not None:
                mixing.bias.copy_(original.bias)
        owner.conv = nn.Sequential(spatial, mixing)
        records.append({'layer': name, 'rank': rank, 'original_rank': original.out_channels,
                        'retained_weight_energy': float(s[:rank].square().sum() / s.square().sum())})
    return records


def compress(args):
    if not 0 < args.ratio < .9:
        raise ValueError('--ratio must be between 0 and 0.9')
    if not 0 < args.head_rank <= 1:
        raise ValueError('--head-rank must be between 0 and 1')
    torch.set_num_threads(args.cpu_threads)
    source = load_teacher(args.upstream, args.checkpoint, 'cpu')
    model = Pruned37(source).eval().requires_grad_(True)
    sample = torch.zeros(1, 3, 384, 288)
    before_ops, before_params = tp.utils.count_ops_and_params(model, sample)
    # Keep all 133 head tokens/GAU interactions internally. Slicing the 37 outputs
    # alone does not remove context that the original model learned to use.
    # Head output channels and the 3-channel input are not pruning roots.
    ignored = [m for m in model.head.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    # PixelShuffle's 4:1 channel mapping is not modeled by Torch-Pruning 1.6.1.
    # Preserve this boundary's output width; its input and upstream blocks still shrink.
    ignored.append(model.neck.bottom_up_blocks[-1].final_conv.conv)
    pruner = tp.pruner.MagnitudePruner(model, sample,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=args.ratio, ignored_layers=ignored,
        root_module_types=[nn.Conv2d], round_to=8, global_pruning=False)
    pruner.step()
    factors = factor_head(model.head, args.head_rank) if args.head_rank < 1 else []
    after_ops, after_params = tp.utils.count_ops_and_params(model, sample)
    with torch.no_grad():
        result = model(sample)
        if [list(v.shape) for v in result] != [[1, 37, 576], [1, 37, 768], [1, 37, 576]] or not all(torch.isfinite(v).all() for v in result):
            raise RuntimeError('Pruned model produced invalid output')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'format': FORMAT, 'method': 'structured channel pruning of original learned tensors',
              'random_student_initialization': False, 'teacher_url': URL, 'teacher_sha256': SHA256,
              'source_revision': REVISION, 'channel_pruning_ratio': args.ratio,
              'head_factorization': factors,
              'before_parameters': before_params, 'after_parameters': after_params,
              'before_macs': before_ops, 'after_macs': after_ops,
              'accuracy_measured': False, 'gpu_latency_measured': False}
    torch.save({'format': FORMAT, 'network': model.cpu(), 'compression': report}, output / 'compressed.pt')
    (output / 'compression.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    export(model, output)
    print(json.dumps(report), flush=True)
    print('Saved learned, pruned model: ' + str(output / 'compressed.pt'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', default='teacher-upstream')
    p.add_argument('--checkpoint', default='teacher/' + FILENAME)
    p.add_argument('--output', default='pruned37-output')
    p.add_argument('--ratio', type=float, default=.5)
    p.add_argument('--head-rank', type=float, default=1., help='Optional SVD rank fraction for learned head weights, e.g. 0.25')
    p.add_argument('--cpu-threads', type=int, default=4)
    # Persist the importable class name, not __main__.Pruned37.
    from compress import compress as run_compress
    run_compress(p.parse_args())
