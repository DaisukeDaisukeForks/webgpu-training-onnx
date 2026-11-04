"""Optional accuracy recovery for the already pretrained, pruned RTMW3D model."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
from data import prepare, read_image
from official import FILENAME, REVISION, SHA256, URL, digest, load_teacher, predict_teacher
from compress import IDS, export
from loss import distillation_loss, teacher_targets

FORMAT = 'rtmw3d-pruned37-training-v1'
stopping = False


def request_stop(*_):
    global stopping
    stopping = True
    print('Stopping after the current update; saving complete state.', flush=True)


def atomic_save(value, path):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def save(path, model, optimizer, state, config, source_hash, backup=False):
    payload = {'format': FORMAT, 'network': copy.deepcopy(model).cpu(),
               'optimizer': optimizer.state_dict(), 'state': state, 'config': config,
               'source_hash': source_hash, 'teacher_sha256': SHA256,
               'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
               'rng_torch': torch.get_rng_state(),
               'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    atomic_save(payload, path / 'latest.pt')
    if backup:
        atomic_save(payload, path / time.strftime('backup-%Y%m%d-%H%M%S.pt'))
    report = {'format': FORMAT, 'teacher': {'checkpoint_url': URL, 'sha256': SHA256, 'source_revision': REVISION},
              'config': config, 'epoch': state['epoch'], 'updates': state['updates'],
              'training_hours': round(state['seconds'] / 3600, 1), 'history': state['history'],
              'loss': state.get('loss'), 'loss_source': 'frozen original RTMW3D-X outputs only',
              'accuracy_measured': False, 'joints': IDS}
    (path / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


def train(args):
    device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the cu118 PyTorch wheel and NVIDIA driver, or explicitly use --device cpu.')
    device = torch.device(device_name)
    torch.set_num_threads(args.cpu_threads)
    path = Path(args.output).resolve()
    path.mkdir(parents=True, exist_ok=True)
    source_hash = digest(args.manifest)
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['rows']
    if not rows:
        raise RuntimeError('The image manifest is empty')
    # Load official modules before restoring a structurally changed module tree.
    teacher = load_teacher(args.upstream, args.checkpoint, device)
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resume and (resume.get('format') != FORMAT or resume['source_hash'] != source_hash or resume['teacher_sha256'] != SHA256):
        raise RuntimeError('Checkpoint format, input manifest, or teacher mismatch')
    config = dict(resume['config']) if resume else {'epochs': 100, 'batch': 4, 'lr': .0003, 'seed': 20261104}
    for key in config:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config['epochs'] < 1 or config['batch'] < 1 or not math.isfinite(config['lr']) or not 0 < config['lr'] <= .01:
        raise RuntimeError('Invalid epoch, batch size, or learning rate')
    random.seed(config['seed'])
    np.random.seed(config['seed'] % 2**32)
    torch.manual_seed(config['seed'])
    initial = resume or torch.load(args.model, map_location='cpu', weights_only=False)
    if not resume and initial.get('format') != 'rtmw3d-pruned37-v1':
        raise RuntimeError('Use compressed.pt created from official weights by compress.py')
    model = initial['network'].to(device).train().requires_grad_(True)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], foreach=False)
    state = {'epoch': 0, 'cursor': 0, 'order': [], 'updates': 0, 'seconds': 0., 'history': []}
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        state = resume['state']
        random.setstate(resume['rng_python'])
        np.random.set_state(resume['rng_numpy'])
        torch.set_rng_state(resume['rng_torch'])
        if device_name == 'cuda' and resume['rng_cuda']:
            torch.cuda.set_rng_state_all(resume['rng_cuda'])
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed % 2**32)
            torch.manual_seed(args.seed)
    if state['epoch'] >= config['epochs']:
        export(model, path)
        print('Target epoch already reached. Increase --epochs to continue.')
        return
    print(f'Device={device}; student={sum(p.numel() for p in model.parameters()):,} parameters; inputs={len(rows)}; config={config}', flush=True)
    last_save = last_backup = started = time.monotonic()
    session_steps = 0
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while state['epoch'] < config['epochs'] and not stopping:
            if not state['order']:
                state['order'] = list(range(len(rows)))
                random.shuffle(state['order'])
                state['cursor'] = 0
            while state['cursor'] < len(rows) and not stopping:
                chosen = state['order'][state['cursor']:state['cursor'] + config['batch']]
                images = torch.stack([read_image(rows[i]) for i in chosen]).to(device)
                if random.random() < .5:
                    images = images.flip(-1)
                raw = predict_teacher(teacher, images)
                targets = teacher_targets(raw)
                del raw
                if not bool((targets[1] > 0).any().item()):
                    raise RuntimeError('Teacher detected no reliable joints in this batch. Check the person ROI and images.')
                student_input = images
                student_input = (student_input * random.uniform(.85, 1.15)).clamp(0, 1)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = config['lr'] * min(1., (state['updates'] + 1) / 100)
                loss = distillation_loss(model(student_input), targets)
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise RuntimeError('Non-finite loss; update was not applied')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
                optimizer.step()
                state['cursor'] += len(chosen)
                state['updates'] += 1
                state['loss'] = value
                session_steps += 1
                now = time.monotonic()
                state['seconds'] += now - started
                started = now
                print(f"epoch {state['epoch'] + 1}/{config['epochs']} | {state['cursor']}/{len(rows)} | step {state['updates']} | loss {value:.6f} | {state['seconds'] / 3600:.1f} h", flush=True)
                if now - last_save >= 60:
                    backup = now - last_backup >= 1800
                    save(path, model, optimizer, state, config, source_hash, backup)
                    last_save = now
                    if backup:
                        last_backup = now
                if args.max_steps and session_steps >= args.max_steps:
                    break
            if state['cursor'] == len(rows):
                state['epoch'] += 1
                state['cursor'] = 0
                state['order'] = []
                state['history'].append({'epoch': state['epoch'], 'loss': state['loss'], 'training_hours': round(state['seconds'] / 3600, 1)})
                save(path, model, optimizer, state, config, source_hash)
            if args.max_steps and session_steps >= args.max_steps:
                break
    finally:
        state['seconds'] += time.monotonic() - started
        save(path, model, optimizer, state, config, source_hash)
    export(model, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Use images/videos as inputs; no keypoint annotation files')
    prep.add_argument('--input', nargs='+', required=True)
    prep.add_argument('--output', default='distill-data')
    prep.add_argument('--fps', type=float, default=4)
    prep.add_argument('--roi', default='0,0,1,1', help='Normalized person box x,y,width,height')
    run = commands.add_parser('train')
    run.add_argument('--manifest', required=True)
    run.add_argument('--model', default='pruned37-output/compressed.pt')
    run.add_argument('--upstream', default='teacher-upstream')
    run.add_argument('--checkpoint', default='teacher/' + FILENAME)
    run.add_argument('--output', default='distill-output')
    run.add_argument('--resume')
    run.add_argument('--device', choices=['cuda', 'cpu'])
    run.add_argument('--epochs', type=int)
    run.add_argument('--batch', type=int)
    run.add_argument('--lr', type=float)
    run.add_argument('--seed', type=int)
    run.add_argument('--cpu-threads', type=int, default=4)
    run.add_argument('--max-steps', type=int, default=0, help='Optional limit for this session; checkpoint remains resumable')
    args = parser.parse_args()
    if args.command == 'prepare':
        roi = [float(v) for v in args.roi.split(',')]
        if not math.isfinite(args.fps) or args.fps <= 0 or len(roi) != 4 or not all(map(math.isfinite, roi)) or roi[2] <= 0 or roi[3] <= 0:
            parser.error('Invalid FPS or ROI')
        prepare(args.input, args.output, args.fps, roi)
    else:
        train(args)


if __name__ == '__main__':
    main()
