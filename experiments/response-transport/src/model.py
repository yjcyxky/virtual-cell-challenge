"""Gene-wise response transfer learned in ordered, training-only context episodes."""
import runtime
import gc
import hashlib
import numpy as np
import torch
from torch import nn
from data import write_json
from research import digest


class Transport(nn.Module):
    def __init__(self, spec):
        super().__init__()
        if spec['head'] != 'mlp' or not spec['ntc_conditioning']:
            raise ValueError('Only the registered complete MLP package is implemented')
        h = spec['hidden']
        self.net = nn.Sequential(nn.Linear(17, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1, bias=False))
        nn.init.zeros_(self.net[-1].weight)
        self.scale = spec['response_scale']

    def forward(self, x):
        origin = x.clone()
        origin[:, 0] = 0
        origin[:, 9:] = 0
        return self.scale*(x[:, 0] + (self.net(x)-self.net(origin)).squeeze(-1))


def features(delta, source_gene, target_gene, source_pert, target_pert,
             source_flag, target_flag, support, on_target, spec):
    """Only source response and the two NTC populations; no destination response."""
    d = delta/spec['response_scale']
    columns = [d, source_gene/spec['ntc_scale'], target_gene/spec['ntc_scale'],
               source_pert/spec['ntc_scale'], target_pert/spec['ntc_scale'],
               source_flag, target_flag, torch.log1p(support)/np.log1p(spec['support_scale']), on_target]
    columns += [d*columns[i] for i in (1, 2, 3, 4, 7, 8, 5, 6)]
    return torch.stack(columns, dim=1)


def context_data(cache, contexts, genes, device):
    index = {g: i for i, g in enumerate(genes)}
    result = {}
    for c in contexts:
        s = dict(np.load(cache/f'{c}-statistics.npz'))
        positions = s['positions']
        baseline = np.zeros(len(genes), dtype=np.float32)
        baseline[positions] = s['mean'][0]
        mask = np.zeros(len(genes), dtype=np.float32); mask[positions] = 1
        target_positions = np.asarray([index.get(t, -1) for t in s['labels'][1:]])
        safe = np.maximum(target_positions, 0)
        result[c] = {
            'labels': s['labels'][1:], 'positions': positions,
            'delta': torch.as_tensor(s['mean'][1:]-s['mean'][0], device=device),
            'baseline': torch.as_tensor(baseline, device=device),
            'mask': torch.as_tensor(mask, device=device),
            'pert': torch.as_tensor(np.where(target_positions >= 0, baseline[safe], 0), device=device),
            'pert_mask': torch.as_tensor(np.where(target_positions >= 0, mask[safe], 0), device=device),
            'target_positions': torch.as_tensor(target_positions, device=device),
            'counts': torch.as_tensor(s['counts'][1:].astype(np.float32), device=device),
        }
    return result


def heldout(target, spec):
    key = f'{spec["validation_seed"]}:{target}'.encode()
    return int(hashlib.sha256(key).hexdigest(), 16) % spec['validation_modulus'] == 0


def episodes(data, spec):
    result = []
    for a, source in data.items():
        for b, destination in data.items():
            if a == b:
                continue
            targets, si, di = np.intersect1d(source['labels'], destination['labels'], return_indices=True)
            genes, sg, dg = np.intersect1d(source['positions'], destination['positions'], return_indices=True)
            validation = np.asarray([heldout(t, spec) for t in targets])
            if not validation.any() or validation.all() or not len(genes):
                raise ValueError('Empty episode split')
            result.append(dict(source=a, destination=b, targets=targets, si=si, di=di,
                               genes=genes, sg=sg, dg=dg, train=np.flatnonzero(~validation), valid=np.flatnonzero(validation)))
    return result


def batch(data, episode, rng, n, mode, spec):
    a, b = data[episode['source']], data[episode['destination']]
    rows = rng.choice(episode[mode], n)
    cols = rng.integers(0, len(episode['genes']), n)
    device = a['delta'].device
    si, di, sg, dg, genes = [torch.as_tensor(x, device=device) for x in
                            (episode['si'][rows], episode['di'][rows], episode['sg'][cols], episode['dg'][cols], episode['genes'][cols])]
    x = features(a['delta'][si, sg], a['baseline'][genes], b['baseline'][genes],
                 a['pert'][si], b['pert'][di], a['pert_mask'][si], b['pert_mask'][di],
                 a['counts'][si], (genes == a['target_positions'][si]).float(), spec)
    return x, b['delta'][di, dg]


def optimizer_for(net, spec):
    return torch.optim.AdamW(net.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])


def save_state(path, net, optimizer, rng, step, config):
    temporary = path.with_suffix('.tmp')
    torch.save({'network': net.state_dict(), 'optimizer': optimizer.state_dict(), 'sampler': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                'step': step, 'config_sha256': digest(config)}, temporary)
    temporary.replace(path)


def restore_state(path, net, optimizer, rng, config):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['config_sha256'] != digest(config):
        raise ValueError('Training checkpoint configuration mismatch')
    net.load_state_dict(state['network']); optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['sampler']; torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']


def fit(output, genes, split, config, run):
    spec = config['transport']
    torch.set_num_threads(config['evaluation']['runtime']['num_threads'])
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    device = spec['device']
    data = context_data(output/'cache', split['training_contexts'], genes, device)
    pairs = episodes(data, spec)
    net = Transport(spec).to(device)
    optimizer = optimizer_for(net, spec)
    rng = np.random.default_rng(config['seed'])
    checkpoint = output/'checkpoints/transport.pt'
    step = restore_state(checkpoint, net, optimizer, rng, config) if checkpoint.exists() else 0
    if step > spec['steps']:
        raise ValueError('Checkpoint exceeds frozen budget')
    validation_rng = np.random.default_rng(spec['validation_seed'])
    validation = [batch(data, pair, validation_rng, spec['validation_examples_per_pair'], 'valid', spec) for pair in pairs]
    with torch.no_grad():
        source_mse = float(torch.stack([torch.mean((x[:, 0]*spec['response_scale']-y)**2) for x, y in validation]).mean())
    print(f'Training {len(pairs)} ordered context episodes; {sum(p.numel() for p in net.parameters())} parameters; {step}/{spec["steps"]} steps', flush=True)
    while step < spec['steps']:
        pair = pairs[rng.integers(len(pairs))]
        x, y = batch(data, pair, rng, spec['batch_examples'], 'train', spec)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((net(x)-y)**2)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), spec['gradient_clip'])
        optimizer.step(); step += 1
        if step % spec['checkpoint_every'] == 0 or step == spec['steps']:
            save_state(checkpoint, net, optimizer, rng, step, config)
            with torch.no_grad():
                validation_mse = float(torch.stack([torch.mean((net(x)-y)**2) for x, y in validation]).mean())
            run.log({'train/step': step, 'train/loss': float(loss.detach()), 'train/validation_mse': validation_mse,
                     'train/validation_source_mse': source_mse})
            print(f'Step {step}: loss={float(loss.detach()):.7f} validation={validation_mse:.7f} source={source_mse:.7f}', flush=True)
    with torch.no_grad():
        validation_mse = float(torch.stack([torch.mean((net(x)-y)**2) for x, y in validation]).mean())
    audit = {'steps_completed': step, 'validation_mse': validation_mse, 'validation_source_mse': source_mse,
             'parameters': sum(p.numel() for p in net.parameters()), 'training_contexts': list(data),
             'episode_counts': [{'source': p['source'], 'destination': p['destination'], 'train_targets': len(p['train']),
                                  'diagnostic_targets': len(p['valid']), 'measured_genes': len(p['genes'])} for p in pairs],
             'checkpoint_selection': 'fixed_last_step', 'heldout_H1_response_used': False}
    write_json(output/'cache/fit.json', audit)
    del optimizer, validation
    gc.collect(); torch.cuda.empty_cache()
    return checkpoint, dict(network=net.eval(), data=data, genes=genes, spec=spec), audit


def predict(model, stats, targets, arm, source=None):
    if arm != 'linear':
        raise ValueError('Only registered candidate arm supported')
    net, spec = model['network'], model['spec']
    device = next(net.parameters()).device
    genes = model['genes']; gene_index = {g: i for i, g in enumerate(genes)}
    baseline = np.zeros(len(genes), dtype=np.float32); baseline[stats['positions']] = stats['mean'][0]
    mask = np.zeros(len(genes), dtype=np.float32); mask[stats['positions']] = 1
    baseline = torch.as_tensor(baseline, device=device); mask = torch.as_tensor(mask, device=device)
    values = np.zeros((len(targets), len(genes)), dtype=np.float64)
    mass = np.zeros_like(values)
    with torch.no_grad():
        for data in model['data'].values():
            lookup = {t: i for i, t in enumerate(data['labels'])}
            positions = data['positions']; col = torch.as_tensor(positions, device=device)
            n = len(col)
            for i, target in enumerate(targets):
                if target not in lookup:
                    continue
                row = lookup[target]; tp = gene_index.get(target, -1)
                x = features(data['delta'][row], data['baseline'][col], baseline[col], data['pert'][row].expand(n),
                             (baseline[tp] if tp >= 0 else baseline.new_tensor(0)).expand(n), data['pert_mask'][row].expand(n),
                             (mask[tp] if tp >= 0 else mask.new_tensor(0)).expand(n), data['counts'][row].expand(n),
                             (col == tp).float(), spec)
                response = net(x).cpu().numpy()
                weight = float(data['counts'][row])/spec['weight_cell_unit']
                values[i, positions] += weight*response; mass[i, positions] += weight
    result = np.divide(values, mass, out=np.zeros_like(values), where=mass > 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite transport prediction')
    return result


def generate(output, model, genes, split, config):
    from evaluation import generate as frozen_generate
    return frozen_generate(output, model, genes, split, config)
