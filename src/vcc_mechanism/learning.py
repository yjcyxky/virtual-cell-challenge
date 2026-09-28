"""Complete stochastic training state shared by the new learned packages."""
import numpy as np
import torch
from research import digest


def initialize(config):
    torch.set_num_threads(config['evaluation']['runtime']['num_threads'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(config['seed'])


def save_state(path, net, optimizer, rng, step, config):
    temporary = path.with_suffix('.tmp')
    torch.save({'network': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'sampler': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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


def optimizer_for(net, spec):
    return torch.optim.AdamW(net.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'],
                             betas=tuple(spec['optimizer_betas']), eps=spec['optimizer_epsilon'])


def response_loss(prediction, truth, spec):
    """Magnitude weighting and direction supervision use source training labels only."""
    scaled = truth/spec['response_scale']
    predicted = prediction/spec['response_scale']
    weight = 1 + spec['magnitude_weight']*torch.tanh(scaled.abs())
    mse = (weight*(predicted-scaled).square()).mean()
    active = truth.abs() >= spec['direction_threshold']
    direction = (torch.nn.functional.softplus(-predicted*truth.sign())*active).sum()/active.sum().clamp_min(1)
    return mse + spec['direction_weight']*direction
