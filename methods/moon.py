"""Adapter for the released MOON code, NOT a transcription of Algorithm 1.

The optimizer is byte-identical upstream code. Weight classes are extracted
verbatim. Scores use the *previous* optimizer update including auxiliary Adam
blocks, excluding LR and decoupled weight decay, exactly as upstream trainers.
"""
import torch
from methods.moon_reference_weights import MOON
from methods.moon_muon import SingleDeviceMuonWithAuxAdam

IMPLEMENTATION = 'upstream-37319d1-v1'

class ReferenceMOON(MOON):
    def state_dict(self):
        return dict(implementation=IMPLEMENTATION, w=self.w.detach().cpu(),
                    min_losses=self.min_losses.detach().cpu(), w_opt=self.w_opt.state_dict())

    def load_state_dict(self, state):
        if state.get('implementation') != IMPLEMENTATION:
            raise ValueError('Incompatible MOON checkpoint: restart the old clean-room run in a new output directory.')
        with torch.no_grad():
            self.w.copy_(state['w'].to(self.device))
            self.min_losses.copy_(state['min_losses'].to(self.device))
        self.w_opt.load_state_dict(state['w_opt'])


def reference_parameter_groups(model, dataset, lr):
    muon, adam = [], []
    first_conv = None if dataset == 'multimnist' else model.segnet.encoder_block[0][0].weight
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if dataset == 'multimnist':
            use_muon = name.startswith('transformer_enc') and p.ndim >= 2
        else:
            use_muon = 'pred' not in name and p.ndim >= 2 and p is not first_conv
        (muon if use_muon else adam).append(p)
    if not muon or not adam:
        raise ValueError('Reference MOON requires both Muon hidden matrices and auxiliary Adam parameters.')
    return [dict(params=muon, use_muon=True, lr=lr, weight_decay=0.001),
            dict(params=adam, use_muon=False, lr=lr, betas=(0.9,0.95), weight_decay=0.001)]


def make_reference_optimizer(model, dataset, lr):
    return SingleDeviceMuonWithAuxAdam(reference_parameter_groups(model, dataset, lr))


def previous_update_scores(losses, parameters, optimizer):
    params = [p for p in parameters if p.requires_grad]
    # Match upstream task-gradient and accumulation ordering, including step zero.
    gradients = [torch.autograd.grad(loss, params, retain_graph=True,
                  create_graph=False, allow_unused=True) for loss in losses]
    delta = torch.zeros(len(losses), device=losses.device, dtype=losses.dtype)
    for j, p in enumerate(params):
        update = optimizer.state[p].get('last_update')
        if update is None:
            continue
        for i, grads in enumerate(gradients):
            if grads[j] is not None:
                delta[i] += (grads[j] * update).sum()
    if not torch.isfinite(delta).all():
        raise FloatingPointError('MOON scores are not finite')
    return delta.detach()
