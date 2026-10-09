"""Export trusted VQ-VAE pickles as CPU state dictionaries, keeping the originals.

Run from the repository root: python -m attribute_disentanglement.vqvae.export_state_dicts
"""

import argparse
import io
import pickle
from pathlib import Path

import torch


class CPUUnpickler(pickle.Unpickler):
    """Load tensor storage on CPU, including CUDA optimizer state in old dumps."""

    def find_class(self, module, name):
        if (module, name) == ('torch.storage', '_load_from_bytes'):
            return lambda data: torch.load(
                io.BytesIO(data), map_location='cpu', weights_only=False,
            )
        if module == 'vqvae' or module.startswith('vqvae.'):
            module = f'attribute_disentanglement.{module}'
        return super().find_class(module, name)


def export_state_dict(source, destination):
    with source.open('rb') as f:
        checkpoint = CPUUnpickler(f).load()
    state = checkpoint['net'].cpu().state_dict()

    # Never overwrite an existing checkpoint. A matching export can be reused.
    if not destination.exists():
        with destination.open('xb') as f:
            torch.save(state, f)
    restored = torch.load(destination, map_location='cpu', weights_only=True)
    if state.keys() != restored.keys():
        raise ValueError(f'State dictionary keys differ: {destination}')
    for name, tensor in state.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)
    print(f'Verified {source} -> {destination} ({len(state)} tensors)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='*', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='destination for explicitly supplied checkpoints')
    args = parser.parse_args()
    if args.output_dir is not None and not args.checkpoints:
        parser.error('--output-dir requires explicit checkpoint paths')
    sources = args.checkpoints or sorted(Path('ckpts/vqvae').rglob('*.pkl'))
    if not sources:
        parser.error('no VQ-VAE pickles found')
    for source in sources:
        destination = source.with_suffix('.pth')
        if args.output_dir is not None:
            destination = args.output_dir / destination.name
        export_state_dict(source, destination)


if __name__ == '__main__':
    main()
