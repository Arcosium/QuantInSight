"""Deterministic CUDA pooling for the frozen 32x65 image geometry."""
import argparse
from pathlib import Path
import torch
from torch import nn

if __package__:
    from quant import timefolio_heatmap_gpu_worker as legacy
else:
    import worker_v1 as legacy

original_reference = legacy.reference


class FixedHeatmapPool(nn.Module):
    def forward(self, x):
        if x.shape[-2:] != (8, 16): raise ValueError('Frozen CNN feature map must be8x16')
        # AdaptiveAvgPool2d((2,4)) partitions8x16 into disjoint4x4 blocks.
        # Fixed AvgPool has the same forward/gradient, without atomic backward.
        return nn.functional.avg_pool2d(x, kernel_size=4, stride=4)


def reference(root):
    ref = original_reference(root); base = ref.AlternativeNet
    class PortableNet(base):
        def __init__(self, architecture):
            super().__init__(architecture)
            if architecture == 'cnn':
                assert isinstance(self.full.net[8], nn.AdaptiveAvgPool2d)
                assert self.full.net[8].output_size == (2, 4)
                self.full.net[8] = FixedHeatmapPool()
    ref.AlternativeNet = PortableNet
    return ref


def run(root, out, seed, device='cuda'):
    legacy.reference = reference
    try: legacy.run(root, out, seed, device)
    finally: legacy.reference = original_reference


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=parser.parse_args();run(a.root,a.out,a.seed,a.device)
