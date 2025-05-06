"""
Code base from: https://github.com/boschresearch/one-shot-synthesis
"""

from .AugmentPipe_kornia import AugmentPipe_kornia


class augment_pipe():
    def __init__(self, opt):
        self.augment_func = AugmentPipe_kornia(opt.prob_augm, opt.no_masks).to(opt.device)

    def __call__(self, batch, real=True):
        return self.augment_func(batch)


