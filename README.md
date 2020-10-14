# Losses
Just a personal losses zoo.

[Boundary loss](https://github.com/LIVIAETS/surface-loss) for highly unbalanced segmentation, based on class boundaries, from [Kervadec et al. 2019](https://arxiv.org/abs/1812.07032). 

Dice losses:
[Generalized Dice Loss by Sudre et al.](https://arxiv.org/abs/1707.03237) in [Kervadec's implementation](https://github.com/LIVIAETS/surface-loss/blob/master/losses.py)
A basic Dice loss implementation

Cross entropy losses:
A CCE loss with added weight on the boundaries, implemented based on [DeepNAT](https://arxiv.org/pdf/1702.08192.pdf), with its binary cross entropy brother.
