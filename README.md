## Understanding of some MAML sinusoid details

### Pre-training baseline
1. Look only at the "training" portion of each task's batch.
1. The loss of a single task is the mean-squared-error, taking the mean over this training portion.
1. The overall loss is the mean of the losses across all tasks in the batch.
1. Use the meta optimizer (Adam, lr=0.001), with this overall loss. 70000 batches.

## Useful links

* https://arxiv.org/pdf/1703.03400.pdf - MAML paper
* https://arxiv.org/pdf/1909.09157.pdf - MAML feature reuse, introduces ANIL
* https://github.com/cbfinn/maml - Code that accompanies the MAML paper
* https://github.com/dbaranchuk/memory-efficient-maml - Makes a comment about deterministic mode for cudnn.