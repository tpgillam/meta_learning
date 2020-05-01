## Understanding of some MAML sinusoid details

### Pre-training baseline
1. Look only at the "training" portion of each task's batch.
1. The loss of a single task is the mean-squared-error, taking the mean over this training portion.
1. The overall loss is the mean of the losses across all tasks in the batch.
1. Use the meta optimizer (Adam, lr=0.001), with this overall loss. 70000 batches.
1. For testing the baseline, we fine-tune on a new unseen task. This involves some number of gradient steps on the
    training portion.


## Useful links
* https://arxiv.org/pdf/1703.03400.pdf - MAML paper
* https://arxiv.org/pdf/1909.09157.pdf - MAML feature reuse, introduces ANIL
* https://github.com/cbfinn/maml - Code that accompanies the MAML paper
* https://github.com/dbaranchuk/memory-efficient-maml - Makes a comment about deterministic mode for cudnn.


## General meta-learning references
* https://arxiv.org/pdf/2004.05439.pdf - review paper of meta-learning, April 2020
* https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html - overview of meta-learning

## Other interesting topics
* https://arxiv.org/pdf/1803.02999.pdf - Reptile - first order techniques, compared to MAML and FOMAML (First Order MAML)
* https://arxiv.org/pdf/1801.08930.pdf - MAML <-> hierarchical Bayes
* https://arxiv.org/pdf/2004.14539.pdf - differentiable linear programming, for incorporation in neural nets.
* https://arxiv.org/abs/1810.09502 - MAML++
* https://arxiv.org/pdf/1711.06025.pdf - Relation Nets. Meta-learning, but based on learning a metric / embedding?