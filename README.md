# a Neural Programmer-Interpreters implementation with Tensorflow

paper: https://arxiv.org/abs/1511.06279

This implementation can learn only Addition task.
One major difference from the original is that this model does not compute the termination probability. Instead that have RETURN command as sub-program.

Note: This model cannot learn sufficiently. It seems difficult to calculate numeric addition.

## Requirements
- Python3
- Tensorflow 1.3

## Start
```
$ python npi3.py
```
