## SINGER: Stochastic Network Graph Evolving Operator for High Dimensional PDEs
Official implementation of ICLR 2025 paper "SINGER: Stochastic Network Graph Evolving Operator for High Dimensional PDEs". Available: https://openreview.net/forum?id=wVADj7yKee


### Requirements:
```
torch
torchdiffeq
torch_geometric
matplotlib
nvidia-ml-py3
```

### Usage:
**Code**:
`models.py` The implementation of the networks
`loss_func.py` Implements the loss functions based on different equation types and network types.
`singer.py` The core code of the project enables experiments with the Heat equation and Hamilton-Jacobi-Bellman (HJB) equations, with available network architectures including node (Node), pino (PINO), and singer (SINGER). 
The code supports data generation, training, and testing operations. The command-line parameters include:
* `--eqn_type`: Equation type selection ('heat' or 'hjb');
* `--cur_dim`: Dimensionality of the current equation (default: 10);
* `--v_type`: Neural solver selection ('node', 'pino', or 'singer');
* `--drop_out`: Dropout activation switch;
* `--train_mode`: Operational mode ('0' for test dataset generation, '1' for network training and testing with existing dataset).

**Example**: 10D Heat Equation Experiment
To run experimental tests for a 10-dimensional Heat equation, use the following commands:

```cmd
# Phase 1: Generate test data
python singer.py --eqn_type heat --cur_dim 10 --train_mode 0

# Phase 2: Train/test with Our solver (SINGER)
python singer.py --eqn_type heat --cur_dim 10 --v_type singer --drop_out 1 --train_mode 1

```

and run two baselines by:
```cmd
# Baseline 1: Train/test with Node solver (NODE)
python singer.py --eqn_type heat --cur_dim 10 --v_type node --drop_out 1 --train_mode 1

# Baseline 2: Train/test with PINO solver (PINO)
python singer.py --eqn_type heat --cur_dim 10 --v_type pino --train_mode 1

```
**Configs**:
Specific parameters for experiments are defined in corresponding JSON files within the `configs/` directory, following the naming convention:
`{eqn_type}_{v_type}_d{cur_dim}.json`
(e.g., `heat_node_d10.json` for Node-based 10D Heat equation configurations)

**Checkpoints**:
Experimentally generated test data and trained model parameters are stored in the `checkpoints/` directory.

**Logging**:
Experiment logs will be saved in the `.log/` directory.

### Devices:
AMD Ryzen Threadripper 3970X + NVIDIA GeForce RTX 3090
