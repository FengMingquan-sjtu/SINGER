my implementation of paper [1] N. Gaby and X. Ye, “Approximation of Solution Operators for High-dimensional PDEs.” arXiv, Jan. 18, 2024. Accessed: Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2401.10385


### requirements
torch

torchdiffeq

torch_geometric

matplotlib

nvidia-ml-py3

### Usage:
`python3 NO.py` generate test dataset (about 12 min)

`python3 heat.py` train NODE (about 5 min)

`python3 NO3.py` distill NO (about 2.5 hours)

### Devices:
AMD Ryzen Threadripper 3970X + NVIDIA GeForce RTX 3090


### CMD
CUDA_VISIBLE_DEVICES=1,2 setsid python3 heat.py
pgrep -f heat.py | xargs kill -9
pgrep -f hjb.py | xargs kill -9



CUDA_VISIBLE_DEVICES=1 python main.py --config_path=configs/pricing_default_risk_d10.json --exp_name=pricing_default_risk_d10 --n=1