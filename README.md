# NAR4TSP

<h1 align="center"> Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems </h1>

The PyTorch Implementation of *Arxiv-2308.00560 -- "Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems"*[pdf](https://arxiv.org/abs/2308.00560).

<p align="center"><img src="./imgs/main.png" width=95%></p>


This paper propose the first non-autogressive model trained using reinforcement learning for solving TSPs.


### How to Run

```shell
# 1. Training
python -u train.py

# 2. Testing
python -u val.py
```

### Acknowledgments

* We would like to thank the following repositories, which are baselines of our code:

  * https://github.com/xbresson/TSP_Transformer



### Citation

If you find our paper and code useful, please cite our paper:

```tex
@misc{Xiao2023,
      title={Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems}, 
      author={Yubin Xiao and Di Wang and Boyang Li and Huanhuan Chen and Wei Pang and Xuan Wu and Hao Li and Dong Xu and Yanchun Liang and You Zhou},
      year={2023},
      eprint={2308.00560},
      archivePrefix={arXiv},
}
```
