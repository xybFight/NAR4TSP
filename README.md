<h1 align="center"> Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems </h1>

The PyTorch Implementation of *Arxiv-2308.00560 -- "Reinforcement Learning-based Non-Autoregressive Solver for Traveling Salesman Problems"*[pdf](https://arxiv.org/abs/2308.00560).

This manuscript has been accepted by TNNLS in Oct 2024, and the lastest version has been updated in Arxiv.

<p align="center"><img src="./main.jpg" width=95%></p>


This paper propose the first non-autogressive model trained using reinforcement learning for solving TSPs.


### How to Run

```shell
# 1. Training
python -u train.py

# 2. Testing
python -u val.py
```

### Acknowledgments

* We would like to thank the following repository, which is the baseline of our code:

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
