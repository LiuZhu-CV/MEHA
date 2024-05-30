## Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-Loop and Hessian-Free Solution Strategy (MEHA)
This repo contains code accompaning the paper, Moreau Envelope for Nonconvex Bi-Level Optimization:  
	A Single-Loop and Hessian-Free Solution Strategy (Liu et al., ICML 2024). 


### Abstract
This work focuses on addressing two major challenges in the context of large-scale nonconvex Bi-Level Optimization (BLO) problems, which are increasingly applied in machine learning due to their ability to model nested structures. These challenges involve ensuring computational efficiency and providing theoretical guarantees. While recent advances in scalable BLO algorithms have primarily relied on lower-level convexity simplification, our work specifically tackles large-scale BLO problems involving nonconvexity in both the upper and lower levels. We simultaneously address computational and theoretical challenges by introducing an innovative single-loop gradient-based algorithm, utilizing the Moreau envelope-based reformulation, and providing non-asymptotic convergence analysis for general nonconvex BLO problems. Notably, our algorithm relies solely on first-order gradient information, enhancing its practicality and efficiency, especially for large-scale BLO learning tasks. We validate our approach's effectiveness through experiments on various synthetic problems, two typical hyper-parameter learning tasks, and a real-world neural architecture search application, collectively demonstrating its superior performance.
### Dependencies
This code mainly requires the following:
- Python 3.*
- Pytorch


We also provide an implementation using [Jittor](https://github.com/Jittor/jittor) framework.

### Usage

You can run the python file to test the performance of different methods following the script below:

Non-convex case:
```
cd non_convex
Python non_convex_1dim.py 
```

Non-smooth case:
```
cd non_smooth_case1
Python non_smooth_firstorder_1000dim.py 
```

### Jittor Version
We also provide a version with Jittor framework
```
cd Jittor
Python MEHA.py 
```

### Citation

If this work has been helpful to you, please feel free to cite our paper!
- Risheng Liu, Zhu Liu, Wei Yao, Shangzhi Zeng, Jin Zhang. ["Moreau Envelope for Nonconvex Bi-Level Optimization: Single-Loop and Hessian-Free Solution Strategy"](https://arxiv.org/abs/2405.09927). ICML, 2024.

If you have any other questions about the code, please email [Zhu Liu](mailto:liuzhu_ssdut@foxmail.com).

### License 

MIT License

Copyright (c) 2024 Vision & Optimization Group (VOG) 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
