# SemiContinuousTOMO

# [Co-Optimization of Structure and Manufacturable Semi-Continuous Layers for Laminated Composites](https:////github.com/RyanTaoLiu/SemiContinuousTOMO/)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https:////github.com/RyanTaoLiu/SemiContinuousTOMO/)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)

[**SIGGRAPH 2026 (ToG)**](https://dl.acm.org/doi/abs/10.1145/3811393)

**Authors:**  
[Tao Liu\*](#), [Aoran Lyu\*](#), Yongxue Chen, Yu Jiang, Petty Michael James,, and [Charlie C.L. Wang†](https://mewangcl.github.io/)

---
## 📖 Overview
To enable the design and manufacturing of optimized composite structures using fabric plies, we propose a field-driven optimization framework that jointly optimizes structural topology and manufacturable layers. A central challenge in this setting is the modeling and optimization of partial fabric layers with near-uniform thickness, which we formulate as a semi-continuous periodic scalar field parameterized by a continuous implicit neural vector field. Within this concurrent structure–layer optimization framework, we further derive a formulation of inter-layer anisotropic mechanical behavior that enables effective modeling of mechanical property transitions induced by partial-layer boundaries, together with additional objectives for manufacturability and field regularization. We validate the effectiveness of our approach through both numerical simulations and physical experiments, demonstrating that the optimized fabric-reinforced laminated composites achieve up to 43.8% higher stiffness compared to counterparts fabricated using planar fabric plies.
![Overview](https://ryantaoliu.github.io/SemiContinuousTOMO/figs/teaser.png "Overview")
Computational pipeline of the proposed framework. Two continuous fields -- a density field and a vector field -- are parameterized by neural networks (NN) and jointly optimized in a self-learning loop via backpropagation. Mechanical performance is evaluated through FEA, while manufacturability is enforced using geometry-based loss terms. The optimized fields are then used to extract a collection of curved semi-continuous layers, which are flattened into planar panel layouts for the fabrication of laminated composites.
![Pipeline](https://ryantaoliu.github.io/SemiContinuousTOMO/figs/process.png "Pipeline")
## 🚀 Installation

### Environment
- **OS:** Ubuntu 20.04 LTS  
- **GPU:** Nvidia RTX 4080 / 4090 (≥16 GB VRAM)  
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13700K  
- **Memory:** 32 GB  


### Git download the code
```bash
git clone https://github.com/RyanTaoLiu/NeuralTOMO
cd NeuralTOMO
mkdir data
```

### Create and activate the Python environment
```
conda create -n NeuralTO python=3.10
conda activate NeuralTO
```

### Install dependencies
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-nvcc=11.8.89 cuda-cudart=11.8.89 cuda-toolkit=11.8.0

conda install -c conda-forge scikit-sparse

pip install comet_ml paramiko matplotlib ninja pyvista meshio
pip install pymkl
pip install diso==0.1.2
```

### 🧪Example of TipCantilever
```
python main.py \
  --problem=TipCantilever_30_20_20_midLoad \
  --material=PLAPlus \
  --no-isotropic \
  --wSR 1 \
  --desireVolumeFraction 0.15 \
  --wHarmonic 1e4 \
  --numLayers 5 \
  --numNeuronsPerLayer 128 \
  --fourierMap \
  --learningRate 0.001
```
Results will be saved in *./data*.

```
python main.py --help
```
Will show the help documents for args.

### 📂Data structure & Notes
- **Boundary condition**, See [*'./settings/problems/testproblem.py'*](https://github.com/RyanTaoLiu/NeuralTOMO/blob/main/settings/problems/testProblem.py). 
Based on the paper, **An efficient 3D topology optimization code written in Matlab**\[1\]. Add any new boundary condition as the new py file follow the *testproblem.py* and can be called by the arg *'--problem'*.

- The result saved every 50 iterations, includes a '*.obj' file for the topology optimization marching cubes result(via diso lib), and '*.vtk’ file for the voxel-based density('density'), fiber direction ('fiber'), and local printing direction('lpd').

## Reference
- [1] K. Liu and A. Tovar. **An efficient 3D topology optimization code written in Matlab**. *Structural and Multidisciplinary Optimization*, **50**(6):1175–1196, 2014. [https://doi.org/10.1007/s00158-014-1107-x](https://doi.org/10.1007/s00158-014-1107-x)  

- [2] T. Liu, T. Zhang, Y. Chen, Y. Huang, and C.C.L. Wang. **Neural Slicer for Multi-Axis 3D Printing**. *ACM Transactions on Graphics*, **43**(4), Article 85, 15 pages, July 2024. [https://doi.org/10.1145/3658212](https://doi.org/10.1145/3658212)  

- [3] T. Liu, T. Zhang, Y. Chen, W. Wang, Y. Jiang, Y. Huang, and C.C.L. Wang. **Neural Co-Optimization of Structural Topology, Manufacturable Layers, and Path Orientations for Fiber-Reinforced Composites**. *ACM Transactions on Graphics*, **44**(4), Article 128, 17 pages, August 2025. [https://doi.org/10.1145/3730922](https://doi.org/10.1145/3730922)  

- [4] T. Liu, A. Lyu, Y. Chen, Y. Jiang, M。 Petty, and C.C.L. Wang. **Co-Optimization of Structure and Manufacturable Semi-Continuous Layers for Laminated Composites**. *ACM Transactions on Graphics*, **44**(4), Article 146, 17 pages, July 2026. [https://dl.acm.org/doi/abs/10.1145/3811393](https://dl.acm.org/doi/abs/10.1145/3811393)  
