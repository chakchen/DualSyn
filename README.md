# DualSyn: A Dual-Level Feature Interaction Method to Predict Synergistic Drug Combinations

## Introduction

Drug combination therapy can reduce drug resistance and improve treatment efficacy, making it an increasingly promising cancer treatment method. Although existing computational methods have achieved significant success, predictions on unseen data remain a challenge. There are complex associations between drug pairs and cell lines, and existing models cannot capture more general feature interaction patterns among them, which hinders the ability of models to generalize from seen samples to unseen samples. To address this problem, we propose a dual-level feature interaction model called DualSyn to efficiently predict the synergy of drug combination therapy. This model first achieves interaction at the drug pair level through the drugs feature extraction module. We also designed two modules to further deepen the interaction at the drug pair and cell line level from two different perspectives. The high-order relation module is used to capture the high-order relationships among the three features, and the global information module focuses on preserving global information details. DualSyn not only improves the AUC by 2.15\% compared with the state-of-the-art methods in the transductive task of the benchmark dataset, but also surpasses them in all four tasks under the inductive setting. Overall, DualSyn shows great potential in predicting and explaining drug synergistic therapies, providing a powerful new tool for future clinical applications.

![DualSyn](https://github.com/chakchen/DualSyn/blob/main/image/DualSyn.jpg)

# Installation

You can create a virtual environment using conda

```
conda create -n ddi python=3.7
source activate ddi
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch-geometric==2.0.3
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c rdkit rdkit
```

## Usage

Train the model for transductive task

```
python train_transductive.py
```

Train the model for inductive task (Include leave-out setting and independent dataset setting)

```
bash inductive.sh
```
