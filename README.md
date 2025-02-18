# Reproducibility Instructions for "Differentially Private Graph Diffusion with Applications in Personalized PageRanks" Experiments
### **Authors：** **Rongzhe Wei**, **Eli Chien**, **Pan Li**

This repository contains the code and instructions to reproduce the experiments presented in our paper on privacy-preserving graph diffusion framework. Please follow the steps below to set up the environment and run the experiments.

## 1. Create and Activate the Virtual Environment:

   Open your terminal or command prompt and run the following commands to create and activate the virtual environment:

   ```bash
   conda create -n private_graph_diffusion python=3.10.4
   conda activate private_graph_diffusion
   ```

## 2. Install the Required Packages:

```
pip install -r requirements.txt
```


## 3. Running the Experiments:
The experiments can be executed by running the main.py script. The script supports various configurations through command-line arguments.

**Usage:**
```bash
python main.py --dataset <DATASET> --method <METHOD> [OPTIONS]
```

**Arguments:**
```bash
`--dataset`: The dataset to use for the experiment. Options are `BlogCatalog`, `Themarker`, `Flickr`.
`--method`: The method to run. Options are `our`, `pushflow`, `edgeflipping`, `all`.
`--epsilon`: The DP privacy budget of graph diffusion (default: 0.1).
`--max_iter`: The propagation iteration for PPR (default: 100).
`--beta`: The teleport probability for PPR (default: 0.8).
```


**Example:**
To run the experiment using the `BlogCatalog` dataset with all methods, use the following command:
```bash
python main.py --dataset BlogCatalog --method all
```

### **Paper Link**
📄 [Differentially Private Graph Diffusion with Applications in Personalized PageRanks](https://arxiv.org/abs/2407.00077)

### **BibTeX Citation**
Cite our paper:

```bibtex
@article{wei2024differentially,
  title={Differentially private graph diffusion with applications in personalized pageranks},
  author={Wei, Rongzhe and Chien, Eli and Li, Pan},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
