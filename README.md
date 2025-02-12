# Spectral SSM

<div align="center">
  <img src="docs/stu.webp" alt="A blue Dragon in a twister" width="480">
</div>

---

<br />

Base research repository for the [Hazan Lab @ Princeton](https://sites.google.com/view/gbrainprinceton/projects/spectral-transformers?authuser=0) for experimenting with the [<em>Spectral State Space Model</em>](https://arxiv.org/abs/2312.06837) on linear dynamical systems.

This is a research repository, not a polished library. Expect to see magic numbers, hard-coded paths, etc.

## Setup

> Note: Please use [uv](https://docs.astral.sh/uv/getting-started/installation/). You'll have more energy. Your skin will be clearer. Your eye sight will improve.

### 1. Virtual environment (optional):

Create a virtual environment with one of the following options:

Conda:
```zsh
envsubst < environment_template.yml > environment.yml
conda env create -f environment.yml
```

uv:
```zsh
uv venv --prompt stu .venv
```

Python/pip:
```zsh
python3 -m venv --prompt stu .venv
```

### Small Remark Regarding Conda

If your HPC environment uses Conda, you can initialize your virtual environment with Conda and use uv afterwards.

If you already have an existing Conda environment, you can run

```zsh
conda env update --name your_existing_env_name --file environment.yml --prune
```

Note that `--prune` will remove packages that aren’t listed in the yml file, helping to bring the environment into exact alignment with what’s specified. You can remove this flag if you'd like.

For a fully reproducible environment, you might prefer creating one from an explicit spec file via:

```zsh
conda create --name new_env_name --file spec-file.txt
```

---


### 2. Installing packages:

> Note: If you want to use [Flash FFT](https://github.com/HazyResearch/flash-fft-conv) and/or [Flash Attention](https://github.com/Dao-AILab/flash-attention), you will need to have a CUDA-enabled device. Please see their repositories for further instructions on installation.

Install the required packages with:

uv:
```python3
uv pip install -e .
```

Python/pip:
```python3
pip install -e .
```

---

To install FlashFFTConv, you can run the following command:
```zsh
module load gcc-toolset/13  # This is Della-specific; make sure you have a valid C/C++ compiler
pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
pip install git+https://github.com/HazyResearch/flash-fft-conv.git
```

## Training

First, make sure you `cd` into the `spectral_ssm` folder.

To train the STU model, run

```zsh
python train_stu.py
```

To train the Transformer model, run

```zsh
python train_transformer.py
```

You can adjust the training configurations for the models in their respective `config.json` files.

## Acknowledgments

Some of the utility functions are adapted from Daniel Suo's [JAX implementation of STU](https://github.com/google-deepmind/spectral_ssm).

Special thanks to (in no particular order):
- Naman Agarwal, Elad Hazan, and the authors of the [Spectral State Space Models](https://arxiv.org/abs/2312.06837) paper
- Yagiz Devre, Evan Dogariu, Isabel Liu, Windsor Nguyen


## Contributions

We welcome contributors to:

- Submit pull requests
- Report issues
- Help improve the project overall

## License

This free open-source software is MIT-licensed. See the [LICENSE](LICENSE) file for more details.

## Citation
If you use this repository or find our work valuable, please consider citing it:

```
@article{spectralssm,
      title={Spectral State Space Models}, 
      author={Naman Agarwal and Daniel Suo and Xinyi Chen and Elad Hazan},
      year={2024},
      eprint={2312.06837},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2312.06837}, 
}
```
```
@misc{flashstu,
      title={Flash STU: Fast Spectral Transform Units}, 
      author={Y. Isabel Liu and Windsor Nguyen and Yagiz Devre and Evan Dogariu and Anirudha Majumdar and Elad Hazan},
      year={2024},
      eprint={2409.10489},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.10489}, 
}
```
