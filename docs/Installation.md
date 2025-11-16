# Installation

These installation instructions are very preliminary, and surely will not work on all systems.
But if any problems come up, do not hesitate to contact us (ideally via Github issue).

## Step 1: Create a virtual environment
Before running any scripts, you should create a virtual Python environment.
In these instructions, we use Miniconda for managing your virtual environments,
but any alternative like Conda, Mamba, virtualenv, venv, ... should be fine.

If you don't have any, you could install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

Now you can create a new virtual environment using
```shell
conda create --name <env_name> python=3.9
```

In order to use it, you need to activate the environment:
```shell
conda activate <env_name>
```

## Step 2: Installation via PyPI (Option 1, recommended)


## Step 2 alternative: Installation via cloning our repository (Option 2, recommended)

Make sure to have git installed, then run
```shell
git clone https://github.com/CellArchLab/membrain-pick.git
```

Move to the folder "membrain-pick" (from the cloned repository above) that contains the "src" folder.
Here, run

```shell
cd membrain-pick
install .
```

This will install MemBrain-seg and all dependencies required for segmenting your tomograms.

## Step 3: Validate installation
As a first check whether the installation was successful, you can run
```shell
membrain_pick
```
This should display the different options you can choose from MemBrain, like "segment" and "train", similar to the screenshot below:


## Step 4: Install surforama
In order to visualize the results of MemBrain-pick, you can use surforama.
However, the current official version of surforama has some compatibility issues with the current version of MemBrain-pick. Therefore, it's best to install it via

1. close repo https://github.com/LorenzLamm/surforama.git
2. switch to branch "star_file_loading_adjustments"
3. install via `pip install .`

```shell
git clone https://github.com/LorenzLamm/surforama.git
cd surforama
git checkout star_file_loading_adjustments
pip install .
pip install napari
```

We are working on a more stable version of surforama that will be compatible with the current version of MemBrain-pick.
