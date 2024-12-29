# MemBrain-pick Training

## Introduction
What happens during training is:
1. All meshes are loaded into the RAM and for each mesh a distance map is computed based on the given ground truth particle positions.
2. Meshes are split into small "partitions" (default size: 2000 vertices) to improve training efficiency and gradient flow.
3. Training starts with samples being the partitions

## Training
### Preparing the training data
The training data should be in the following format:
- A folder containing the subfolders "train" and "val" (for training and validation data, respectively)
- Each of the two folders should contain the .h5 containers generated in the [Data Preparation step](Data_Preparation.md), as well as .star files that have the exact same name as the .h5 files, but with the .star extension

Here is an example of the folder structure:
```
training_data
│
└───train
│   │   membrane1.h5
│   │   membrane1.star
│   │   membrane2.h5
│   │   membrane2.star
│   │   ...
│
└───val
    │   membrane3.h5
    │   membrane3.star
    │   membrane4.h5
    │   membrane4.star
    │   ...
```

### Running the training
To start the training, run the following command:
```bash
membrain_pick train --data-dir <path-to-your-training-folder> --training-dir <path-to-your-output-folder>
```
Hereby, `<path-to-your-training-folder>` is the path to the folder containing the training data, and `<path-to-your-output-folder>` is the path to the folder where the training output should be stored.

The following options are available:
- `--data-dir`: Path to the folder containing the training data. (required)
- `--training-dir`: Path to the folder where the training output should be stored. (default: `./training_output`)
- `--project-name`: Name of the project. (default: `membrain_pick`)
- `--sub-name`: Subname of the project. (default: `0`)
- `--force-recompute-partitioning`: Should the partitioning be recomputed? (default: `no-force-recompute-partitioning`)
- `--input-pixel-size`: Pixel size of the tomogram. (default: `10.0`)
- `--device`: Device to use. (default: `cuda:0`)
- `--max-epochs`: Maximum number of epochs. (default: `200`)
- `--help`: Show this message and exit.


### Outputs
The output of this step is a folder `<path-to-your-output-folder>` containing subfolders `logs` and `mesh_cache`. The `logs` folder contains the training logs, while the `mesh_cache` folder contains temporary mesh data that is used during training. In case you would like to re-run training, you can keep the `mesh_cache` folder to speed up the process slightly. Otherwise, you can delete it.

Also, a folder `checkpoints` is created in `<path-to-your-output-folder>`, which contains the model checkpoints. These checkpoints can be used to resume training or to perform inference on new data.
By default, the last checkpoint, as well as the top 3 checkpoints based on the validation loss, are saved.

#### Note:
Training will be relatively slow in the first epoch, as spectral operators need to be computed once. From the second epoch onwards, training will be much faster.

### Next steps
Once the training is finished, you can use the trained model to perform inference on new data. Check the [Prediction](Prediction.md) documentation for more information.


#### Advanced options
In addition to the options above, there are several advanced options available for training. You can check the available options by running `membrain_pick train_advanced`.
Some of these are rather experimental and should be used with caution.


Options:
- `--position-tokens`: Tokens for the positions, as they are also specified in the _rlnClassNumber column of the GT star file. If columns are not present, the tokens are ignored and all positions used. (default: `None`)
- `--augment-all`: Should all data augmentations be used? (default: `augment-all`)
- `--aug-prob-to-one`: Should the probability be set to one (strong augmentations)? (default: `aug-prob-to-one`)
- `--k-eig`: Number of eigenvectors. (default: `128`)
- `--n-block`: Number of blocks. (default: `4`)
- `--c-width`: Width of the convolution. (default: `16`)
- `--conv-width`: Width of the convolution. (default: `16`)
- `--dropout`: Should dropout be used? (default: `no-dropout`)
- `--with-gradient-features`: Should the gradient features be used? (default: `with-gradient-features`)
- `--with-gradient-rotations`: Should the gradient rotations be used? (default: `with-gradient-rotations`)
- `--one-d-conv-first`: Should 1D convolution be used first? (default: `one-d-conv-first`)
- `--mean-shift-output`: Should the output be mean shifted and loss be computed on the clustering performance? (default: `no-mean-shift-output`)
- `--mean-shift-bandwidth`: Bandwidth for the mean shift. (default: `7.0`)
- `--mean-shift-max-iter`: Maximum number of iterations for the mean shift. (default: `10`)
- `--mean-shift-margin`: Margin for the mean shift. (default: `2.0`)

If mean shift is used, the loss is computed after the clustering. This can be useful in very crowded areas, and can also deal with label sparsity. However, efficiency is reduced. This feature is **highly** experimental and should be used with caution.


