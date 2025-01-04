# MemBrain-pick prediction

## Overview
The prediction step is used to predict partcle positions on membrane surfaces using a trained model. What it does it:
1. Partition the meshes into small "partitions" (default size: 2000 vertices) to match training data
2. Predict particle distance maps for each partition
3. Combine the distance maps into a single one convering the whole mesh
4. Extract particle positions from the distance map using Mean Shift clustering
5. Export the particle positions to a .star file, together with membrane-alignment orientations

## Prediction
### Preparing the prediction data
The prediction data should be stored in one folder containing the .h5 containers generated in the [Data Preparation step](Data_Preparation.md). Here is an example of the folder structure:
```
prediction_data
│
└───membrane1.h5
└───membrane2.h5
└───...
```

You can theoretically use the entire folder generated in the Data Preparation step.

### Running the prediction
To start the prediction, run the following command:
```bash
membrain_pick predict --data-dir <path-to-your-datafolder> --ckpt-path <path-to-your-checkpoint> --out-dir <path-to-store-output>
```

This command will predict the output of the trained model on the given data. The following options are available:
- `--data-dir`: Path to the folder containing the data to predict. (required)
- `--ckpt-path`: Path to the checkpoint. (required)
- `--out-dir`: Path to the folder where the output should be stored. (default: `./predict_output`)
- `--partition-size`: Size of the partition. (default: `2000`)
- `--input-pixel-size`: Pixel size of the tomogram. (default: `10.0`)
- `--force-recompute-partitioning`: Should the partitioning be recomputed? (default: `no-force-recompute-partitioning`)
- `--n-block`: Number of blocks. (default: `4`). Important to match the number chosen during training.
- `--c-width`: Width of the convolution. (default: `16`). Important to match the number chosen during training.
- `--conv-width`: Width of the convolution. (default: `16`). Important to match the number chosen during training.
- `--k-eig`: Number of eigenvectors. (default: `128`). Important to match the number chosen during training.
- `--mean-shift-output`: Should the output be clustered? (default: `mean-shift-output`)
- `--mean-shift-bandwidth`: Bandwidth for the mean shift. (default: `7.0`)
- `--mean-shift-max-iter`: Maximum number of iterations for the mean shift. (default: `150`)
- `--mean-shift-margin`: Margin for the mean shift. (default: `0.0`)
- `--mean-shift-score-threshold`: Score threshold for the mean shift. (default: `9.0`)
- `--mean-shift-device`: Device to use for the mean shift. (default: `cpu`)
- `--help`: Show this message and exit.

If the mean-shift-output option is set to `mean-shift-output`, the output will be clustered using Mean Shift clustering. Otherwise, only distance maps will be generated.

### Outputs
The output will be stored in the specified output folder. For each input .h5 file, a new .h5 file will be created containing the predicted distance maps and potential particle positions (if clustering was enabled). Additionally, a .star file will be created containing the particle positions and membrane-alignment orientations (again, if clustering was enabled).
Here is an example of the output folder structure:
```
predict_output
│
└───membrane1.h5
└───membrane1.star
└───membrane2.h5
└───membrane2.star
└───...
```

### Next steps
Once the prediction is finished, you can use the predicted particle positions to perform further analysis, e.g. subtomogram averaging, or extract some statistics using [MemBrain-stats](https://github.com/LorenzLamm/membrain-stats/tree/main/src/membrain_stats). 

Before doing further analysis, it is highly recommended to first look at the predicted particle positions in surforama to check if the prediction was successful. You can find instructions on how to do that in the [MemBrain-pick surforama documentation](Surforama_Inspection.md).


## Alternative: run clustering on a precomputed distance map
If you have already computed the distance maps and want to run the clustering step only (e.g. when you didn't specify the `--mean-shift-output` option during prediction, or you would like to change clustering parameters), you can use the following command:
```bash
membrain_pick mean_shift --h5-path <path-to-your-h5> --out-dir <path-to-store-output> --bandwidth <bandwidth> 
```

This command will cluster the distance map stored in the given .h5 file. The following options are available:
- `--h5-path`: Path to the .h5 file. (required)
- `--out-dir`: Path to the folder where the output should be stored. (default: `./mean_shift_output`)
- `--bandwidth`: Bandwidth for the mean shift. (default: `7.0`)
- `--max-iter`: Maximum number of iterations for the mean shift. (default: `150`)
- `--margin`: Margin for the mean shift. (default: `0.0`)
- `--score-threshold`: Score threshold for the mean shift. (default: `9.0`)
- `--method`: Method to use for the mean shift. Choose from `membrain_pick` or `membrainv1`. (default: `membrain_pick`)
- `--device`: Device to use for the mean shift. (default: `cuda:0`)
- `--help`: Show this message and exit.


### Note:
The `--method` option is used to specify which version of the mean shift algorithm to use. The `membrain_pick` method is the default and recommended method. The `membrainv1` method can be a good alternative if you run into any memory issues with the default method. However, the `membrainv1` method is slower and might produce slightly different results.
