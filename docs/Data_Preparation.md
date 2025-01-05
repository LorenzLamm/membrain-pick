# Data Preparation

## Overview
MemBrain-pick's first step is to generate mesh representations of the membrane segmentations. These membrane segmentations should be in the form of binary masks. Ideally, these masks should depict single membrane instances. The membrane segmentations can be generated using tools like [MemBrain-seg](https://github.com/teamtomo/membrain-seg). Cropping of individual membranes can be done in the MemBrain Napari plugin or any other software like Amira, IMOD, etc.

### Important Explanations
This step is crucial for the performance of MemBrain-pick. Particularly, the choices of 
- **step numbers** for the normal vectors, and
- **step size** for the normal vectors,

are important because they determine the amount and granularity of the data the network will see.
Therefore, after generating the meshes, it makes sense to inspect the generated `normal_values` in surforama, as described in the [MemBrain-pick surforama documentation](Surforama_Inspection.md).

Here is a schematic visualization of the normal vectors:

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/0df51395-d939-4b1f-8fcc-75db42865fc5" alt="stepsize_schematic" width="50%" />
</div>

### Inputs
- **tomogram file**: The tomogram file -- can be raw, denoised, filtered. We observed best results with denoised tomograms, and potentially with some filtering applied.
- **membrane segmentations**: Each membrane segmentation should be stored in a separate binary mask file. The binary mask should be in the form of a 3D MRC file. The membrane segmentations should ideally depict single membrane instances.

### Functions
Depending on your data structure, you can use the following functions to prepare your data for MemBrain-pick:

#### 1. You have a folder full with membrane segmentations:


```bash
some_directory
├── tomograms
│   ├── Tomo0001.mrc
│   └── Tomo0002.mrc
│       ...
├── Tomo0001_membranes
│   ├── mb1.mrc
│   ├── mb2.mrc
│   ├── mb3.mrc
│   └── mb4.mrc
│       ...
├── Tomo0002_membranes
...
``` 

Then, you can process the segmentation folders in bulk using the following command:

```bash

membrain_pick convert_mb_folder --mb-folder <path-to-your-folder> --tomo-path <path-to-tomo> --out-folder <path-to-store-meshes>
```

This command will generate mesh representations of the membrane segmentations and project the tomogram densities onto the membrane meshes. The generated meshes will be stored in the specified output folder.

#### More Options

- `--mb-folder` (TEXT, required): Path to the folder containing the membrane segmentations. [default: None] 
- `--tomo-path` (TEXT, required): Path to the tomogram to be projected. [default: None] 
- `--out-folder` (TEXT): Path to the folder where mesh projections should be stored. [default: ./mesh_data] 
- `--step-numbers` (INTEGER): Step numbers for the normal vectors. [default: (-10, 10)] 
- `--step-size` (FLOAT): Step size for the normal vectors. [default: 2.5] 
- `--mesh-smoothing` (INTEGER): Smoothing factor for the mesh. [default: 1000] 
- `--barycentric-area` (FLOAT): Barycentric area for the mesh. [default: 400.0] 

Note: options may change faster than these docs. You can check the available options by running `membrain_pick convert_mb_folder`.

#### 2. You have a single membrane segmentation

You can also convert a single membrane segmentation to a mesh using the following command:

```bash
membrain_pick convert_file --tomogram-path <path-to-your-tomo> --mb-path <path-to-your-membrane-segmentation> --out-folder <path-to-store-meshes>
```

This command will generate a mesh representation of the membrane segmentation and project the tomogram densities onto the membrane mesh. The generated mesh will be stored in the specified output folder.

#### More Options

Other options are similar to the `convert_mb_folder` command. You can check the available options by running `membrain-pick convert_file`.


### Outputs
The output of this step is a set of data containers which contain all necessary information about the meshes, e.g.
- mesh information (vertices, faces)
- pixel size
- tomogram path

All this information is stored together in some `.h5` files, which can be inspected in surforama. 
At the same time, the meshes are stored as `.obj` files in the specified output folder, e.g.

```bash
output_directory
├── Tomo0001_mb1.obj
├── Tomo0001_mb1.h5
├── Tomo0001_mb2.obj
├── Tomo0001_mb2.h5
├── ...
```

You can visualize the meshes in surforama to check if the mesh generation was successful, and to annotate ground truth particle positions.
Check instructions on that in the [MemBrain-pick surforama documentation](Surforama_Inspection.md).