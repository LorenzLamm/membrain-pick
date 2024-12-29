# Angle Assignment

## Overview

This functionality simply takes as input a set of points and a membrane (either segmentation or mesh) and assigns angles to each point based on the normal vector of the membrane at that point. This will lead to the membrane being horizonally aligned in subvolumes extracted and rotated with the corresponding angles.
Output is in the form of a star file with the angles assigned to each point.

## Example Usage

```bash
membrain_pick assign_angles --position-file <path_to_positions_file> --obj-file <path_to_obj_file> --out-dir <path_to_output_directory> --position-scale-factor <rescale_factor> 
```

This will assign angles to each point in the positions file based on the normal vector of the membrane mesh. The output will be stored in the output directory as a star file with the angles assigned to each point.
If you don't have a mesh file, you can provide a segmentation file instead and a mesh will be generated from the segmentation file internally:

```bash
membrain_pick assign_angles --position-file <path_to_positions_file> --segmentation-file <path_to_segmentation_file> --out-dir <path_to_output_directory> --position-scale-factor <rescale_factor> 
```

Note that the positions scale factor is important to assign the positions to their correct places on the membranes. By default, MemBrain-pick's output is scaled in Angstrom, whereas MemBrain-pick meshes are scaled in pixels. Therefore, you need to provide the scale factor to rescale the positions to match the mesh dimensions (e.g. 1 / pixel size).
