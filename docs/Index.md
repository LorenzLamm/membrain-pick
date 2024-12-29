# Membrain-Pick

## Overview
**Membrain-Pick** is part of the **MemBrain** suite of tools for processing membranes in cryo-electron tomography. MemBrain-pick's main purpose is to localize membrane-associated particles in the tomograms.
To this end, MemBrain-picks takes as input already existing membrane segmentations and processes these to limit the search space for membrane-associated particles. The output of MemBrain-pick is a set of coordinates that can be used for further analysis.

## Workflow
The workflow of MemBrain-pick is as follows:
1. **Input**: Membrane segmentations in the form of a binary mask (.mrc). Ideally, these segmentations should depict single membrane instances.
2. **Mesh Generation**: The membrane segmentations are converted into a mesh representation. At this stage, also tomogram densities are projected onto the membrane mesh.
3. **Ground Truth Generation**: The membrane mesh can be loaded into **surforama** to manually annotate membrane-associated particles. These annotations can then be used to train a MemBrain-pick model.
4. **Training**: The generated meshes, along with the annotations, are used to train a model that can predict the location of membrane-associated particles.
5. **Prediction**: The trained model is used to predict the location of membrane-associated particles in the membrane segmentations.

### Key Functionalities
- **Mesh Conversion**: Transform membrane segmentations into a mesh representation that can easily be processed by MemBrain-pick and surforama.
- **Model training**: Train a model to predict the location of membrane-associated particles.
- **Prediction**: Use the trained model to predict the location of membrane-associated particles in membrane segmentations.
- **Initial orientaton assignment**: Given a set of positions, MemBrain-pick can assign initial orientations to the particles by aligning them with the membrane normal.
-- **integration with surforama**: MemBrain-pick can be used in conjunction with surforama to manually annotate membrane-associated particles.
---

## Jump to
- [Installation](Installation.md)
- [Data Preparation](Data_Preparation.md)
- [Training](Training.md)
- [Prediction](Prediction.md)
- [Angle Assignment](Angle_Assignment.md)
- [Visualization in Surforama](Surforama_Inspection.md)