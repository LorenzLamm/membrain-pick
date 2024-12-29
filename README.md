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
- [Installation](docs/Installation.md)
- [Data Preparation](docs/Data_Preparation.md)
- [Training](docs/Training.md)
- [Prediction](docs/Prediction.md)
- [Angle Assignment](docs/Angle_Assignment.md)
- [Visualization in Surforama](docs/Surforama_Inspection.md)


MemBrain-pick is part of the MemBrain v2 [1] package and still under early development. If you have any questions or suggestions, please contact us at lorenz.lamm@helmholtz-munich.de

```
[1] Lamm, L., Zufferey, S., Righetto, R.D., Wietrzynski, W., Yamauchi, K.A., Burt, A., Liu, Y., Zhang, H., Martinez-Sanchez, A., Ziegler, S., Isensee, F., Schnabel, J.A., Engel, B.D., and Peng, T, 2024. MemBrain v2: an end-to-end tool for the analysis of membranes in cryo-electron tomography. bioRxiv, https://doi.org/10.1101/2024.01.05.574336

```