# MemBrain-pick's surforama usage

## Overview
**surforama** is a tool for visualizing and annotating membrane meshes. MemBrain-pick works closely with surforama to inspect generated meshes, projected tomogram densities, and manually annotate membrane-associated particles. After prediction, you can also visualize the predicted heatmaps and particle center positions in surforama. This document provides an overview of how to use surforama to inspect membrane containers and manually annotate membrane-associated particles.

## Command
If you have generated a membrane containers, e.g. as described in the [Data Preparation](Data_Preparation.md) documentation, or as output of the MemBrain-pick model, you can inspect these in surforama using the following command:

```bash
membrain_pick surforama --h5-path <path-to-your-h5-container>
```

This will automatically start Napari with the membrane meshes and projected tomogram densities loaded. You can then use the Napari GUI to inspect the meshes and densities, and manually annotate membrane-associated particles.

### More options:
- `--h5-path` (TEXT, required): Path to the h5 container. [default: None]
- `--tomogram-path` (TEXT): Path to the tomogram to be projected (overwrites the path in the h5 container). [default: None]
- `--normal-offset` (FLOAT): Offset for the normal vectors. [default: 0.0]
- `--point-size` (FLOAT): Size of the points. [default: 5.0]

### Pro Tip:
If you run the command with `h5-path` pointing to a directory, all h5 files in that directory will be loaded into surforama.


## Usage

### 1. Inspecting MemBrain-pick input 
After loading the membrane container, a screen like the following will appear. 
You can then visualize the densities stored in the container by changing the selected feature channel in the bottom right corner:

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/2d593a3e-b540-46b4-9db9-a3a4980ae0d2" alt="surforama_initial_screen" width="49%" />
    <img src="https://github.com/user-attachments/assets/4dd2d086-fc53-4dae-99c4-bfdd5feec31a" alt="surforama_projected_input" width="49%" />
</div>

### 2. Annotating Membrane-Associated Particles
#### Activate annotation mode
To annotate membrane particles, you should deactivate the "Projections" layer by clicking on the eye icon next to it. This shows the "Surfogram" layer, where you can smoothly change the distance of the normal projection via the slider on the right.
Clicking on the "Enable" button on the right will initialize the Points layer on the left, where annotated points will intermediately be stored. 



<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/3958aa60-99a5-4be5-b45e-978aa5086d10" alt="surforama_surforama" width="49%" />
    <img src="https://github.com/user-attachments/assets/943da629-eb24-4f16-a91a-fc0114728dc9" alt="surforama_points_enable" width="49%" />
</div>

#### Annotate particle positions

By clicking on positions on the mesh, new points will be added to the Points layer. Once you are done with the annotations, you can save out the positions as a RELION-type .star file by clicking "Save to star file" after specifying the output path.

**Important!** Make sure to have the "Surforama" layer active when clicking the points, otherwise it will not work.

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/11415e76-d5ea-4824-80fd-4b3b7de3da7c" alt="surforama_add_points" width="49%" />
    <img src="https://github.com/user-attachments/assets/1c457184-cc92-4dcf-b93c-0bd1b24652ae" alt="surforama_save_star" width="49%" />
</div>

#### Alter particle positions
In case you are not happy with the clicks you made or you would like to correct MemBrain-pick's predictions, you can alter the positions by 

1. activating the Points layer on the left
2. click the selection icon on the top left
3. a) **drag** the point to a new position or b) **delete** the point by first clicking the point and then clicking the delete button on the top left


<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/e25d36aa-17d3-4574-b40a-5856b24eed3c" alt="surforama_drag_points" width="49%" />
    <img src="https://github.com/user-attachments/assets/50784233-caac-4f45-aa59-4b4a39ccaf17" alt="surforama_delete_points" width="49%" />
</div>


In the workflow of MemBrain-pick, you can use the surforama-generated GT files to train your first MemBrain-pick model, as described in the [Training](Training.md) documentation.


### 3. Inspecting MemBrain-pick output

After running prediction as described in the [Prediction](Prediction.md) documentation, you can inspect the predicted heatmaps and particle center positions in surforama. To do this, you can run the following command:

```bash
membrain_pick surforama --h5-path <path-to-your-predicted-h5-container>
```

This will automatically start Napari with the predicted heatmaps and particle center positions loaded. You can then use the Napari GUI to inspect the heatmaps and particle center positions, and manually alter the positions if needed.
By default, the surforama view is loaded together with the predicted positions. If you would like to look at the heatmaps instead of the projected tomogram densities, you can simply un-visualize the "Surfogram" layer to make the "Scores" layer visible:

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/e3131f7d-8bb7-41c9-8750-e77c62a70da3" alt="surforama_inspect_points" width="49%" />
    <img src="https://github.com/user-attachments/assets/54f82cc1-c1e4-431f-917f-09fbf410d3c3" alt="surforama_inspect_heatmap" width="49%" />
</div>


https://private-user-images.githubusercontent.com/34575029/398803859-0df51395-d939-4b1f-8fcc-75db42865fc5.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzYxMDEzODEsIm5iZiI6MTczNjEwMTA4MSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDM4NTktMGRmNTEzOTUtZDkzOS00YjFmLThmY2MtNzVkYjQyODY1ZmM1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAxMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMTA1VDE4MTgwMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRkOTMwM2RlMzlmNTY2Mjk2ZWYxMzI2ZDQxOWMyZGEyMjYwMGRlYTJlOGQyMDUwM2RjYTU2ODYyZmEyOTViZGUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.6cc5AMQlIRSEBeAqS-z6BSCKAH8l4bz_kNw1BcBAzIY
If desired, you can now alter the positions of the predicted particles as described in the previous section, and save the altered positions as a RELION-type .star file. With these new positions, you can either train a more refined model or use them for further analysis.