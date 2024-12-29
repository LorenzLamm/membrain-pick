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
    <img src="https://private-user-images.githubusercontent.com/34575029/398807289-2d593a3e-b540-46b4-9db9-a3a4980ae0d2.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEwMjIsIm5iZiI6MTczNTI1MDcyMiwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODktMmQ1OTNhM2UtYjU0MC00NmI0LTlkYjktYTNhNDk4MGFlMGQyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMDUyMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM3N2Y5YTQ2MDM4ZjY5ODk3NTNjOTQyNzVlYmY2YTJhNzcxMmZiNDM3ZTUxMDExNjYyYTZlNGNmZDdmYTRhOGImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.gp0ImZhcceZji7pBrBJkvC2UNbWgOCOSR6tKjIpCrXk" alt="surforama_initial_screen" width="49%" />
    <img src="https://private-user-images.githubusercontent.com/34575029/398807287-4dd2d086-fc53-4dae-99c4-bfdd5feec31a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEwMjIsIm5iZiI6MTczNTI1MDcyMiwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODctNGRkMmQwODYtZmM1My00ZGFlLTk5YzQtYmZkZDVmZWVjMzFhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMDUyMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTljNDVlYWIxZTUzZTY1N2MwZjIxOWE0NjdkNDE0MmI4ZDgwNWM5MzM2YmMzYzJkNTE1NWYxMjM3ODgyNDZjODQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.DUuIjfwWtbqTU2ZdvlqPIij1DAtm1e81OJDbAf3Gexc" alt="surforama_projected_input" width="49%" />
</div>

### 2. Annotating Membrane-Associated Particles
#### Activate annotation mode
To annotate membrane particles, you should deactivate the "Projections" layer by clicking on the eye icon next to it. This shows the "Surfogram" layer, where you can smoothly change the distance of the normal projection via the slider on the right.
Clicking on the "Enable" button on the right will initialize the Points layer on the left, where annotated points will intermediately be stored. 



<div style="text-align: center;">
    <img src="https://private-user-images.githubusercontent.com/34575029/398807285-3958aa60-99a5-4be5-b45e-978aa5086d10.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEzMjksIm5iZiI6MTczNTI1MTAyOSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODUtMzk1OGFhNjAtOTlhNS00YmU1LWI0NWUtOTc4YWE1MDg2ZDEwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTAyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM4YWJlNGY5ZDdhYzljOTljY2YyYjU3OTI2OWIzNjRlMzU2YjcyY2FhMjBmNDlkZTRkOTZiNDVhZGJkZmQwOTQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.roS5zYWdNkyIhDf5hiRFvPKUQBlwuoxo9CDvod8naO4" alt="surforama_surforama" width="49%" />
    <img src="https://private-user-images.githubusercontent.com/34575029/398807283-943da629-eb24-4f16-a91a-fc0114728dc9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEzMjksIm5iZiI6MTczNTI1MTAyOSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODMtOTQzZGE2MjktZWIyNC00ZjE2LWE5MWEtZmMwMTE0NzI4ZGM5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTAyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVkYTA0NjQzYWY5MTJjNGFmN2I0ZDdmYThjOGZhYmU4OTYyNzIzMGI1MjQ3MDIyYmIyYjBlN2ZjZGM0ZmVjMTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.5aBT840QDd225iVH15NhtygO-43jq9zU_KXWK4qif54" alt="surforama_points_enable" width="49%" />
</div>

#### Annotate particle positions

By clicking on positions on the mesh, new points will be added to the Points layer. Once you are done with the annotations, you can save out the positions as a RELION-type .star file by clicking "Save to star file" after specifying the output path.

**Important!** Make sure to have the "Surforama" layer active when clicking the points, otherwise it will not work.

<div style="text-align: center;">
    <img src="https://private-user-images.githubusercontent.com/34575029/398807281-11415e76-d5ea-4824-80fd-4b3b7de3da7c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEzMjksIm5iZiI6MTczNTI1MTAyOSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODEtMTE0MTVlNzYtZDVlYS00ODI0LTgwZmQtNGIzYjdkZTNkYTdjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTAyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWUyZTg0YTc1MzFjMTkwNWE4NzAxYjZlZWQyYTlkNzcxM2IxZWE4MWZmNzNlNTZhNDJiODc0NzgzODBkN2FkZTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.XRiHHLCi5dWzlsmcDiYe827FaK9n9WLVbQ_yMo_WUU4" alt="surforama_add_points" width="49%" />
    <img src="https://private-user-images.githubusercontent.com/34575029/398807280-1c457184-cc92-4dcf-b93c-0bd1b24652ae.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTEzMjksIm5iZiI6MTczNTI1MTAyOSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyODAtMWM0NTcxODQtY2M5Mi00ZGNmLWI5M2MtMGJkMWIyNDY1MmFlLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTAyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZiNDg5ODhmMzQ5NDI1ZTc4ZTVkMmE0NTUyYjBlOGE1NmM1YTg3YWI4MGI3NzI2NWE4YzA3ZTUwMjFkMzc4MzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.gx5PayeHnDCISj0aTzSTK-Dp-eeR6rV7PBzsbwobXj8" alt="surforama_save_star" width="49%" />
</div>

#### Alter particle positions
In case you are not happy with the clicks you made or you would like to correct MemBrain-pick's predictions, you can alter the positions by 

1. activating the Points layer on the left
2. click the selection icon on the top left
3. a) **drag** the point to a new position or b) **delete** the point by first clicking the point and then clicking the delete button on the top left


<div style="text-align: center;">
    <img src="https://private-user-images.githubusercontent.com/34575029/398807279-e25d36aa-17d3-4574-b40a-5856b24eed3c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTE4NDksIm5iZiI6MTczNTI1MTU0OSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyNzktZTI1ZDM2YWEtMTdkMy00NTc0LWI0MGEtNTg1NmIyNGVlZDNjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTkwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcxNjA4N2IzNzVkYWM1YjBkY2JlZmE0NjlkMmNmY2JjNjhmMzE2ZWY3ZDU0NGZiODVkNTQyNzUzODZiY2RjN2QmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.Nua03PtnWYuQyWrkHS54ZZkJsifSO874zrp9s1w0o4E" alt="surforama_drag_points" width="49%" />
    <img src="https://private-user-images.githubusercontent.com/34575029/398807277-50784233-caac-4f45-aa59-4b4a39ccaf17.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzUyNTE4NDksIm5iZiI6MTczNTI1MTU0OSwicGF0aCI6Ii8zNDU3NTAyOS8zOTg4MDcyNzctNTA3ODQyMzMtY2FhYy00ZjQ1LWFhNTktNGI0YTM5Y2NhZjE3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI2VDIyMTkwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU3MzUyN2VlZDAyMTEwNzA3ZjA1M2Q3YjU2ZjZlZjg5ZDgxMDg2ZTU5ZjQzOGUyOWZjYzU4NDA2ZmMyYjYzODkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.yjkw7ycpUIxXJwpcjHvvN_Qz1xoDMZ1eeu1-wv_DIfc" alt="surforama_delete_points" width="49%" />
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
    <img src="https://private-user-images.githubusercontent.com/34575029/399162666-e3131f7d-8bb7-41c9-8750-e77c62a70da3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzU0Nzg1OTYsIm5iZiI6MTczNTQ3ODI5NiwicGF0aCI6Ii8zNDU3NTAyOS8zOTkxNjI2NjYtZTMxMzFmN2QtOGJiNy00MWM5LTg3NTAtZTc3YzYyYTcwZGEzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI5VDEzMTgxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTMzZjgwNmYxNDIxNTEzZDAzYzQ5N2IwMzdmNTQ3MWZmZGFkMzg2ZGM1NDdkYTdmYjM4NDc1NmRlN2Y0NDQ3YmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.bd0tWG4JXb0Bw8dGjzyxmeh9gk1LS-BiXOs3oxEtr-A" alt="surforama_drag_points" width="49%" />
    <img src="https://private-user-images.githubusercontent.com/34575029/399162682-54f82cc1-c1e4-431f-917f-09fbf410d3c3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzU0Nzg1OTYsIm5iZiI6MTczNTQ3ODI5NiwicGF0aCI6Ii8zNDU3NTAyOS8zOTkxNjI2ODItNTRmODJjYzEtYzFlNC00MzFmLTkxN2YtMDlmYmY0MTBkM2MzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjI5VDEzMTgxNlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI2N2UzMGE0NjNhM2FiYTQ1YTdhYTU2NzA4MTliNmM3MzQ0ZjM5NmUxNTZmYzRmYzExYTE0NTQ1NDc3YmJhMzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.PkoJoJlD_WKl0n8USvgG4Q23_mRM8I9zyE1TPVq5g9Y" alt="surforama_delete_points" width="49%" />
</div>

If desired, you can now alter the positions of the predicted particles as described in the previous section, and save the altered positions as a RELION-type .star file. With these new positions, you can either train a more refined model or use them for further analysis.