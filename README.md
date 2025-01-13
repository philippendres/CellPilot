# CellPilot
This repository contains the code of CellPilot, a deep learning-based method for the automatic and interactive segmentation of cells and glands in histological images. CellPilot uses [SAM](https://arxiv.org/abs/2304.02643) and [CellViT](https://arxiv.org/abs/2306.15350) and is fine-tuned on large-scale segmentation datasets of histological images.

A Preprint of the CellPilot paper is available on [Arxiv](https://arxiv.org/abs/2411.15514).
The model architecture is shown below:
![Model](./images/model.png).


## Key Features
- **Automatic Segmentation**: CellPilot allows users to automatically segment cells in histological images.
- **Interactive Segmentation**: CellPilot allows users to interactively segment cells and glands in histological images.

## Setup
We provide a manual setup which allows for training and inference and a docker setup solely for inference.
### Manual Setup
1. Clone the repository with: `git clone https://github.com/philippendres/CellPilot.git`
2.  - For inference and training: Download the weights of the CellPilot model and the CellViT model: [CellPilot](https://1drv.ms/u/c/696e40c0eaa91ac3/EfwN6MP4IqpHityPWHnHkkgBh5MyZEdYGVf9soXDs0gKOg?e=M8DVNQ), [CellViT](https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q), [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    - For model comparisons: Download the weights of the SimpleClick and the MedSAM model: [SimpleClick](https://drive.google.com/file/d/1dLAEFXhnk_Nebq3Net11sf6MjRCBEe0O/view?usp=drive_link), [MedSAM](https://drive.google.com/file/d/1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_/view?usp=drive_link)
3. Place the downloaded weights in the models directory.
4. Create a new conda environment with the provided environment.yml file:
    ```
    conda env create -f environment.yml
    conda activate histo3.10
    ```
5. Install the resources:
    ```
    cd resources
    cd CellViT
    git submodule init
    git submodule update
    git submodule update --remote
    pip install -e .
    cd ..
    cd SimpleClick
    git submodule init
    git submodule update
    git submodule update --remote
    pip install -e .
    cd ..
    cd ..
    ```
6. Install our package: 
    ```
    pip install -e .
    ```


### Docker Setup
1. Clone the repository with: `git clone https://github.com/philippendres/CellPilot.git`
2. Download the weights of the CellPilot model and the CellViT model: [CellPilot](https://1drv.ms/u/c/696e40c0eaa91ac3/EfwN6MP4IqpHityPWHnHkkgBh5MyZEdYGVf9soXDs0gKOg?e=M8DVNQ), [CellViT](https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q), [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
3. Place the downloaded weights in the models directory.
4. Build the docker image with: `docker build -t cellpilot .`
5. Run the docker image with: `docker run -p 7860:7860 cellpilot`
6. Open the webapplication with the following link: [localhost:7860](http://localhost:7860)

## Usage with Manual Setup
### App
Run the gradio webapplication with the following command:
```
python scripts/app.py --model_dir models --model_name samhi --cellvit_model CellViT-256-x40.pth
```
The app has the following arguments:
- **model_dir**: The directory where the CellPilot and the CellViT model are stored.
- **model_name**: The name of the CellPilot model. (append .ckpt)
- **cellvit_model**: The name of the CellViT model.

The command above will generate a link to a webapplication where you can upload your own images and segment them with SAMHI.
The app will look like this:
![App](./images/app.png)
The app has the following features:
- **Upload Image**: Upload your own image to segment in the upper left corner.
- **Auto Segment**: Automatically segment the uploaded image with CellPilot.
- **Add Mask**: Interactively add a mask with CellPilot by drawing points and bounding boxes on the image.
- **Refine Mask**: Refine an existing mask by drawing points and bounding boxes on the image.
- **Remove Mask**: Remove an existing mask by clicking on it.
- **Move the Image**: Move the image with the arrow symbols.
- **Zoom the Image**: Zoom the image with the zoom bar.

### Inference
For code-based inference with CellPilot, have a look at the [automatic segmentation](./notebooks/automatic.ipynb) and [interactive segmentation](./notebooks/interactive.ipynb) notebooks.

### Training
For training the CellPilot model, you first need to process the data. Have a look at the [data processing](./cellpilot/data_processing/data_processing.md) for more information.
Then you can train the CellPilot model with the following command and log your results on [Weights & Biases (W&B)](https://wandb.ai/):
```
python scripts/train.py --project_name <project_name> --entity <entity> --run_directory <run_directory> --model_dir <model_dir> --display_name <display_name> --datasets <training datasets> --test_datasets <test datasets> --cluster <cluster>
```
The training script has the following arguments:
- **project_name**: The name of the project on W&B.
- **entity**: The name of the entity on W&B.
- **run_directory**: The directory where the training runs are stored.
- **model_dir**: The directory where the CellPilot and the SAM model are stored.
- **display_name**: The name of the display on W&B.
- **datasets**: The training datasets named according to [data_processing.md](./cellpilot/data_processing/data_processing.md).
- **test_datasets**: The test datasets named according to [data_processing.md](./cellpilot/data_processing/data_processing.md).
- **cluster**: cluster to run the training on. For your own computer set this to custom and follow the instructions on [data_processing.md](./cellpilot/data_processing/data_processing.md). If you are working at Helmholtz and have access to the cluster set this to helmholtz.
You can modify all these parameters to fit your needs. Have a look at the [training](./scripts/train.py) script for more information.
## Code Structure
### Overview
The codebase is organized as follows:

- `cellpilot/`: Contains the data processing, model definitions and inference code .
- `notebooks/`: Contains examplenotebooks for automatic and interactive segmentation
- `resources/`: Includes necessary resources and submodules like CellViT and SimpleClick.
- `scripts/`: Contains the main scripts for running the application and training the models.
- `utils/`: Utility functions that assist in data preprocessing.




## Citation
If you use CellPilot in your research, please cite our paper:
```bibtex
@misc{endres2024cellpilotunifiedapproachautomatic,
    title={CellPilot: A unified approach to automatic and interactive segmentation in 
    histopathology}, 
    author={Philipp Endres and Valentin Koch and Julia A. Schnabel and Carsten Marr},
    year={2024},
    eprint={2411.15514},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2411.15514}, 
}
```

<!-- ### Training
Check that the data is stored in the structure given in [data_processing.md](./samhi/data_processing/data_processing.md)
Run the training script with the following command:
```
python train.py
```
The evaluation script has the following arguments:
- **cluster**: The cluster to run the training on.
- **model_dir**: The directory where the SAMHI and the CellViT model are stored. -->


