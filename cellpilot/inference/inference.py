import torch
from cellpilot.modeling.model import SamHI
import wandb
from pathlib import Path
import numpy as np
import os 
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image, InterpolationMode
from cellpilot.modeling.predictor import SamHIPredictor
from cellvit.models.segmentation.cell_segmentation.cellvit import CellViT256
from torchvision import transforms as T
import torch.nn.functional as F
from typing import Dict, List, Tuple

class Inference:
    def __init__(self, config):
        self.load_config(config)
        self.initialize_model()
        self.initialize_cellvit()
    
    def load_config(self, config):
        self.device = config["device"]
        self.model_dir = config["model_dir"]
        self.model_name = config["model_name"]

    def initialize_model(self):
        model_path = os.path.join(self.model_dir, self.model_name)
        if not Path(model_path).is_file():
            print("Downloading model from wandb")
            api = wandb.Api()
            artifact = api.artifact("philippresearch/SAMHI/" + self.model_name, type="model")
            artifact_dir = artifact.download(root=self.model_dir)
            os.rename(artifact_dir + "model.ckpt", model_path)
        model_config = torch.load(model_path, map_location=lambda storage, loc: storage)["hyper_parameters"]["config"]["model_config"]
        model_config["model_mode"] = "inference"
        model_config["model_dir"] = self.model_dir
        config = {"model_config": model_config}
        model = SamHI.load_from_checkpoint(model_path, config=config).to(self.device)
        model.eval()
        self.model = model
        self.predictor = SamHIPredictor(model.model, p_tuning=model.p_tuning)

    def initialize_cellvit(self, model_name="CellViT-256-x40.pth"):
        model_path = os.path.join(self.model_dir, model_name)
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        model = CellViT256(model256_path=None,
                num_nuclei_classes=config["data.num_nuclei_classes"],
                num_tissue_classes=config["data.num_tissue_classes"],
                regression_loss=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        inference_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=mean, std=std)]
        )
        self.cellvit = model
        self.cellvit_transforms = inference_transforms
        self.cellvit_mean = mean
        self.cellvit_std = std
        self.cellvit.eval()


    def segment(
        self, 
        prompt, 
        image = None
    ) -> np.ndarray:
        """
        Segment the image based on the prompt

        Arguments:
            prompt (dict): The prompt for the segmentation. It should contain the following keys:
                - point_coords (np.ndarray): The coordinates of the points. 
                    A Nx2 array of point prompts to the model.
                    Each point is in (X,Y) in pixels.
                - point_labels (np.ndarray): The labels of the points.
                    A length N array of labels for the point prompts. 
                    1 indicates a foreground point and 0 indicates background point.
                - boxes (np.ndarray): The bounding boxes.
                    A length 4 array given a box prompt to the model, in XYXY format.
            image (np.ndarray): The image to segment. If None, it is assumed that the image is already set.
                Expects an image in HWC uint8 format, with pixel values in [0, 255].
        
        Returns:
            np.ndarray: The mask of the segmentation. A binary mask of the same shape as the input image.
        """
        if image is not None:
            try:
                self.predictor.set_image(image)
            except:
                raise("Error: Please provide an image")
        mask, _, _ = self.predictor.predict(point_coords=prompt.get("point_coords", None), point_labels=prompt.get("point_labels", None), box=prompt.get("boxes", None), multimask_output=False)
        return mask[0]
    
    def segment_automatically(
        self, 
        image
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Segment the image automatically using CellViT

        Arguments:
            image (np.ndarray): The image to segment. 
                Expects an image in HWC uint8 format, with pixel values in [0, 255].

        Returns:
            Tuple[np.ndarray, List[Dict]]: The segmented image and the prompts for the segmentation.   
        """
        if image.shape[0] > 1024 or image.shape[1] > 1024:
            h, w = ResizeLongestSide.get_preprocess_shape(img.shape[0], img.shape[1], 1024)
            img = resize(to_pil_image(img), (h, w), InterpolationMode.BILINEAR)
        instance_types = self.detect_cellvit(image)
        self.predictor.set_image(image)
        masks = np.zeros(image.shape[:2])
        prompts = []
        for k in instance_types[0].keys():
            box = instance_types[0][k]["bbox"]
            box = np.array([box[0,1], box[0,0], box[1,1], box[1,0]])
            mask, _, _ = self.predictor.predict(box=np.array(box), multimask_output=False)
            masks[mask[0] == 1] = int(k)
            prompts.append({"boxes":np.array(box)})
        return masks, prompts

    def detect_cellvit(self, image):
        """
        Detect the nuclei in the image using CellViT

        Arguments:
            image (np.ndarray): The image to segment. 
                Expects an image in HWC uint8 format, with pixel values in [0, 255].
        """
        img = torch.tensor(image).unsqueeze(0).float()
        img = img.permute(0, 3, 1, 2)
        img = torch.nn.functional.pad(img, (0, 1024-img.shape[3], 0, 1024-img.shape[2]), value=0)
        img_norm = (img/256-torch.tensor(self.cellvit_mean).view(3, 1, 1)/torch.tensor(self.cellvit_std).view(3, 1, 1))
        with torch.no_grad():
            predictions = self.cellvit(img_norm)
            predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)  # shape: (batch_size, 2, H, W)
            predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)  # shape: (batch_size, num_nuclei_classes, H, W)
            (_, instance_types,) = self.cellvit.calculate_instance_map(predictions)
        return instance_types
