import numpy as np
from torchvision.transforms.functional import resize, to_pil_image, InterpolationMode
from segment_anything.utils.transforms import ResizeLongestSide
import cv2
from .inference import Inference
import gradio as gr
from datetime import datetime
from PIL import Image

class App(Inference):
    def __init__(self, config):
        self.config = config
        self.inference_config = config["inference_config"]
        super().__init__(self.inference_config)
        self.zoom_factor = 1
        self.middle = (512,512)
        self.upper = int(512 + self.zoom_factor*self.middle[0])
        self.left = int(512 + self.zoom_factor*self.middle[1])
        self.current_refinement = None
        self.prompts = []
        self.transform = ResizeLongestSide(1024)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.saved_image_path = f"/tmp/saved_image_{self.timestamp}.png"
        self.saved_mask_path = f"/tmp/saved_mask_{self.timestamp}.png"

    def load_image(self, image, interpolation_mode=InterpolationMode.BILINEAR):
        img = image["image"]    
        self.orig_h, self.orig_w = img.shape[:2]
        self.orig_masks = np.zeros((self.orig_h, self.orig_w), dtype=np.int16)
        self.orig_img = img
        self.h, self.w = ResizeLongestSide.get_preprocess_shape(img.shape[0], img.shape[1], 1024)
        img = np.array(resize(to_pil_image(img), (self.h, self.w), interpolation_mode))
        self.predictor.set_image(img)
        self.img = img
        self.masks = np.zeros((self.h, self.w), dtype=np.uint8)
        self.zoom_factor = 1
        self.middle = (512,512)
        self.upper = int(512 + self.zoom_factor*self.middle[0])
        self.left = int(512 + self.zoom_factor*self.middle[1])
        self.current_refinement = None
        self.prompts = []
        self.transform = ResizeLongestSide(1024)
        img = np.pad(img, ((0,1024-self.h), (0,1024-self.w), (0,0)))
        self.current_image = np.pad(img, ((1024,1024), (1024,1024), (0,0)))
        self.current_masks = np.zeros((3072,3072), dtype=np.int16)
        self.masks = np.zeros((self.h, self.w))
        self.orig_masks = np.array(resize(to_pil_image(self.masks.astype(np.int16)), (int(self.orig_h), int(self.orig_w)), InterpolationMode.NEAREST))
        return {"image": img.astype(np.uint8)}

    def download_image(self):
        self.save_image()
        return self.saved_image_path
    
    def download_mask(self):
        self.save_mask()
        return self.saved_mask_path
    
    def save_image(self):
        img = self.display_current_image()["image"]
        img_pil = Image.fromarray(img)
        img_pil.save(self.saved_image_path)

    def save_mask(self):
        mask_pil = Image.fromarray(self.current_masks[self.upper:self.upper + 1024, self.left:self.left + 1024])
        mask_pil.save(self.saved_mask_path)

    def zoom(self, zoom_factor, image):
        points = image.get("points", [])
        if points != []:
            old_points = points
            points = points.copy()
            for i in range(len(points)):
                orig_points = ((self.left -1024 + old_points[i][0])/self.zoom_factor, (self.upper - 1024 + old_points[i][1])/self.zoom_factor)
                points[i][0] = int(orig_points[0] * zoom_factor)
                points[i][1] = int(orig_points[1] * zoom_factor)
        self.zoom_factor = zoom_factor
        img = np.array(resize(to_pil_image(self.orig_img.astype(np.uint8)), (int(zoom_factor * self.h), int(zoom_factor*self.w)), InterpolationMode.BILINEAR))
        self.current_image = np.pad(img, ((1024,1024), (1024,1024), (0,0)))
        self.zoom_masks()
        return self.display_current_image(points)
    
    def zoom_masks(self):
        self.current_masks = np.array(resize(to_pil_image(self.orig_masks.astype(np.int16)), (int(self.zoom_factor * self.h), int(self.zoom_factor*self.w)), InterpolationMode.NEAREST))
        self.current_masks = np.pad(self.current_masks, ((1024,1024), (1024,1024)))
        
    def color_mask(self, mask):
        if self.current_refinement is not None:
            mask = np.where(mask == self.current_refinement, -1, mask)
        mask = np.where(mask > 0, 1, mask)
        h, w = mask.shape[-2:]
        color = np.array([30, 144, 255]).astype(np.uint8)
        mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask = mask.astype(np.uint8)
        return mask

    def move_up(self, amount=10):
        self.middle = (self.middle[0] - amount/self.zoom_factor, self.middle[1])   
        return self.display_current_image()
    
    def move_down(self, amount=10):
        self.middle = (self.middle[0] + amount/self.zoom_factor, self.middle[1])
        return self.display_current_image()
    
    def move_left(self, amount=10):
        self.middle = (self.middle[0], self.middle[1] - amount/self.zoom_factor)
        return self.display_current_image()
    
    def move_right(self, amount=10):
        self.middle = (self.middle[0], self.middle[1] + amount/self.zoom_factor)
        return self.display_current_image()
    
    def display_current_image(self, points=[]):
        self.upper = int(512 + self.zoom_factor*self.middle[0])
        self.left = int(512 + self.zoom_factor*self.middle[1])
        img = self.current_image[self.upper:self.upper + 1024, self.left:self.left + 1024,:]
        mask = self.current_masks[self.upper:self.upper + 1024, self.left:self.left + 1024]
        color_mask = self.color_mask(mask)
        img =  np.where(color_mask > 0, 0.4 * img, img)
        img = img.astype(np.uint8)
        img = cv2.addWeighted(img, 1.0, color_mask, 0.6, 0.0)
        if points != []:
            for i in range(len(points)):
                points[i][0] = points[i][0] - self.upper + 1024
                points[i][1] = points[i][1] - self.left + 1024
        return {"image": img.astype(np.uint8),"points": points}


    def segment_automatically_app(self):
        masks, self.prompts = self.segment_automatically(self.img)
        self.masks = masks[:self.h, :self.w]
        self.orig_masks = np.array(resize(to_pil_image(self.masks.astype(np.int16)), (int(self.orig_h), int(self.orig_w)), InterpolationMode.NEAREST))
        self.zoom_masks()
        return self.display_current_image()
         
    def add_mask(self, input):
        prompts = input.get("points", [])
        if prompts == []:
            return self.display_current_image()
        new_prompt = self.new_prompt(prompts, [], [], [])
        self.prompts.append(new_prompt)
        new_mask = self.segment(new_prompt)
        self.masks = np.where(new_mask == 1, len(self.prompts), self.masks)
        self.orig_masks = np.array(resize(to_pil_image(self.masks.astype(np.int16)), (int(self.orig_h), int(self.orig_w)), InterpolationMode.NEAREST))
        self.zoom_masks()
        return self.display_current_image(points=[])
    
    def new_prompt(self, prompts, point_coords=[], point_labels=[], boxes=[]):
        for prompt in prompts:
            p0 = (prompt[0]+self.left-1024)/self.zoom_factor
            p1 = (prompt[1]+self.upper-1024)/self.zoom_factor
            if prompt[3] == 0:
                point_coords.append([p0, p1])
                point_labels.append(1)
            else:
                p2= (prompt[3]+self.left-1024)/self.zoom_factor
                p3 = (prompt[4]+self.upper-1024)/self.zoom_factor
                boxes = [p0, p1, p2, p3]     
        if point_coords == []:
            point_coords = None
            point_labels = None
        else:
            point_coords = np.array(point_coords)
            point_labels = np.array(point_labels)
        if boxes == []:
            boxes = None
        else:
            boxes = np.array(boxes)
        new_prompt = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "boxes": boxes
        }
        return new_prompt  

    def start_refine_mask(self, input):
        prompts = input.get("points", [])
        if prompts == []:
            return self.display_current_image()
        point = (int(input["points"][0][1]), int(input["points"][0][0]))
        value = self.current_masks[self.upper + point[0], self.left + point[1]]
        self.current_refinement = value
        return self.display_current_image(), gr.Column(visible=True), gr.Column(visible=False)

    def remove_mask(self, input):
        prompts = input.get("points", [])
        if prompts == []:
            return self.display_current_image()
        point = (int(input["points"][0][1]), int(input["points"][0][0]))
        value = self.current_masks[self.upper + point[0], self.left + point[1]]
        self.orig_masks = np.where(self.orig_masks == value, 0, self.orig_masks)
        self.current_masks = np.where(self.current_masks == value, 0, self.current_masks)
        self.masks = np.where(self.masks == value, 0, self.masks)
        return self.display_current_image()
    
    def refine_mask(self, input):
        prompts = input.get("points", [])
        if prompts == []:
            return self.display_current_image()
        prompts = input["points"]
        point_coords = self.prompts[self.current_refinement-1].get("point_coords", np.array([]))
        point_labels = self.prompts[self.current_refinement-1].get("point_labels", np.array([]))
        boxes = self.prompts[self.current_refinement-1].get("boxes", np.array([]))
        point_coords = point_coords.tolist() if point_coords is not None else []
        point_labels = point_labels.tolist() if point_labels is not None else []
        boxes = boxes.tolist() if boxes is not None else []
        new_prompt = self.new_prompt(prompts, point_coords, point_labels, boxes)
        self.prompts[self.current_refinement-1] = new_prompt
        new_mask = self.segment(new_prompt, self.img)
        self.masks = np.where(self.masks == self.current_refinement, 0, self.masks)
        self.masks = np.where(new_mask == 1, self.current_refinement, self.masks)
        self.orig_masks = np.array(resize(to_pil_image(self.masks.astype(np.int16)), (int(self.orig_h), int(self.orig_w)), InterpolationMode.NEAREST))
        self.zoom_masks()
        return self.display_current_image()
    
    def finish_mask(self):
        self.current_refinement = None
        return self.display_current_image(), gr.Column(visible=True), gr.Column(visible=False)
