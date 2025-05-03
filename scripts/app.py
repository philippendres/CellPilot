import torch
#from cellpilot.inference.inference import Inference
from cellpilot.inference.app_tools import App
import gradio as gr
from gradio_image_prompter import ImagePrompter
import argparse

def load_image(app, image):
    return app.load_image(image)

def download_image(app):
    return app.download_image()

def download_mask(app):
    return app.download_mask()

def move_left(app, amount):
    return app.move_left(amount)

def move_right(app, amount):
    return app.move_right(amount)

def move_up(app, amount):
    return app.move_up(amount)

def move_down(app, amount):
    return app.move_down(amount)

def zoom(app, zoom_bar, image):
    return app.zoom(zoom_bar, image)

def segment_automatically_app(app, image):
    return app.segment_automatically_app()

def add_mask(app, image):
    return app.add_mask(image)

def remove_mask(app, image):
    return app.remove_mask(image)




parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", type=str, default="/vol/data/models/")
parser.add_argument("--model_name", type=str, default="model-ap0xl4l1:v19")
parser.add_argument("--cellvit_model", type=str, default="CellViT-256-x40.pth")


args = parser.parse_args()

inference_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dir": args.model_dir,
    "model_name": args.model_name,
}
app_config = {
    
}
# 4.36.1
config = {
    "inference_config": inference_config,
    "app_config": app_config,
}




with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as demo: 
    app = gr.State(App(config))
    with gr.Row():
        image_size = 512
        with gr.Column(min_width=image_size):
            image = ImagePrompter(label="Image", height=image_size, width= image_size)
            image.upload(load_image, inputs=[app, image], outputs=[image])
        with gr.Column(min_width=0):
            gr.Markdown("""
                # CellPilot 
                This demo allows you to segment cells in microscopy images.
                To start, upload an image and click on "Auto Segment" to segment the image automatically.
                You can then add or remove individual cells by drawing a bounding box or clicking on the cell.    
            """)
            with gr.Tab(label="Segment") as segment_tab:
                # gr.Markdown("""
                # You can segment the image automatically ("Auto segment") and add ("Add Cell") or remove ("Remove Cell") individual cells.
                # """)
                auto_segment_btn = gr.Button("Auto Segment")
                # gr.Markdown("""
                # "Add Cell" and "Remove Cell" require you to draw a bounding box or click on the cell you want to add or remove.
                # """)
                add_mask_btn = gr.Button("Add Cell")         
                remove_mask_btn = gr.Button("Remove Cell")
                auto_segment_btn.click(segment_automatically_app, inputs=[app], outputs=[image])
                add_mask_btn.click(add_mask, inputs=[app,image], outputs=[image])
                remove_mask_btn.click(remove_mask, inputs=[app,image], outputs=[image])
                # Could also allow to refine the segmentation of a single cell 
                # with gr.Column(visible=True) as initial_buttons:
                # start_refine_mask_btn = gr.Button("Refine Mask")
                # with gr.Column(visible=False) as refine_buttons:
                #     refine_mask_btn = gr.Button("Refine")
                #     refine_mask_btn.click(app.refine_mask, inputs=[image], outputs=[image])
                #     finish_mask_btn = gr.Button("Finish Mask")  
                # start_refine_mask_btn.click(app.start_refine_mask, inputs=[image], outputs=[image, refine_buttons, initial_buttons])
                # finish_mask_btn.click(app.finish_mask, inputs=[], outputs=[image, initial_buttons, refine_buttons])
            with gr.Tab(label="Navigation") as navigation_tab:
                # gr.Markdown("""
                # You can move the image in any direction by clicking the arrows or zoom in and out using the slider.
                # """)
                amount = gr.Number(value=100, label="Step Size", visible = False)
                #amount = 100
                with gr.Column():
                    # Top button
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            gr.Column()
                        with gr.Column(scale=1, min_width=100):
                            up_button = gr.Button(value="\U0001F815")
                        with gr.Column(scale=1, min_width=100):
                            gr.Column()
                    # Middle row
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            left_button = gr.Button(value="\U0001F814")
                        with gr.Column(scale=1, min_width=100):
                            gr.Column()
                        with gr.Column(scale=1, min_width=100):
                            right_button = gr.Button(value="\U0001F816")
                    # Bottom button
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=100):
                            gr.Column()
                        with gr.Column(scale=1, min_width=100):
                            down_button = gr.Button(value="\U0001F817")
                        with gr.Column(scale=1, min_width=100):
                            gr.Column()
                    zoom_bar = gr.Slider(minimum=1, maximum=5, step=1, label="Zoom Factor", value=1)
                    left_button.click(move_left, inputs=[app,amount], outputs=[image])
                    up_button.click(move_up, inputs=[app,amount], outputs=[image])
                    down_button.click(move_down, inputs=[app,amount], outputs=[image])
                    right_button.click(move_right, inputs=[app,amount], outputs=[image])
                    zoom_bar.release(zoom, inputs=[app,zoom_bar, image], outputs=[image])
            with gr.Tab(label="Download") as save_tab:
                # gr.Markdown("""
                # You can download the current state of the image and the mask by clicking on "Download".
                # """)
                download_image_btn = gr.Button("Download Image")
                download_image_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_image_btn_hidden")
                download_image_btn.click(fn=download_image, inputs=[app], outputs=[download_image_btn_hidden]).then(fn=None, inputs=None, outputs=None, js="() => document.querySelector('#download_image_btn_hidden').click()")
                download_mask_btn = gr.Button("Download Mask")
                download_mask_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_mask_btn_hidden")
                download_mask_btn.click(fn=download_mask, inputs=[app], outputs=[download_mask_btn_hidden]).then(fn=None, inputs=None, outputs=None, js="() => document.querySelector('#download_mask_btn_hidden').click()")
                #download_image_btn = gr.DownloadButton("Download Image")
                #download_mask_btn = gr.DownloadButton("Download Mask")
                #download_image_btn.click(download_image, inputs=[app], outputs=[download_image_btn])
                #download_mask_btn.click(download_mask, inputs=[app], outputs=[download_mask_btn])
demo.launch(share=True)

