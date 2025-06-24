from pathlib import Path
from typing import Any, Dict
import gradio as gr
import cv2
import numpy as np
import random
import tempfile
import os
import io
import base64
from PIL import Image
import torch
from torch._tensor import Tensor
from torchvision.io import read_video
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipe import (
    MultiModalAuthPipeline,
    ImagePreprocessor,
    AudioPreprocessor,
    ImagePreprocessor,
    AdaFace,
    ReDimNet,
    ClassifierHead,
)

from synthweave.utils.fusion import get_fusion

import json

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

# print the current working directory
print(f"Current working directory: {Path.cwd()}")

models_dir = Path("../../../models")
print(f"Models directory: {models_dir}")

sample_dir = Path("../notebooks/samples")
print(f"Sample directory: {sample_dir}")
# config
args = json.loads((models_dir / "CAFF" / "args.json").read_text())
args = dotdict(args)

# weights
weights_path = models_dir / "CAFF" / "detection_module.ckpt"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")

preprocessors = {
    "video": ImagePreprocessor(
        window_len=4,
        step=1,
        estimate_quality=False,
        models_dir=models_dir,
        quality_model_type="ir50",
        device=device,
    ),
    "audio": AudioPreprocessor(window_len=4, step=1, use_vad=False, device=device),
}

models = {
    "video": AdaFace(
        path=models_dir,
        model_type="ir50",
        device=device,
    ),
    "audio": ReDimNet(),
}

fusion = get_fusion(
    fusion_name=args.fusion,
    output_dim=args.emb_dim,
    modality_keys=["video", "audio"],
    input_dims={"video": 512, "audio": 192},
    out_proj_dim=args.proj_dim,
)

detection_head = ClassifierHead(input_dim=args.emb_dim, num_classes=1)

pipe = MultiModalAuthPipeline(
    processors=preprocessors,
    models=models,
    fusion=fusion,
    detection_head=detection_head,
    freeze_backbone=True,
    iil_mode=args.iil_mode,
)

state_dict = torch.load(weights_path, map_location="cpu")["state_dict"]
state_dict = {k.replace("pipeline.", ""): v for k, v in state_dict.items()}
pipe.load_state_dict(state_dict, strict=False)

pipe = pipe.to(device)
pipe.eval()


def get_results_css_styles():
    """Get CSS styles for the results HTML display"""
    return """
    .below-threshold { background: rgba(255, 0, 0, 0.33) !important; font-weight: bold; }
    .above-threshold { background: rgba(0, 255, 0, 0.33) !important; font-weight: bold; }
    """


def get_metric_colors():
    """Get color configurations for different metrics"""
    return {
        "audio": "#FFD700",  # Gold
        "face": "#98FB98",  # Light Green
        "deepfake": "#FF6B6B",  # Light Red
    }


def model_inference(ref, sample):
    print("Running model inference...")
    with torch.no_grad():
        sample_out = pipe(sample)  # Sample for pipeline inference
        ref_out = pipe(ref)  # Simulates database
        print("Original len: ", ref_out['org_len'])
        print("Processed len: ", ref_out['valid_len'])

    probs = torch.sigmoid(sample_out["logits"]).cpu()
    prob_per_clip = probs.mean()
    preds_per_clip = (
        prob_per_clip >= deepfake_threshold.value  
    ).long() 

    print("Pred:", "Bonafide" if preds_per_clip.item() == 0 else "DeepFake")

    with torch.no_grad():
        sim = pipe.verify(
            {
                "video": sample_out["video"].cpu(),
                "audio": sample_out["audio"].cpu(),
                "video_ref": ref_out["video"].cpu(), 
                "audio_ref": ref_out["audio"].cpu(),
            }
        )

    # NOTE: Select aggregation methods across windows
    # vid_sim = sim["video"].max(dim=1).values                # Max similarity to any reference
    vid_sim = torch.tensor(sim["video"]).mean(dim=1)  # Mean similarity to reference

    vid_th = face_threshold.value # NOTE: set threshold for video similarity

    passed = vid_sim > vid_th
    print("Pass status:", passed.tolist())
    print("Face verified:", passed.all().item())

    # NOTE: Select aggregation methods across windows
    # aud_sim = sim["audio"].max(dim=1).values                # Max similarity to any reference
    aud_sim = torch.tensor(sim["audio"]).mean(dim=1)  # Mean similarity to reference

    aud_th = audio_threshold.value  # NOTE: set threshold for audio similarity 0 - 1

    passed = aud_sim > aud_th
    print("Pass status:", passed.tolist())
    print(
        "Voice verified:", passed.all().item()
    )  # True if all windows passed the threshold

    return [aud_sim.mean().item(), vid_sim.mean().item(), prob_per_clip.mean().item()]


def preprocess_video(video_file) -> tuple[Tensor, Tensor, Dict[str, Any]]:
    # load video using torchvision
    print(f"Loading video from: {video_file}")
    video, audio, info = read_video(video_file, pts_unit="sec")
    return video, audio, info


def process_video(ref_video, sample_video):
    """Process uploaded or recorded video"""
    if ref_video is None:
        return "No video provided", None, ""
    if sample_video is None:
        return "No sample video provided", None, ""

    ref_vid, ref_audio, ref_info = preprocess_video(ref_video)
    print(f"Reference video shape: {ref_vid.shape}, Audio shape: {ref_audio.shape}, Info: {ref_info}")
    sample_vid, sample_audio, sample_info = preprocess_video(sample_video)
    print(f"Sample video shape: {sample_vid.shape}, Audio shape: {sample_audio.shape}, Info: {sample_info}")

    ref = {
        "video": [ref_vid, ref_info["video_fps"]],
        "audio": [ref_audio, ref_info["audio_fps"]],
        "metadata": ref_info,
    }

    sample = {
        "video": [sample_vid, sample_info["video_fps"]],
        "audio": [sample_audio, sample_info["audio_fps"]],
        "metadata": sample_info,
    }

    return model_inference(ref, sample)


def record_video():
    """Handle webcam recording - this will be handled by Gradio's built-in video recording"""
    return "Video recorded! Click 'Process Video' to analyze."

def authenticate(ref_video, sample_video, audio_thresh, face_thresh):
    results = process_video(ref_video, sample_video)
    audio_sim, face_sim, deepfake = results
    color_updates = update_colors(
        audio_sim, face_sim, audio_thresh, face_thresh, deepfake
    )
    return [*color_updates, deepfake]


# Create Gradio interface
with gr.Blocks(
    title="MultimodalAuth Demo",
    theme=gr.themes.Ocean(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.blue,
        text_size=gr.themes.utils.sizes.text_lg,
    ),
    css=get_results_css_styles(),
) as app:
    with gr.Row():
        # Left Column - Logo and Title in HTML
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="display: flex; align-items: center; gap: 20px;">
                <h1 style="margin: 0; font-size: 4rem; font-weight: bold;">MultimodalAuth</h1>
            </div>
            """)

    with gr.Row():
        # Left Column - Input Controls
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):

                    gr.Markdown("## 📹 Reference Video")

                    ref_video = gr.Video(
                        sources=["upload", "webcam"],
                        label="Reference Video",
                        include_audio=True,
                        visible=True,
                        webcam_options=gr.WebcamOptions(mirror=False),
                        format="mp4"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("## 📹 Sample Video")

                    sample_video = gr.Video(
                        sources=["upload", "webcam"],
                        label="Sample Video",
                        include_audio=True,
                        visible=True,
                        webcam_options=gr.WebcamOptions(mirror=False),
                        format="mp4"
                    )

        # Right Column - Results
        with gr.Column(scale=1):
            # Nested columns for thresholds and settings
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_threshold = gr.Slider(
                            label="Select Audio Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.75,
                            min_width=100,
                        )
                        audio_feedback = gr.Textbox(
                            value="Audio threshold set to 0.75",
                            interactive=False,
                            show_label=False,
                            container=False,
                            visible=True,
                        )

                    with gr.Column(scale=1):
                        face_threshold = gr.Slider(
                            label="Select Face Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.75,
                            min_width=100,
                        )
                        face_feedback = gr.Textbox(
                            value="Face threshold set to 0.75",
                            interactive=False,
                            show_label=False,
                            container=False,
                            visible=True,
                        )

                    with gr.Column(scale=1):
                        deepfake_threshold = gr.Slider(
                            label="Select Deepfake Threshold",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.5,
                            min_width=100,
                        )
                        deepfake_feedback = gr.Textbox(
                            value="Deepfake threshold set to 0.5",
                            interactive=False,
                            show_label=False,
                            container=False,
                            visible=True,
                        )

            authenticate_btn = gr.Button("Authenticate", variant="primary", size="lg")

            audio_results = gr.Number(
                label="Audio Similarity",
                value=0.0,
                interactive=False,
                precision=2,
            )
            face_results = gr.Number(
                label="Face Similarity",
                value=0.0,
                interactive=False,
                precision=2
            )
            deepfake_results = gr.Number(
                label="Deepfake Probability", value=0.0, interactive=False, precision=2
            )

    with gr.Row():
        gr.Examples(
            examples=[
                [
                    f"{sample_dir}/john_real.mp4",
                    f"{sample_dir}/john_face_fake.mp4",
                    0.60,
                    0.60,
                    0.5,
                    0.66,
                    0.70,
                    0.83,
                ],
                [
                    f"{sample_dir}/john_real.mp4",
                    f"{sample_dir}/john_face_fake.mp4",
                    0.60,
                    0.14,
                    0.5,
                    0.14,
                    0.52,
                    0.55,
                ],
                [
                    f"{sample_dir}/john_real.mp4",
                    f"{sample_dir}/john_face_fake.mp4",
                    0.22,
                    0.60,
                    0.70,
                    0.66,
                    0.14,
                    0.39,
                ],
            ],
            inputs=[
                ref_video,
                sample_video,
                audio_threshold,
                face_threshold,
                deepfake_threshold,
                audio_results,
                face_results,
                deepfake_results,
            ],
            outputs=[
                audio_results,
                face_results,
                deepfake_results,
            ],
        )

    def update_ref_video_inputs(choice):
        return [
            gr.update(visible=(choice == "Upload Video")),
            gr.update(visible=(choice == "Record with Webcam")),
        ]

    def update_sample_video_inputs(choice):
        return [
            gr.update(visible=(choice == "Upload Video")),
            gr.update(visible=(choice == "Record with Webcam")),
        ]

    def video_handler(video_input):
        """Handle sample video input from either upload or webcam"""
        if video_input is None:
            return None
        return gr.update(value=video_input, visible=True)

    def handle_video_upload(video_file):
        """Handle video upload and show preview"""
        if video_file is not None:
            return gr.update(value=video_file, visible=True)
        else:
            return gr.update(value=None, visible=False)

    def update_audio_feedback(audio_sim, face_sim, deepfake_prob, audio_thresh, face_thresh, deepfake_thresh):
        color_updates = update_colors(
            audio_sim,
            face_sim,
            deepfake_prob,
            audio_thresh,
            face_thresh,
            deepfake_thresh,
        )
        return [f"Audio threshold set to {audio_thresh:.2f}", *color_updates]

    def update_face_feedback(audio_sim, face_sim, deepfake_prob, audio_thresh, face_thresh, deepfake_thresh):
        color_updates = update_colors(audio_sim, face_sim, deepfake_prob, audio_thresh, face_thresh, deepfake_thresh)
        return [f"Face detection threshold set to {face_thresh:.2f}", *color_updates]

    def update_deepfake_feedback(
        audio_sim, face_sim, deepfake_prob, audio_thresh, face_thresh, deepfake_thresh
    ):
        color_updates = update_colors(
            audio_sim,
            face_sim,
            deepfake_prob,
            audio_thresh,
            face_thresh,
            deepfake_thresh,
        )
        return [f"Deepfake threshold set to {deepfake_thresh:.2f}", *color_updates]

    audio_threshold.change(
        fn=update_audio_feedback,
        inputs=[
            audio_results,
            face_results,
            deepfake_results,
            audio_threshold,
            face_threshold,
            deepfake_threshold
        ],
        outputs=[
            audio_feedback,
            audio_results,
            face_results,
            deepfake_results
        ]
    )

    face_threshold.change(
        fn=update_face_feedback,
        inputs=[
            audio_results,
            face_results,
            deepfake_results,
            audio_threshold,
            face_threshold,
            deepfake_threshold
        ],
        outputs=[
            face_feedback,
            audio_results,
            face_results,
            deepfake_results
        ]
    )

    deepfake_threshold.change(
        fn=update_deepfake_feedback,
        inputs=[
            audio_results,
            face_results,
            deepfake_results,
            audio_threshold,
            face_threshold,
            deepfake_threshold,
        ],
        outputs=[deepfake_feedback, audio_results, face_results, deepfake_results],
    )

    authenticate_btn.click(
        fn=authenticate,
        inputs=[ref_video, sample_video, audio_threshold, face_threshold],
        outputs=[audio_results, face_results, deepfake_results],
    )

    def update_colors(audio_sim, face_sim, deepfake, audio_threshold, face_threshold, deepfake_threshold):
        audio_class = (
            "above-threshold"
            if float(audio_sim) >= float(audio_threshold)
            else "below-threshold"
        )
        face_class = (
            "above-threshold"
            if float(face_sim) >= float(face_threshold)
            else "below-threshold"
        )
        deepfake_class = (
            "above-threshold"
            if float(deepfake) <= float(deepfake_threshold)
            else "below-threshold"
        )

        return [
            gr.update(value=audio_sim, elem_classes=audio_class),
            gr.update(value=face_sim, elem_classes=face_class),
            gr.update(value=deepfake, elem_classes=deepfake_class),
        ]


# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        allowed_paths=[sample_dir],
    )
