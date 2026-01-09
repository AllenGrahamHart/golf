"""
HMR 2.0 inference wrapper for golf swing analysis.

Usage:
    python -m src.inference --image input/swing.jpg --output output/
    python -m src.inference --image input/swing.jpg --output output/ --save_mesh
    python -m src.inference --video input/swing.mp4 --output output/
"""

import argparse
import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_detector():
    """Load YOLOv8 detector (lightweight alternative to ViTDet)."""
    from ultralytics import YOLO

    # YOLOv8n is the nano model - very lightweight (~6MB)
    # Use 'yolov8s.pt' for better accuracy or 'yolov8n.pt' for speed
    detector = YOLO("yolov8n.pt")
    return detector


def load_hmr2_model(device: str = "cpu"):
    """Load HMR 2.0 model separately from detector."""
    from hmr2.models import load_hmr2, download_models

    # Download model checkpoints if needed
    download_models()

    # Load HMR 2.0 model
    model, model_cfg = load_hmr2()
    model = model.to(device)
    model.eval()

    return model, model_cfg


def detect_humans(image: np.ndarray, detector, threshold: float = 0.5):
    """Detect humans in image using YOLOv8 and return bounding boxes."""
    # Run YOLO detection
    results = detector(image, classes=[0], verbose=False)  # class 0 = person

    boxes = []
    for result in results:
        for box in result.boxes:
            if box.conf.item() > threshold:
                # Get xyxy format bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(xyxy)

    return np.array(boxes) if boxes else np.array([]).reshape(0, 4)


def run_hmr2(image: np.ndarray, boxes: np.ndarray, model, model_cfg, device: str = "cpu"):
    """Run HMR 2.0 on detected humans."""
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    # Create dataset from detections
    dataset = ViTDetDataset(model_cfg, image, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(boxes), shuffle=False, num_workers=0)

    all_results = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out = model(batch)

        # Extract predictions
        pred_cam = out["pred_cam"]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        # Convert camera to translation
        pred_cam_t_full = cam_to_full_translation(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        )

        batch_size = pred_cam.shape[0]
        for i in range(batch_size):
            result = {
                "vertices": out["pred_vertices"][i].cpu().numpy(),
                "cam_t": pred_cam_t_full[i].cpu().numpy(),
                "smpl_params": {
                    "body_pose": out["pred_smpl_params"]["body_pose"][i].cpu().numpy(),
                    "betas": out["pred_smpl_params"]["betas"][i].cpu().numpy(),
                    "global_orient": out["pred_smpl_params"]["global_orient"][i].cpu().numpy(),
                },
                "box": boxes[i],
            }
            all_results.append(result)

    return all_results


def cam_to_full_translation(pred_cam, box_center, box_size, img_size, focal_length):
    """Convert predicted camera parameters to full translation."""
    pred_cam_t = torch.stack([
        pred_cam[:, 1],
        pred_cam[:, 2],
        2 * focal_length / (pred_cam[:, 0] * box_size + 1e-9)
    ], dim=-1)

    pred_cam_t[:, :2] += (box_center - img_size / 2) * pred_cam_t[:, 2:] / focal_length

    return pred_cam_t


def render_results(image: np.ndarray, results: list, model_cfg, output_dir: Path,
                   image_name: str, save_mesh: bool = False, side_view: bool = True,
                   smpl_faces: np.ndarray = None):
    """Render mesh overlay on image."""
    from hmr2.utils.renderer import Renderer

    img_h, img_w = image.shape[:2]
    focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(img_h, img_w)

    # Initialize renderer with pre-loaded SMPL faces
    renderer = Renderer(model_cfg, faces=smpl_faces)
    faces = smpl_faces

    # Render all detected people
    all_verts = []
    all_cam_t = []

    for result in results:
        all_verts.append(result["vertices"])
        all_cam_t.append(result["cam_t"])

    # Stack for batch rendering
    all_verts = np.stack(all_verts)
    all_cam_t = np.stack(all_cam_t)

    # Render front view
    misc_args = {
        "mesh_base_color": (0.8, 0.3, 0.3),
        "scene_bg_color": (1, 1, 1),
    }

    cam_view = renderer.render_rgba_multiple(
        all_verts, cam_t=all_cam_t,
        render_res=[img_w, img_h],
        focal_length=focal_length,
        **misc_args
    )

    # Composite on original image
    input_img = image.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)

    output_img = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    output_img = (output_img * 255).astype(np.uint8)

    # Save front view
    front_path = output_dir / f"{image_name}_render.png"
    Image.fromarray(output_img).save(front_path)
    print(f"Saved front view: {front_path}")

    # Render side view
    if side_view:
        side_verts = all_verts.copy()
        # Rotate 90 degrees around Y axis
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        side_verts = np.einsum("ij,bnj->bni", rotation, side_verts)

        side_view_img = renderer.render_rgba_multiple(
            side_verts, cam_t=all_cam_t,
            render_res=[img_w, img_h],
            focal_length=focal_length,
            **misc_args
        )
        side_output = (side_view_img[:, :, :3] * 255).astype(np.uint8)
        side_path = output_dir / f"{image_name}_side.png"
        Image.fromarray(side_output).save(side_path)
        print(f"Saved side view: {side_path}")

    # Save meshes as OBJ
    if save_mesh:
        import trimesh
        for i, result in enumerate(results):
            mesh = trimesh.Trimesh(vertices=result["vertices"], faces=faces)
            mesh_path = output_dir / f"{image_name}_person{i}.obj"
            mesh.export(mesh_path)
            print(f"Saved mesh: {mesh_path}")

    return output_img


def save_params(results: list, output_dir: Path, image_name: str):
    """Save SMPL parameters to npz file."""
    params = {
        "body_pose": np.stack([r["smpl_params"]["body_pose"] for r in results]),
        "betas": np.stack([r["smpl_params"]["betas"] for r in results]),
        "global_orient": np.stack([r["smpl_params"]["global_orient"] for r in results]),
        "cam_t": np.stack([r["cam_t"] for r in results]),
        "boxes": np.stack([r["box"] for r in results]),
    }

    param_path = output_dir / f"{image_name}_params.npz"
    np.savez(param_path, **params)
    print(f"Saved SMPL parameters: {param_path}")

    return param_path


# =============================================================================
# Video Processing Functions
# =============================================================================


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Extract all frames from a video file.

    Returns:
        frames: List of BGR images (numpy arrays)
        fps: Original video framerate
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Extracted {len(frames)} frames at {fps:.2f} fps")
    return frames, fps


def frames_to_video(frames: List[np.ndarray], output_path: str, fps: float):
    """Reassemble frames into a video file."""
    if not frames:
        raise ValueError("No frames to write")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        # Convert RGB to BGR if needed (our renders are RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)

    writer.release()
    print(f"Saved video: {output_path}")


def save_video_params(all_results: List[List[dict]], output_dir: Path, video_name: str):
    """Save SMPL parameters for all frames to a single npz file.

    Shape convention: (T, N, ...) where T=frames, N=people per frame
    For golf, N=1 typically.
    """
    # Assuming single person per frame for golf
    # Each frame's results is a list with one dict
    params = {
        "body_pose": np.stack([r[0]["smpl_params"]["body_pose"] for r in all_results]),
        "betas": np.stack([r[0]["smpl_params"]["betas"] for r in all_results]),
        "global_orient": np.stack([r[0]["smpl_params"]["global_orient"] for r in all_results]),
        "cam_t": np.stack([r[0]["cam_t"] for r in all_results]),
        "boxes": np.stack([r[0]["box"] for r in all_results]),
    }

    param_path = output_dir / f"{video_name}_params.npz"
    np.savez(param_path, **params)
    print(f"Saved video SMPL parameters: {param_path} (shape: {params['body_pose'].shape[0]} frames)")

    return param_path


def process_video(video_path: str, output_dir: str, device: str = "cpu"):
    """Process a video through the HMR 2.0 pipeline (parameters only, no rendering).

    Pipeline:
        1. Extract all frames
        2. Run detection on all frames
        3. Run HMR2 on all frames
        4. Save parameters
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem

    print(f"Processing video: {video_path}")

    # Step 1: Extract frames
    print("\n[1/4] Extracting frames...")
    frames, fps = extract_frames(str(video_path))

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Step 2: Detection on all frames
    print("\n[2/4] Running detection on all frames...")
    detector = load_detector()

    all_boxes = []
    for frame in tqdm(frames, desc="Detecting"):
        boxes = detect_humans(frame, detector)
        if len(boxes) == 0:
            print(f"Warning: No person detected in frame {len(all_boxes)}")
            # Use previous box if available, otherwise raise error
            if all_boxes:
                boxes = all_boxes[-1]
            else:
                raise ValueError("No person detected in first frame")
        all_boxes.append(boxes)

    # Free detector memory
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 3: HMR2 inference on all frames
    print("\n[3/4] Running HMR 2.0 on all frames...")
    model, model_cfg = load_hmr2_model(device)

    all_results = []
    for i, (frame, boxes) in enumerate(tqdm(zip(frames, all_boxes), total=len(frames), desc="HMR2")):
        results = run_hmr2(frame, boxes, model, model_cfg, device)
        all_results.append(results)

    # Free HMR2 model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 4: Save parameters
    print("\n[4/4] Saving parameters...")
    save_video_params(all_results, output_dir, video_name)

    print("\nDone!")
    return all_results


def process_image(image_path: str, output_dir: str, save_mesh: bool = False,
                  side_view: bool = True, device: str = "cpu"):
    """Process a single image through the HMR 2.0 pipeline."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = image_path.stem

    print(f"Processing: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Stage 1: Detection with lightweight YOLOv8
    print("Loading YOLOv8 detector...")
    detector = load_detector()

    print("Detecting humans...")
    boxes = detect_humans(image, detector)

    # Free detector memory before loading HMR2.0
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(boxes) == 0:
        print("No humans detected in the image.")
        return None

    print(f"Found {len(boxes)} person(s)")

    # Stage 2: HMR2.0 inference (loaded after detector is freed)
    print("Loading HMR 2.0 model...")
    model, model_cfg = load_hmr2_model(device)

    print("Running HMR 2.0...")
    results = run_hmr2(image, boxes, model, model_cfg, device)

    # Extract SMPL faces before freeing model
    smpl_faces = model.smpl.faces.copy()

    # Free HMR2.0 model memory before rendering
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Rendering results...")
    render_results(image, results, model_cfg, output_dir, image_name, save_mesh, side_view, smpl_faces)

    print("Saving parameters...")
    save_params(results, output_dir, image_name)

    print("Done!")
    return results


def main():
    parser = argparse.ArgumentParser(description="HMR 2.0 inference for golf swing analysis")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video (extracts SMPL params only)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--save_mesh", action="store_true", help="Save mesh as OBJ file (image only)")
    parser.add_argument("--no_side_view", action="store_true", help="Skip side view rendering (image only)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("Either --image or --video must be provided")

    if args.image and args.video:
        parser.error("Cannot specify both --image and --video")

    if args.image:
        process_image(
            image_path=args.image,
            output_dir=args.output,
            save_mesh=args.save_mesh,
            side_view=not args.no_side_view,
            device=args.device,
        )
    else:
        process_video(
            video_path=args.video,
            output_dir=args.output,
            device=args.device,
        )


if __name__ == "__main__":
    main()
