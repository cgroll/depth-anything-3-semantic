# %%
# Import required libraries
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# %%
# Setup paths and parameters
VIDEO_PATH = Path("data/inputs/videos/short.mp4")
OUTPUT_PATH = Path("data/outputs/videos/yolo_segmented.mp4")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set to None to process all frames, or specify a number (e.g., 5) to limit
NUM_FRAMES = None  # Set to None for all frames

print(f"Video path: {VIDEO_PATH}")
print(f"Output path: {OUTPUT_PATH}")
if NUM_FRAMES is None:
    print("Processing ALL frames in the video")
else:
    print(f"Processing first {NUM_FRAMES} frames")

# %%
# Load YOLO instance segmentation model
# Using YOLOv8 segmentation model (medium size)
model = YOLO('yolov8m-seg.pt')
print("Model loaded successfully")

# %%
# Load video and extract frames
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise ValueError(f"Could not open video: {VIDEO_PATH}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} @ {fps} FPS")
print(f"Total frames in video: {total_frames}")

frames = []
frame_count = 0
max_frames = NUM_FRAMES if NUM_FRAMES is not None else total_frames

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print(f"Reached end of video at frame {frame_count}")
        break
    frames.append(frame)
    frame_count += 1

    # Progress indicator for large videos
    if NUM_FRAMES is None and frame_count % 100 == 0:
        print(f"Loaded {frame_count}/{total_frames} frames...")

cap.release()
print(f"Extracted {len(frames)} frames")

# %%
# Run instance segmentation on frames
results_list = []
for i, frame in enumerate(frames):
    print(f"Processing frame {i+1}/{len(frames)}...")
    results = model(frame, verbose=False)
    results_list.append(results[0])

print("Segmentation complete")

# %%
# Create visualization with highlighted masks
annotated_frames = []

for i, (frame, results) in enumerate(zip(frames, results_list)):
    # Create a copy of the frame
    annotated_frame = frame.copy()

    # Check if masks exist
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()

        # Generate random colors for each instance
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

        # Apply each mask with a different color
        for j, (mask, box, color) in enumerate(zip(masks, boxes, colors)):
            # Resize mask to frame size
            mask_resized = cv2.resize(mask, (width, height))
            mask_bool = mask_resized > 0.5

            # Create colored overlay
            overlay = annotated_frame.copy()
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + color * 0.5
            annotated_frame = overlay.astype(np.uint8)

            # Draw bounding box and label
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{model.names[int(cls)]}: {conf:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.tolist(), 2)
            cv2.putText(annotated_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        print(f"Frame {i+1}: Detected {len(masks)} objects")
    else:
        print(f"Frame {i+1}: No objects detected")

    annotated_frames.append(annotated_frame)

# %%
# Save annotated frames as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

for frame in annotated_frames:
    out.write(frame)

out.release()
print(f"Video saved to: {OUTPUT_PATH}")

# %%
