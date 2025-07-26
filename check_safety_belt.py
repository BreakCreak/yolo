# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.plots import Annotator
from utils.augmentations import letterbox
from utils.torch_utils import select_device

def load_model(weights, device):
    """Load YOLOv5 model"""
    model = attempt_load(weights, device=device)
    return model

def detect_safety_belt(model, img, device, imgsz=640):
    """Detect safety belt in image"""
    # Prepare image
    orig_img = img.copy()
    img = letterbox(img, imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0-255 to 0.0-1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    
    # Inference
    pred = model(img, augment=False, visualize=False)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=[3], agnostic=False)
    
    # Process detections
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to original image size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], orig_img.shape).round()
        return det.cpu().numpy()
    return []

def load_pose_model(weights, device):
    """Load YOLOv5 pose estimation model"""
    model = attempt_load(weights, device=device)
    return model

def detect_pose(model, img, device, imgsz=640):
    """Detect human pose in image"""
    # Prepare image
    orig_img = img.copy()
    img = letterbox(img, imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0-255 to 0.0-1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    
    # Inference
    pred = model(img, augment=False, visualize=False)[0]
    
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=0, agnostic=False, kpt_shape=(17, 3))
    
    # Process detections
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to original image size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], orig_img.shape).round()
        # Extract keypoints (assuming 17 keypoints in COCO format)
        if det.shape[1] > 6:  # Check if keypoints are present
            kpts = det[:, 6:]  # Extract keypoints part
            return det.cpu().numpy(), kpts.cpu().numpy()
    return [], []

def is_belt_worn_correctly(belt_boxes, keypoints):
    """
    Check if safety belt is worn correctly.
    Belt should NOT be under both knees.
    Keypoints format (COCO): [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
                              left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
                              left_knee, right_knee, left_ankle, right_ankle]
    """
    for kpt in keypoints:
        # Reshape keypoints: each keypoint has (x, y, conf)
        kpt = kpt.reshape(-1, 3)
        
        # Get knee positions (index 13 and 14)
        left_knee = kpt[13]
        right_knee = kpt[14]
        
        # Only proceed if both knees are detected with confidence
        if left_knee[2] > 0.5 and right_knee[2] > 0.5:
            # Get the lower knee y-coordinate (higher y value)
            lower_knee_y = max(left_knee[1], right_knee[1])
            
            # Check if any belt is below the lower knee
            for belt in belt_boxes:
                # Belt format: [x1, y1, x2, y2, conf, class]
                belt_top_y = min(belt[1], belt[3])
                
                # If belt top is below the lower knee, it's incorrectly worn
                if belt_top_y > lower_knee_y:
                    return False  # Belt is below knees - incorrectly worn
    return True  # Belt is correctly worn or no issues detected

def process_image(safety_belt_model, pose_model, img_path, device):
    """Process image to check safety belt wearing status"""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    
    # Detect safety belts
    belt_detections = detect_safety_belt(safety_belt_model, img, device)
    
    if len(belt_detections) == 0:
        print("No safety belts detected in the image")
        return
    
    # Detect human poses
    pose_detections, keypoints = detect_pose(pose_model, img, device)
    
    if len(pose_detections) == 0:
        print("No people detected in the image")
        return
    
    # Check if belts are worn correctly
    is_correct = is_belt_worn_correctly(belt_detections, keypoints)
    
    # Visualization
    annotator = Annotator(img, line_width=2, example=str(['badge', 'offground', 'ground', 'safebelt']))
    
    # Draw safety belt detections
    for *box, conf, cls in belt_detections:
        annotator.box_label(box, f'Safebelt {conf:.2f}', color=(0, 255, 0))
    
    # Draw pose detections
    for i, kpt in enumerate(keypoints):
        kpt = kpt.reshape(-1, 3)
        annotator.kpts(kpt, shape=img.shape[:2])
    
    # Add result text
    status = "CORRECT" if is_correct else "INCORRECT"
    color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.putText(img, f'Belt Status: {status}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Save result
    output_path = Path(img_path).parent / f"result_{Path(img_path).name}"
    cv2.imwrite(str(output_path), img)
    print(f"Result saved to: {output_path}")
    print(f"Safety belt wearing status: {status}")

def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--safety-belt-weights', type=str, default='safety_belt.pt', help='safety belt detection model path')
    parser.add_argument('--pose-weights', type=str, required=True, help='pose estimation model path')
    parser.add_argument('--source', type=str, required=True, help='image path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def main(opt):
    """Main function"""
    device = select_device(opt.device)
    
    # Load models
    safety_belt_model = load_model(opt.safety_belt_weights, device)
    pose_model = load_model(opt.pose_weights, device)
    
    # Set model to evaluation mode
    safety_belt_model.eval()
    pose_model.eval()
    
    # Process image
    process_image(safety_belt_model, pose_model, opt.source, device)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)