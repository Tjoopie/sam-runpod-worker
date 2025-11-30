"""
RunPod Serverless Handler for SAM (Segment Anything Model)
Click-to-segment functionality for farm boundary detection
"""

import runpod
import torch
import numpy as np
import cv2
import base64
import io
import time
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Global model instance
sam_predictor = None

def load_model():
    """Load SAM model (cached after first call)"""
    global sam_predictor
    
    if sam_predictor is not None:
        return sam_predictor
    
    print("Loading SAM model...")
    start = time.time()
    
    model_type = "vit_b"  # Use vit_b for faster inference, vit_h for better quality
    checkpoint = "/app/sam_vit_b_01ec64.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    print(f"SAM model loaded in {time.time() - start:.2f}s on {device}")
    return sam_predictor


def base64_to_image(base64_string):
    """Convert base64 string to numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image.convert('RGB'))


def mask_to_polygon(mask, simplify_tolerance=2.0):
    """Convert binary mask to polygon coordinates"""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour
    epsilon = simplify_tolerance
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    coords = approx.squeeze().tolist()
    
    if len(coords) < 3:
        return None
    
    # Close polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    return {"type": "Polygon", "coordinates": [coords]}


def pixel_to_geo(pixel_x, pixel_y, bounds, image_size):
    """Convert pixel to geographic coordinates"""
    width, height = image_size
    lng = bounds['west'] + (pixel_x / width) * (bounds['east'] - bounds['west'])
    lat = bounds['north'] - (pixel_y / height) * (bounds['north'] - bounds['south'])
    return [lng, lat]


def transform_polygon_to_geo(polygon, bounds, image_size):
    """Transform polygon from pixel to geographic coordinates"""
    if polygon is None:
        return None
    
    coords = polygon['coordinates'][0]
    geo_coords = [pixel_to_geo(x, y, bounds, image_size) for x, y in coords]
    
    return {"type": "Polygon", "coordinates": [geo_coords]}


def handler(job):
    """
    RunPod serverless handler for SAM segmentation
    
    Input:
    {
        "input": {
            "image_base64": "...",
            "click_x": 320,  # pixel x coordinate
            "click_y": 240,  # pixel y coordinate
            "bounds": {"west": ..., "east": ..., "north": ..., "south": ...}
        }
    }
    
    Output:
    {
        "polygon": {...},  # GeoJSON polygon
        "confidence": 0.95,
        "processing_time_ms": 150
    }
    """
    start_time = time.time()
    
    try:
        job_input = job["input"]
        
        # Load model
        predictor = load_model()
        
        # Parse input
        image_base64 = job_input["image_base64"]
        click_x = job_input.get("click_x", 320)
        click_y = job_input.get("click_y", 240)
        bounds = job_input.get("bounds")
        
        # Decode image
        image = base64_to_image(image_base64)
        height, width = image.shape[:2]
        
        print(f"Segmenting at pixel ({click_x}, {click_y}) on {width}x{height} image")
        
        # Set image for predictor
        predictor.set_image(image)
        
        # Create input point
        input_point = np.array([[click_x, click_y]])
        input_label = np.array([1])  # foreground
        
        # Predict mask
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Get best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = float(scores[best_idx])
        
        # Convert to polygon
        polygon_pixels = mask_to_polygon(best_mask)
        
        if polygon_pixels is None:
            return {
                "error": "Could not create polygon from mask",
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
        
        # Transform to geographic if bounds provided
        if bounds:
            polygon_geo = transform_polygon_to_geo(polygon_pixels, bounds, (width, height))
        else:
            polygon_geo = polygon_pixels
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "polygon": polygon_geo,
            "confidence": best_score,
            "processing_time_ms": processing_time,
            "mask_area_pixels": int(np.sum(best_mask))
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})

