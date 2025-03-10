import numpy as np
import torch
import os
import tempfile
import base64
from io import BytesIO
import cv2
from PIL import Image

def detect_tuberculosis(image_path, age, sex, position, device=None):
    """
    Detect tuberculosis in a chest X-ray image.
    
    Args:
        image_path: Path to the chest X-ray image.
        age: Patient's age.
        sex: Patient's sex (0 for Male, 1 for Female).
        position: Image position (0 for AP, 1 for PA).
        device: Device to run the model on (cpu or cuda).
    
    Returns:
        Dict containing segmentation mask path, original image path, image and mask base64 encodings,
        and calculated features for LLM processing.
    """
    model_path = 'saved_models/_best_model.pt'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"TB model not found at {model_path}. Please ensure the model file exists.")
        return None
    
    device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    
    # Define preprocessing transformations
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
        transforms.Resize((256, 256)),  # Resize to model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])
    
    # Load and preprocess image
    original_img = Image.open(image_path).convert("L")  # Convert to grayscale
    original_np = np.array(original_img)
    
    # Transform for model
    image = transform(original_img).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run prediction
    with torch.no_grad():
        try:
            output = model(image)  # Try passing only the image
        except TypeError:
            metadata = torch.tensor([[age / 100.0, sex, position]], dtype=torch.float32).to(device)
            output = model(image, metadata)  # If error, try with metadata
            
    # Ensure mask has the correct shape (remove extra dimensions)
    mask = output.squeeze().cpu().numpy()
            
    # Ensure it's a 2D array for PIL
    if mask.ndim == 3:
        mask = mask[0]  # Take first channel if needed
            
    # Create binary mask (values above threshold are considered lung areas)
    threshold = 0.5
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    
    # Resize binary mask to match original image dimensions if needed
    if binary_mask.shape != original_np.shape:
        binary_mask = cv2.resize(binary_mask, (original_np.shape[1], original_np.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
    
    # Calculate features
    features = calculate_lung_features(original_np, binary_mask)
    
    # Save the segmentation mask as a PNG file
    mask_image = Image.fromarray(binary_mask)
    
    # Create a temporary file for the mask
    temp_mask_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    mask_filename = temp_mask_file.name
    mask_image.save(mask_filename)
    
    # Encode original image and mask to base64
    original_img_b64 = image_to_base64(original_img)
    mask_img_b64 = image_to_base64(mask_image)
    
    # Create masked image (original image with mask applied)
    masked_img = apply_mask_to_image(original_np, binary_mask)
    masked_img_pil = Image.fromarray(masked_img)
    masked_img_b64 = image_to_base64(masked_img_pil)
    
    return {
        "mask_path": mask_filename,
        "original_image": image_path,
        "original_image_b64": original_img_b64,
        "mask_b64": mask_img_b64,
        "masked_image_b64": masked_img_b64,
        "features": features
    }

def calculate_lung_features(image, mask):
    """
    Calculate various features from the lung segmentation.
    
    Args:
        image: Original image as numpy array
        mask: Binary mask as numpy array (255 for lung areas, 0 for background)
    
    Returns:
        Dictionary of calculated features
    """
    # Ensure mask is binary (0 and 1)
    binary_mask = mask.astype(bool)
    
    # Total lung area (number of white pixels)
    lung_area = np.sum(binary_mask)
    
    # Total image area
    total_area = image.shape[0] * image.shape[1]
    
    # Lung area ratio
    lung_area_ratio = lung_area / total_area
    
    # Calculate intensity statistics within the lung region
    if lung_area > 0:
        # Apply mask to image to consider only lung regions
        masked_image = image * binary_mask
        
        # Only consider pixels inside the mask for statistics
        lung_pixels = image[binary_mask]
        
        mean_intensity = np.mean(lung_pixels)
        std_intensity = np.std(lung_pixels)
        min_intensity = np.min(lung_pixels)
        max_intensity = np.max(lung_pixels)
    else:
        mean_intensity = 0
        std_intensity = 0
        min_intensity = 0
        max_intensity = 0
    
    # Split mask into left and right lungs
    height, width = mask.shape
    mid_point = width // 2
    
    left_mask = binary_mask.copy()
    left_mask[:, mid_point:] = False
    
    right_mask = binary_mask.copy()
    right_mask[:, :mid_point] = False
    
    left_area = np.sum(left_mask)
    right_area = np.sum(right_mask)
    
    # Left-to-right lung ratio
    left_to_right_ratio = left_area / right_area if right_area > 0 else 0
    
    # Calculate opacity score (higher value = more opaque/white areas in the lung)
    # This is a simple heuristic - you might want to refine this based on medical knowledge
    if lung_area > 0:
        # Higher values in X-ray typically represent more opaque areas
        # Normalize the mean intensity to be between 0 and 1
        opacity_score = mean_intensity / 255
    else:
        opacity_score = 0
    
    return {
        "lung_area_ratio": float(lung_area_ratio),
        "mean_intensity": float(mean_intensity),
        "std_intensity": float(std_intensity),
        "min_intensity": float(min_intensity),
        "max_intensity": float(max_intensity),
        "left_to_right_ratio": float(left_to_right_ratio),
        "opacity_score": float(opacity_score),
        "lung_area_pixels": int(lung_area)
    }

def apply_mask_to_image(image, mask):
    """
    Apply the binary mask to the original image to highlight the lung regions.
    
    Args:
        image: Original grayscale image as numpy array
        mask: Binary mask as numpy array
    
    Returns:
        Masked image as numpy array
    """
    # Ensure both are the same shape
    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Convert grayscale to RGB for colored overlay
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    # Create a colored mask overlay (green for lungs)
    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[:, :, 1] = mask  # Set green channel to mask value
    
    # Combine original image with mask overlay
    alpha = 0.3  # Transparency factor
    masked_image = cv2.addWeighted(image_rgb, 1, mask_rgb, alpha, 0)
    
    return masked_image

def image_to_base64(img):
    """
    Convert a PIL Image to base64 encoded string.
    
    Args:
        img: PIL Image object
    
    Returns:
        Base64 encoded string of the image
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str