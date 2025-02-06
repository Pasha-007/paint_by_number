import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import os

def generate_paint_by_number(image_path, num_colors=10, output_dir='output'):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Reshape image for clustering
        pixels = img_rgb.reshape(-1, 3)
        
        # Color quantization using K-means
        kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
        labels = kmeans.labels_
        colors = kmeans.cluster_centers_
        
        # Reconstruct quantized image
        quantized_img = colors[labels].reshape(img_rgb.shape).astype(np.uint8)
        
        # Edge detection for line art
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Save quantized image
        quantized_pil = Image.fromarray(quantized_img)
        quantized_path = os.path.join(output_dir, 'quantized_image.png')
        quantized_pil.save(quantized_path)
        
        # Save line art
        line_art = Image.fromarray(edges)
        line_art_path = os.path.join(output_dir, 'line_art.png')
        line_art.save(line_art_path)
        
        # Generate color palette image
        palette_img = Image.new('RGB', (num_colors * 50, 50))
        draw = ImageDraw.Draw(palette_img)
        
        for i, color in enumerate(colors):
            color_tuple = tuple(map(int, color))
            draw.rectangle([i*50, 0, (i+1)*50, 50], fill=color_tuple)
        
        palette_path = os.path.join(output_dir, 'color_palette.png')
        palette_img.save(palette_path)
        
        print(f"Processed images saved in {output_dir}")
        return {
            'quantized_image_path': quantized_path,
            'line_art_path': line_art_path,
            'color_palette_path': palette_path
        }
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Example usage
result = generate_paint_by_number('/Users/muntahaakhan/Screenshots/clip-cute-cartoon-cats-16.jpg')