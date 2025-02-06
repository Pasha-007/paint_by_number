import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import os

def generate_paint_by_number(image_path, num_colors=10, output_dir='output'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image if too large
    max_size = 800
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        dim = (int(width * scale), int(height * scale))
        img_rgb = cv2.resize(img_rgb, dim, interpolation=cv2.INTER_AREA)
    
    # Reshape image for clustering
    pixels = img_rgb.reshape(-1, 3)
    
    # Color quantization
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    labels = kmeans.labels_
    colors = kmeans.cluster_centers_
    
    # Reconstruct quantized image
    quantized_img = colors[labels].reshape(img_rgb.shape).astype(np.uint8)
    
    # Create PIL images
    quantized_pil = Image.fromarray(quantized_img)
    
    # Create white background with numbered regions
    line_art = Image.new('RGB', quantized_pil.size, color='white')
    draw = ImageDraw.Draw(line_art)
    
    # Detect and draw region boundaries
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Draw black lines for region boundaries
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:
                draw.point((x, y), fill='black')
    
    # Save processed images
    quantized_pil.save(os.path.join(output_dir, 'quantized_image.png'))
    line_art.save(os.path.join(output_dir, 'line_art.png'))
    
    # Create color palette
    palette_img = Image.new('RGB', (num_colors * 50, 50), color='white')
    palette_draw = ImageDraw.Draw(palette_img)
    
    for i, color in enumerate(colors):
        color_tuple = tuple(map(int, color))
        palette_draw.rectangle([i*50, 0, (i+1)*50, 50], fill=color_tuple)
    
    palette_img.save(os.path.join(output_dir, 'color_palette.png'))
    
    print(f"Images saved in {output_dir}")

# Example usage
result = generate_paint_by_number('/Users/muntahaakhan/Screenshots/clip-cute-cartoon-cats-16.jpg')