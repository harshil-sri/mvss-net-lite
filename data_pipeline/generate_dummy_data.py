import os
import random
from PIL import Image, ImageDraw

# Ensure your directories exist
os.makedirs('data/doctamper/images', exist_ok=True)
os.makedirs('data/doctamper/masks', exist_ok=True)

def generate_synthetic_document(index):
    img_size = (512, 512)
    
    # 1. Create a blank "document" (white background)
    image = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw fake text lines to simulate a document
    for y in range(50, 480, 40):
        draw.line([(50, y), (450, y)], fill=(200, 200, 200), width=4)
        
    # 2. Create the Ground Truth Mask (black background)
    mask = Image.new('L', img_size, 'black')
    mask_draw = ImageDraw.Draw(mask)
    
    # 3. Simulate a Forgery (80% chance to be tampered)
    is_forged = random.random() < 0.8
    if is_forged:
        # Generate random coordinates for the fake text box
        x1, y1 = random.randint(100, 300), random.randint(100, 300)
        x2, y2 = x1 + random.randint(100, 150), y1 + random.randint(40, 80)
        
        # Draw the "tampered" block on the RGB image
        draw.rectangle([x1, y1, x2, y2], fill=(230, 230, 230), outline=(150, 150, 150))
        draw.text((x1 + 10, y1 + 15), "FORGED TEXT", fill="black")
        
        # Draw the exact same block in white on the binary mask
        mask_draw.rectangle([x1, y1, x2, y2], fill='white')
    
    # 4. Save the files
    image_path = f'data/doctamper/images/synth_{index:04d}.jpg'
    mask_path = f'data/doctamper/masks/synth_{index:04d}.png'
    
    # Save the RGB image as a JPEG to ensure ELA has compression artifacts to detect
    image.save(image_path, 'JPEG', quality=random.randint(75, 95))
    mask.save(mask_path, 'PNG')

print("Generating 100 synthetic documents and masks...")
for i in range(100):
    generate_synthetic_document(i)
print("Done! You can now test your pipeline.")  