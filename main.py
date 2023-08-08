!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install opencv-python-headlessimport torch
import numpy as np
import os
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files
import clip

# Load the YOLO model
yolov8 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device)

# function to load and preprocess images
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image file '{image_path}' could not be loaded")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

    # function to detect objects in images
def detect_objects(img, model):
    results = model(img)
    # Just use the first four elements of each box for cropping the image
    entities = [img.crop(box[:4].cpu().numpy()) for box in results.xyxy[0]]
    return entities

# Function to get features from images (corrected)
def get_features(imgs):
    images = torch.stack([preprocess(img) for img in imgs]).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(images)
    return features.cpu().numpy()

 # function to compare features
def compare_features(main_features, other_features):
    return cosine_similarity(main_features, other_features)

# function to save images
def save_images(main_img, other_imgs, indices, main_image_path):
    base_name = os.path.splitext(main_image_path)[0]
    os.makedirs(base_name, exist_ok=True)
    main_img.save(f'{base_name}/{base_name}_main.png')
    for i, idx in enumerate(indices):
        # other_imgs[idx] should give a cropped image
        other_imgs[idx].save(f'{base_name}/{base_name}_top{i+1}.png')

# Step 5: Main routine to process each image
def process_images(main_image_path, other_image_paths):
    main_img_pil = load_image(main_image_path)
    main_entities = detect_objects(main_img_pil, yolov8)
    main_features = get_features(main_entities)

    for image_path in other_image_paths:
        img_pil = load_image(image_path)
        entities = detect_objects(img_pil, yolov8)
        features = get_features(entities)
        similarities = compare_features(main_features, features)
        top3_indices = np.argsort(similarities.flatten())[-3:]
        save_images(main_img_pil, entities, top3_indices, main_image_path)

# Step 6: Run the main routine
# Specify the main image and the other images
main_image = 'BurgerStock.jpeg'  # replace with your main image file name
other_images = ['BurgerzStock.jpeg']

# Process the images
process_images(main_image, other_images)
