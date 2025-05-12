import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

import gradio as gr

MODEL_AND_MAP_DIR = '.'

MODEL_PATH = os.path.join(MODEL_AND_MAP_DIR, 'my_fashion_model.keras')
LABEL_MAP_PATH = os.path.join(MODEL_AND_MAP_DIR, 'label_map.json')

IMG_HEIGHT = 224
IMG_WIDTH = 224

print("Starting application...")
print(f"Looking for model and label map in directory: {os.path.abspath(MODEL_AND_MAP_DIR)}")

try:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}.")
    print("Please ensure 'my_fashion_model.keras' is in the same folder as the script.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

try:
    print(f"Loading label map from: {LABEL_MAP_PATH}")
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map_str_keys = json.load(f)
        label_map = {int(k): v for k, v in label_map_str_keys.items()}
    print("Label map loaded successfully.")
except FileNotFoundError:
    print(f"Error: Label map file not found at {LABEL_MAP_PATH}.")
    print("Please ensure 'label_map.json' is in the same folder as the script.")
    label_map = None
except Exception as e:
    print(f"An error occurred while loading the label map: {e}")
    label_map = None

def classify_fashion_image(image: Image.Image):
    if model is None or label_map is None:
        return {"Error": "Model or label map failed to load. Check the console for details."}

    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)

        class_probabilities = predictions[0]

        results = {}
        for i, probability in enumerate(class_probabilities):
             class_name = label_map.get(i, f"Unknown_{i}")
             results[class_name] = float(probability)

        return results

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return {"Error during prediction": str(e)}

if model is not None and label_map is not None:
    print("\nBuilding Gradio interface...")
    interface = gr.Interface(
        fn=classify_fashion_image,
        inputs=gr.Image(type="pil", label="Upload a Fashion Image"),
        outputs=gr.Label(label="Classification Result"),
        title="Fashion Classifier",
        description="Upload an image of a clothing item to get its predicted article type."
    )

    if __name__ == "__main__":
        print("\nStarting Gradio interface...")
        interface.launch(inbrowser=True, share=False)
        print("Gradio interface started. Check your default web browser.")
        print("If browser did not open, access it at http://127.0.0.1:7860/")
else:
    print("\nSkipping Gradio interface launch because model or label map failed to load.")
    print("Please check the console output above for errors during model/label map loading.")