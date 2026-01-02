import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
import cv2
import matplotlib.pyplot as plt
import sys
import os
from utils import preprocess_image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess (crop/resize) to match what the model sees (66, 200)
    # Note: If we want to overlay on the *original* full size image, we need to upsample the heatmap.
    # The utils.preprocess_image crops top/bottom. 
    # Let's visualize what the model sees.
    processed_img = preprocess_image(img) # Returns YUV
    # Convert YUV back to RGB for display
    display_img = cv2.cvtColor(processed_img, cv2.COLOR_YUV2RGB)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match the image size
    jet = cv2.resize(jet, (display_img.shape[1], display_img.shape[0]))
    
    # Convert display_img to BGR for OpenCV
    display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + display_img_bgr
    superimposed_img = np.uint8(superimposed_img)

    # Save the superimposed image
    cv2.imwrite(cam_path, superimposed_img)

    # Display using matplotlib only if not in save-only mode
    if "--save-only" not in sys.argv:
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Grad-CAM Saliency Map')
        plt.show() # This blocks
    else:
        print("Skipping display in save-only mode.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_activation.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("Model or image path does not exist.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, compile=False)
    
    # Find last conv layer
    last_conv_layer_name = ""
    for layer in reversed(model.layers):
        if 'conv2d' in layer.name:
            last_conv_layer_name = layer.name
            break
            
    print(f"Using layer: {last_conv_layer_name} for saliency map.")
    
    print(f"Processing image {image_path}...")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_image(img)
    img_array = np.expand_dims(processed_img, axis=0)
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0)
    
    # Save result
    save_and_display_gradcam(image_path, heatmap, cam_path="saliency_result.jpg")
    print("Saliency map saved as saliency_result.jpg")
