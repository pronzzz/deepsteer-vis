import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model, Model
from utils import preprocess_image

# Initialize SocketIO and Flask
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        
        # Preprocessing
        image_array = np.asarray(image)
        # Keep original for visualization
        original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) 

        # Preprocess logic matches utils.py but we need to resize specifically for model
        # Our utils.preprocess_image does: Crop -> Resize(200,66) -> YUV
        processed_image = preprocess_image(image_array)
        
        # Add batch dimension
        processed_image_batch = np.expand_dims(processed_image, axis=0)

        # Predict
        prediction = model.predict(processed_image_batch, verbose=0)
        steering_angle = float(prediction[0])

        # Throttle Control (PI Controller simple version)
        set_speed = 20
        throttle = 1.2 * ((set_speed - speed) / set_speed) # Simple P-controller
        
        print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")

        # Send control
        send_control(steering_angle, throttle)
        
        # --- COMMAND LINE COMPETITIVE EDGE: Saliency Map ---
        try:
            # We need to find the last conv layer name. 
            # In our model.py, we have 5 conv layers. 
            # We can find it dynamically or hardcode if we knew the name.
            # Let's find the last Conv2D layer.
            last_conv_layer_name = ""
            for layer in reversed(model.layers):
                if 'conv2d' in layer.name:
                    last_conv_layer_name = layer.name
                    break
            
            if last_conv_layer_name:
                heatmap = make_gradcam_heatmap(processed_image_batch, model, last_conv_layer_name, pred_index=0)
                
                # Rescale heatmap to a range 0-255
                heatmap = np.uint8(255 * heatmap)

                # Use jet colormap to colorize heatmap
                jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Resize heatmap to match the original image size (320x160 usually)
                # Note: utils.preprocess_image crops the image. 
                # To overlay on *original* uncropped image, we need to map it back or just overlay on cropped.
                # Project requirement: "overlay a heat map on the drive window".
                # It's cleaner to show the cropped view since that's what the network sees.
                # But typically users want to see the full context.
                # Let's resize to processed size (200x66) first.
                jet_resized = cv2.resize(jet, (200, 66))
                
                # Create an image that is the preprocessed image (YUV) converted back to BGR for display
                # Or just use the cropped original RGB
                cropped_original = image_array[70:-25, :, :]
                cropped_original = cv2.resize(cropped_original, (200, 66))
                cropped_original = cv2.cvtColor(cropped_original, cv2.COLOR_RGB2BGR)
                
                # Superimpose the heatmap on original image
                superimposed_img = jet_resized * 0.4 + cropped_original
                superimposed_img = np.uint8(superimposed_img)

                # Display
                cv2.imshow("DeepSteer-Vis: " + last_conv_layer_name, superimposed_img)
                cv2.waitKey(1)
        except Exception as e:
            print(f"Saliency Map Error: {e}")

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={})

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    
    # Load model
    try:
        model = load_model('model.h5')
    except:
        # If model.h5 doesn't exist, try loading model_final.h5 or warn user
        try:
             model = load_model('model_final.h5')
        except:
             print("Please train the model first and ensure 'model.h5' exists.")
             # We can't really run without model, but for structure we continue
             # exit() 
             pass

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
