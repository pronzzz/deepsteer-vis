from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
import tensorflow as tf

def create_model():
    """
    Creates the modified NVIDIA PilotNet architecture with Hyper-Regularization.
    """
    model = Sequential()
    
    # 1. Input Layer & Preprocessing
    # Preprocessing is done in generator usually, but prompt asked for:
    # "Input Layer: Normalization/Lambda layer."
    # We already resize in utils, but lambda normalization is standard here.
    # Input shape: (66, 200, 3) from our utils.py
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    
    # 2. Convolutional Layers
    # Layer 1: 24 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Layer 2: 36 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Layer 3: 48 filters, 5x5 kernel, 2x2 stride
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Layer 4: 64 filters, 3x3 kernel, non-strided
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Layer 5: 64 filters, 3x3 kernel, non-strided
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # 3. Fully Connected Layers
    model.add(Flatten())
    
    # Dense 1
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Dense 2
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Dense 3
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.5)) # Hyper-Regularization
    
    # Output Layer
    model.add(Dense(1))
    
    return model

if __name__ == '__main__':
    # Training pipeline
    # We will need to import generator and data loading here or in a separate train.py.
    # For this task, "Implement training script/main execution block" is part of Model Implementation.
    # So I will put the training logic here.
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from utils import generator
    from keras.callbacks import ModelCheckpoint
    import os
    import csv # Added missing import

    # Load Data
    # Assuming standard Udacity structure: data/driving_log.csv
    # If not found, user might need to point to it. 
    # For now we assume a path or fail gracefully.
    
    csv_path = 'data/driving_log.csv'
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please ensure 'data/driving_log.csv' exists.")
        exit()
        
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip header if present, Udacity csv usually has no header or careful
        # Udacity Beta simulator has header? 
        # Inspecting first line is safer usually, but let's assume standard.
        # Actually standard Udacity data (old) has no header, newer has header.
        # Let's read with pandas for safety or just use csv
        pass 
    
    # Re-read properly
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] == 'center': continue # Skip header
            lines.append(line)
            
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(validation_samples)}")
    
    # Compile Model
    model = create_model()
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    
    # Generators
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    
    # Checkpoint
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    
    # Train
    # Steps per epoch: len(train_samples) / batch_size
    # Since we augment and use 3 cameras, our generator yields batch_size images.
    # But effectively we have 3x data. 
    # If we want to cover all data once per epoch:
    # Our generator logic in utils yields batch_size items. 
    # But for each line in samples, we produce 3 images? 
    # My utils.py generator implementation:
    # "for offset in range(0, num_samples, batch_size): ... batch_samples = samples[offset:offset+batch_size]"
    # Inside loop: append center, left, right ... roughly 3 * batch_size images per "batch_samples" block?
    # Actually my utils.py accumulates all of them to `images`.
    # So if batch_size=32 (passed to generator), `batch_samples` has 32 lines.
    # `images` will have 32 * 3 = 96 images.
    # Then `yield ...` returns 96 images.
    # Keras `fit` expects `steps_per_epoch` batches.
    # If steps_per_epoch = len(train_samples) / 32
    # Then we cover all samples.
    
    batch_size = 32
    steps_per_epoch = max(1, len(train_samples) // batch_size)
    validation_steps = max(1, len(validation_samples) // batch_size)
    
    history = model.fit(train_generator, 
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        epochs=5,
                        verbose=1,
                        callbacks=[checkpoint])
    
    model.save('model_final.h5')
    print("Model saved.")
