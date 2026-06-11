import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Step 1: Load Data
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'dataset/train',
    target_size=(128, 128),
    batch_size=17,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    'dataset/test',
    target_size=(128, 128),
    batch_size=17,
    class_mode='binary'
)

# Step 2: Build Model
model = Sequential([
    Input(shape=(128, 128, 3)),         # ✅ Fixes the warning
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Step 3: Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # ✅ More stable training
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 4: Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,                 # ✅ More patience
    restore_best_weights=True
)

# Step 5: Train
model.fit(
    train_data,
    epochs=20,                  # ✅ More epochs
    validation_data=test_data,
    callbacks=[early_stop]
)

# Step 6: Save
model.save("model.keras")
print("✅ Model saved successfully!")