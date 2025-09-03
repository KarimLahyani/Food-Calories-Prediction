import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
import os
import matplotlib.pyplot as plt

# 💻 Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU is available!")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled")
else:
    print("❌ No GPU detected. Training will use CPU.")


# 📥 Load the Food101 dataset
(train_data, val_data), ds_info = tfds.load(
    "food101",
    split=["train", "validation"],
    as_supervised=True,
    with_info=True
)

# 📦 Extract class names
class_names = ds_info.features["label"].names

# 💾 Save class names for prediction use
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

# ⚙️ Preprocessing
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))  # Reshape
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

# 🌀 Shuffle, map, batch
BATCH_SIZE = 32
train_data = train_data.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 🧠 Define CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 📊 TensorBoard setup
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 🎯 Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[tensorboard_cb]
)

# 💾 Save the model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/food101_model0.keras")

# Optional: plot accuracy
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy")
plt.savefig("accuracy_plot0.png")
plt.show()
