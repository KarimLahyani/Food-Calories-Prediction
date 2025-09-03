import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
# Load the dataset

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    
    # Enable memory growth to prevent GPU memory from being allocated all at once
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ GPU memory growth enabled")
else:
    print("❌ No GPU detected - training will use CPU")
    
(train_ds, val_ds), ds_info = tfds.load(
    "food101",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

# Get class names
class_names = ds_info.features["label"].names
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

# Preprocessing function
IMG_SIZE = 224

def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

BATCH_SIZE = 32

# Prepare the data
train_batches = (
    train_ds
    .map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_batches = (
    val_ds
    .map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Load MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_batches,
    validation_data=val_batches,
    epochs=50  # adjust if needed
)

# Save model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/food_model_transfer1.keras")

# Optional: plot accuracy
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy")
plt.savefig("accuracy_plot1.png")
plt.show()
