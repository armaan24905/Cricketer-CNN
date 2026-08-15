import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models
from keras.utils import image_dataset_from_directory, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report

# 1. DATASET PATH
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "results" / "train_small"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

# 2. LOAD DATASET
train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#3. CLASS NAMES
class_names = train_dataset.class_names
print("\nCricketer Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")
NUM_CLASSES = len(class_names)
print("\nTotal Classes:", NUM_CLASSES)

# 4. NORMALIZATION
normalization_layer = layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)
validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# 5. DISPLAY SAMPLE IMAGES
plt.figure(figsize=(12, 8))
for images, labels in train_dataset.take(1):
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.suptitle("Cricketer Dataset Samples")
plt.show()

# 6. BUILD CNN MODEL
model = models.Sequential([
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(
        128,
        activation="relu"
    ),
    layers.Dropout(0.5),
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# 7. MODEL SUMMARY
model.summary()

# 8. COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 9. TRAIN MODEL
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# 10. TRAINING / VALIDATION ACCURACY
accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]
epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(
    epochs_range,
    accuracy,
    marker="o",
    label="Training Accuracy"
)
plt.plot(
    epochs_range,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# 11. TRAINING / VALIDATION LOSS
loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.figure(figsize=(10, 6))
plt.plot(
    epochs_range,
    loss,
    marker="o",
    label="Training Loss"
)
plt.plot(
    epochs_range,
    validation_loss,
    marker="o",
    label="Validation Loss"
)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 12. CONFUSION MATRIX
y_true = []
y_pred = []
for images, labels in validation_dataset:
    predictions = model.predict(
        images,
        verbose=0
    )
    predicted_classes = np.argmax(
        predictions,
        axis=1
    )
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)
y_true = np.array(y_true)
y_pred = np.array(y_pred)
conf_matrix = confusion_matrix(
    y_true,
    y_pred
)
plt.figure(figsize=(12, 10))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Cricketer Classification Confusion Matrix")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 13. CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# 14. SAVE MODEL
model.save("cricketer_cnn_model.keras")
print("\nModel saved successfully!")

# 15. PREDICT A NEW CRICKETER IMAGE
from tensorflow.keras.utils import load_img, img_to_array
IMAGE_PATH = "virat kolhi face_45.jpg"
if os.path.exists(IMAGE_PATH):
    img = load_img(
        IMAGE_PATH,
        target_size=IMG_SIZE
    )
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(
        img_array,
        axis=0
    )
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]
    predicted_index = np.argmax(predictions)
    predicted_name = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    print("\n===================================")
    print("CRICKETER PREDICTION")
    print("===================================")
    print("Predicted:", predicted_name)
    print(f"Confidence: {confidence:.2f}%")
    print("===================================")
    print("\nAll Class Probabilities:")
    for i, probability in enumerate(predictions):
        print(
            f"{class_names[i]}: "
            f"{probability * 100:.2f}%"
        )
else:
    print(
        f"\nPrediction image '{IMAGE_PATH}' "
        "not found."
    )