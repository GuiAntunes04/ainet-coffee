import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMAGE_SIZE_MOBILENET, MODEL_MOBILENET_PATH
from utils import load_datasets


preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
train_ds, val_ds, class_names = load_datasets(
    image_size=IMAGE_SIZE_MOBILENET,
    preprocess=preprocess,
)

print("Class order:", class_names)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE_MOBILENET, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE_MOBILENET, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(MODEL_MOBILENET_PATH),
    monitor="val_loss",
    save_best_only=True,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=[checkpoint, early_stop],
)

print(f"Best MobileNet model saved to {MODEL_MOBILENET_PATH.name}")
