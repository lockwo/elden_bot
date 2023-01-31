import tensorflow as tf

# from tensorflow.keras.applications.efficientnet import EfficientNetB2
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.inception_v3 import InceptionV3
import time
import numpy as np
import wandb
from wandb.keras import WandbCallback
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

actions = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
num_actions = len(actions)

num_runs = 32
folderStrings = ["test_images"]
for i in range(1, num_runs + 1):
    folderStrings.append(f"test_images{i}")

data = []
labels = []

for folderString in folderStrings:
    batch = 100
    init = 0
    end = batch
    i = 0
    while True:
        try:
            if i == init:
                ins = np.load(f"{folderString}/{init}_{end}.npy")
                images = np.load(f"{folderString}/img{init}_{end}.npy")
            if i == end:
                init += 100
                end += 100
                ins = np.load(f"{folderString}/{init}_{end}.npy")
                images = np.load(f"{folderString}/img{init}_{end}.npy")
            data.append(images)
            labels.append(ins)
            i += batch
        except FileNotFoundError:
            break

data = np.vstack(data)
labels = np.vstack(labels)
print(data.shape)
# (batch, height, width, new_axis)
data = np.expand_dims(data, axis=-1).transpose(0, 2, 1, 3)
# Frame stacking
# (0, 1, 2, 3, 4)
# ((0, 0, 0), (0, 1, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4))
# data1 = (0, 0, 1, 2, 3)
# data2 = (0, 0, 0, 1, 2)
# inter = ((0, 0), (0, 1), (1))
data1 = np.concatenate((np.array([data[0]]), data[:-1]), axis=0)
data2 = np.concatenate((np.array([data[0]]), data[:-1]), axis=0)
data = np.concatenate((np.concatenate((data2, data1), axis=-1), data), axis=-1)


data = tf.random.shuffle(data, seed=42)
labels = tf.random.shuffle(labels, seed=42)
print(data.shape)

params = {}
params["learning_rate"] = 3e-4
params["epochs"] = 1000
params["split"] = 9 / 10
params["batch_size"] = 128
params["shuffle"] = 100 * num_runs
params["base_model"] = "custom"

split = int(params["split"] * len(data))
training_data = data[:split]
training_labels = labels[:split]
testing_data = data[split:]
testing_labels = labels[split:]

train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_labels))
BATCH_SIZE = params["batch_size"]
SHUFFLE_BUFFER_SIZE = params["shuffle"]

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

inputs = tf.keras.layers.Input(shape=(244, 138, 3))
"""
if params["base_model"] == "InceptionV3":
	base_model = InceptionV3(weights=None, include_top=False, pooling='avg', input_tensor=inputs)
x = base_model.output
x = tf.keras.layers.Dense(num_actions, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)
"""

init = tf.keras.initializers.VarianceScaling(scale=2)
x = tf.keras.layers.Conv2D(
    32, 8, strides=(4, 4), activation="relu", kernel_initializer=init
)(inputs)
x = tf.keras.layers.Conv2D(
    64, 4, strides=(3, 3), activation="relu", kernel_initializer=init
)(x)
x = tf.keras.layers.Conv2D(
    64, 3, strides=(1, 1), activation="relu", kernel_initializer=init
)(x)
# x = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.3)(x)
# x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=init)(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer=init)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(num_actions, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
loss = tf.keras.losses.BinaryCrossentropy()
metrics = ["accuracy"]

model.compile(optimizer=opt, loss=loss, metrics=metrics)

wandb.init(project="supervised_runs", entity="elden_ring_ai", name="video_example1")

checkpoint_filepath = "./video_test/checkpoint"
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=checkpoint_filepath,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        save_weights_only=True,
        monitor="val_loss",
        verbose=1,
    ),
    WandbCallback(),
]

# actions = ["W", "A", "S", "D", "F", "R", "U", "I", "O", "P", " "]
weights = [
    1 / (np.sum(labels[:, i]) + 1) * labels.shape[0] for i in range(len(labels[0]))
]
print(weights)
class_weight = {i: weights[i] for i in range(len(weights))}

model.fit(
    train_dataset,
    epochs=params["epochs"],
    validation_data=test_dataset,
    callbacks=callbacks,
    class_weight=class_weight,
)
