import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2, csv, pickle, numpy as np, matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

path="./ExeImg_Dataset"
model_path="./Model"

# Hyperparameters
num_classes = 2
input_shape = (299, 299, 3)
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 50
batch_size = 32
batch_testsize = 28
image_size = 128
patch_size = 8
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 2

# Size of the transformer layers
transformer_units = [projection_dim * 2, projection_dim,]        

transformer_layers = 1			

# Size of the dense layers of the final classifier
mlp_head_units = [2048, 1024]
key_dim = projection_dim // num_heads

# Improved tensorflow data pipeline, to avoid loading large C-style arrays into RAM.
# even though these numpy arrays are optimized, their collective size is enough to cause a RAM insufficiency
# even on a Kaggle VM, which offers roughly 19 Gigabytes for the latter.
train_ds, val_ds = image_dataset_from_directory(
    path,
    batch_size = batch_size,
    image_size = input_shape[:2],
    shuffle = True,
    validation_split = 0.1,
    subset = "both",
    seed = 8,
    labels = 'inferred'
)
Len = lambda *d: sum(i.cardinality().numpy() for i in d)
tlen = lambda *d: Len(*d) * batch_size

test_ds = train_ds.take(floor(0.2 * Len(train_ds)))

# Checking the shape of the datasets
STEP_SIZE_TRAIN = int(np.ceil(tlen(train_ds) / batch_size))
STEP_SIZE_VALID = int(np.ceil(tlen(val_ds) / batch_size))
STEP_SIZE_TEST =  int(np.ceil(tlen(test_ds) / batch_testsize))

adapt_ds = train_ds.map(lambda x, y: x, num_parallel_calls = tf.data.AUTOTUNE)

data_augmentation = keras.Sequential(
    [ layers.Normalization(), layers.Resizing(image_size, image_size) ], name="data_augmentation"
)

data_augmentation.layers[0].adapt(adapt_ds)

# Building MLP network
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Creating patches using tensorflow's extract_patches module
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
 
    def get_config(self):
        config = {
            "patch_size" : self.patch_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Encoding the patches and adding postion embeddings
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
 
    def get_config(self):
        config = {
            "num_patches" : self.num_patches,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Building Vision Transformer
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    print(f'Shape of inputs: {inputs.shape}')
    
    # Augment data
    augmented = data_augmentation(inputs)
    print(f'Shape of augmented: {augmented.shape}')
    
    # Create patches
    patches = Patches(patch_size)(augmented)
    print(f'Shape of patches: {patches.shape}')
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    print(f'Shape of encoded_patches: {encoded_patches.shape}')
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=key_dim, use_bias = False, dropout=0.1
        )(x1, x1)
        print(f'Shape of attention_output: {attention_output.shape}')
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    print(f'Shape of mlp output: {features.shape}')
    # Classify outputs
    logits = layers.Dense(num_classes)(features)
    print(f'Shape of logits: {logits.shape}')
    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=logits)

    return model

def run_experiment(model):
    # Compiling the model
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    # Save the Keras model weights at some frequency (here, when the best validation accuracy is achieved)
    checkpoint_filepath = "./checkpoint/model.weights.h5"    # the naming explicitly must end in .weights.h5
    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    # Training the model
    history = model.fit(
        train_ds,
        steps_per_epoch = STEP_SIZE_TRAIN,
        batch_size = batch_size,
        epochs = num_epochs,
        validation_data = val_ds,
        validation_steps = STEP_SIZE_VALID,
        callbacks = [checkpoint_callback],
        shuffle=True
    )
   
    # Save the model (saving with h5 does not work anymore.)
    model.save(os.path.join(model_path, "ViT_Model.keras"), overwrite=True)
    
    # Checking the accuracy
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(test_ds, batch_size=batch_testsize)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    
    # Plotting the accuracies
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'Accuracy.png'))
    plt.clf()
    
    # Plotting the loss
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(model_path, 'Loss.png'))
    
    # Making predictions
    y_pred = model.predict(test_ds, steps=STEP_SIZE_TEST)
    y_pred = np.argmax(y_pred,axis=1)
    
    labels = {'Benignware': 0, 'Malware': 1}
    print(f"Labels and their corresponding encodings: {labels}")
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in y_pred]
    print(f"Predictions: {y_pred}")
    print(f"Predictions: {predictions}")
    
    # Get the classification report
    # print(classification_report(y_pred,y_test))
    
    # Get the confusion matrix
    # print(confusion_matrix(y_pred,y_test))

    return history

vit_classifier = create_vit_classifier()
vit_classifier.summary()
history = run_experiment(vit_classifier)
