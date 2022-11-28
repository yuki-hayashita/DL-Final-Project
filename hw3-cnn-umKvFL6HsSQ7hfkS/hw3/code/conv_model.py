from symbol import argument
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

###############################################################################################


def get_default_CNN_model(
    conv_ns=tf.keras.layers,
    norm_ns=tf.keras.layers,
    drop_ns=tf.keras.layers,
    man_conv_ns=tf.keras.layers,
):
    """
    Sets up your model architecture and compiles it using the appropriate optimizer, loss, and
    metrics.

    :param conv_ns, norm_ns, drop_ns: what version of this layer to use (either tf.keras.layers or
                                      your implementation from layers_keras)
    :param man_conv_ns: what version of manual Conv2D to use (use tf.keras.layers until you want to
                        test out your manual implementation from layers_manual)

    :returns compiled model
    """

    Conv2D = conv_ns.Conv2D
    BatchNormalization = norm_ns.BatchNormalization
    Dropout = drop_ns.Dropout
    Conv2D_manual = man_conv_ns.Conv2D

    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 255),
            tf.keras.layers.experimental.preprocessing.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )

    ## TODO 1: Augment the input data (feel free to experiment) 
    ## https://www.tensorflow.org/guide/keras/preprocessing_layers
    
    augment_fn = tf.keras.Sequential([
        # tf.keras.layers.RandomCrop(32, 32)
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        # tf.keras.layers.RandomRotation(0.2),
        ])

    ## TODO 2: Make sure your first Conv2D is Conv2D_manual (after you've
    ## implemented it), has stride 2, 2, and goes up to low channel count
    ## (i.e. 16). This will speed up evaluation.
    ## Some possible layers you can use here are Conv2D, BatchNormalization,
    ## Dropout, tf.keras.layers.Dense, tf.keras.layers.MaxPool2d, and
    ## tf.keras.layers.Flatten
    model = CustomSequential(
        [
        # from Lab 4
        # First Convolution Layer
        Conv2D_manual(filters=16, kernel_size=5, strides=(2, 2), padding="same", name="Conv1"),
        BatchNormalization(name="Conv1-Norm"),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same", name="Conv1-Pool"),
    
        
        
        # Second Convolution Layer
        Conv2D(filters=32, kernel_size=5, strides=(2, 2), padding="same", name="Conv2"),
        BatchNormalization(name="Conv2-Norm"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same", name="Conv2-Pool"),
 
        # Third Convolution Layer
        Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding="same", name="Conv3"),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same", name="Conv6-Pool"),
        BatchNormalization(name="Conv3-Norm"),


        # Three Dense Layers
        # maybe try lowering dropout??
        tf.keras.layers.Flatten(name="Flatten"),
        tf.keras.layers.Dense(160, activation="leaky_relu", name="Dense1"),
        # Dropout(rate=0.3, name="Dropout1"),
        Dropout(rate=0.2, name="Dropout1"),
        tf.keras.layers.Dense(20, activation="leaky_relu", name="Dense2"),
        # Dropout(rate=0.3, name="Dropout2"),
        Dropout(rate=0.2, name="Dropout2"),
        tf.keras.layers.Dense(10, activation="softmax", name="Dense3"),

        ], 
                ## Take a look at the constructor for CustomSequential to see if you
        ## might need to pass in the necessary preparation functions...
        input_prep_fn=input_prep_fn,
        output_prep_fn=output_prep_fn,
        augment_fn=augment_fn
    )

    ## TODO 3: Compile your model using your choice of optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  ## feel free to change
        loss="categorical_crossentropy",  ## do not change loss/metrics
        metrics=["categorical_accuracy"],
    )

    ## TODO 4: Pick an appropriate number of epochs and batch size to use for training
    ## your model. Note that the autograder will time out after 10 minutes.
    return SimpleNamespace(model=model, epochs=30, batch_size=250)


###############################################################################################


class CustomSequential(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.

    DO NOT EDIT

    :param input_prep_fn: Modifies input images prior to running the forward pass
    :param output_prep_fn: Modifies input labels prior to running forward pass
    :param augment_fn: Augments input images prior to running forward pass
    """

    def __init__(
        self,
        *args,
        input_prep_fn=lambda x: x,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

    def batch_step(self, data, training=False):

        x_raw, y_raw = data

        x = self.input_prep_fn(x_raw)
        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)