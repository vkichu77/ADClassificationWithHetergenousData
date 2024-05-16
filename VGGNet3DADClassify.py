### Step 1: Define the Python Class

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class VGGNet3DADClassify:
    def __init__(self, input_shape=(128, 128, 128, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv3D(64, (3, 3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),

            Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),

            Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
            Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
            Conv3D(256, (3, 3, 3), activation='relu', padding='same'),
            MaxPooling3D((2, 2, 2), strides=(2, 2, 2)),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        return model

    def compile(self, learning_rate=0.0001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=5):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                              validation_data=(X_val, y_val))

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def summary(self):
        return self.model.summary()
```

### Step 2: Example Usage
```python
# Assuming you have your preprocessed data loaded as X_train, y_train, X_val, y_val, X_test, y_test

# Initialize the class
vgg_ad_classifier = VGGNet3DADClassify()

# Compile the model
vgg_ad_classifier.compile(learning_rate=0.0001)

# Print model summary
vgg_ad_classifier.summary()

# Train the model
history = vgg_ad_classifier.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=5)

# Evaluate the model
test_loss, test_accuracy = vgg_ad_classifier.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
```

