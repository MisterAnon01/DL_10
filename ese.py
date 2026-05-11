import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# 1. Load MNIST (subset) for training data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train[:5000].astype('float32') - 127.5) / 127.5 # Normalize to [-1, 1]
x_train = np.expand_dims(x_train, axis=-1)

# 2. Build the Generator (Problem 1)
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7*7*128, input_dim=100),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu'),
        layers.Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', activation='tanh')
    ])
    return model

# 3. Build the Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=(28,28,1)),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# 4. Build and Compile GAN
gen = build_generator()
disc = build_discriminator()
disc.trainable = False # Freeze discriminator during generator training
gan = tf.keras.Sequential([gen, disc])
gan.compile(loss='binary_crossentropy', optimizer='adam')

# 5. Training Loop (Simplified)
batch_size = 32
for epoch in range(101): # Small number for demonstration
    # Train Discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    fake_imgs = gen.predict(np.random.normal(0, 1, (batch_size, 100)), verbose=0)
    
    d_loss_real = disc.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = disc.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
    
    # Train Generator (Problem 3)
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} [D loss: {0.5 * (d_loss_real + d_loss_fake):.4f}] [G loss: {g_loss:.4f}]")