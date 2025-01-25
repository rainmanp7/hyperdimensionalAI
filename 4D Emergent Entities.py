import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# =============================================
# Enhanced 4D Data Generation
# =============================================
def generate_4d_data_sample(num_points=10):
    X = np.random.randn(num_points, 4) * 2  # Wider spread
    closest_idx = np.random.randint(num_points)
    X[closest_idx] *= 0.1  # Reduce distance for target point
    distances = np.linalg.norm(X, axis=1)
    y = np.argmin(distances)
    return X, y

def generate_4d_data(num_samples=5000, num_points=10):
    with Pool() as pool:
        data = pool.starmap(generate_4d_data_sample, [(num_points,) for _ in range(num_samples)])
    X, y = zip(*data)
    return np.array(X), np.array(y)

# =============================================
# Geometric 4D Processing Model
# =============================================
def build_4d_model(num_points=10):
    inputs = tf.keras.Input(shape=(num_points, 4))
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    distance_estimates = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(x)
    x = tf.keras.layers.Flatten()(distance_estimates)
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# =============================================
# Validated Training Process
# =============================================
def run_experiment():
    X, y = generate_4d_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = build_4d_model()
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {acc*100:.1f}%")
    print(f"Random Baseline: {100/10}%")

if __name__ == "__main__":
    run_experiment()