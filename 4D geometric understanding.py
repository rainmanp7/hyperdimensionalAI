import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# =============================================
# Enhanced 4D Data Generation
# =============================================
def generate_4d_data_sample(num_points=10):
    """Generates a single 4D data sample with num_points points"""
    X = np.random.randn(num_points, 4) * 2  # 4D coordinates
    closest_idx = np.random.randint(num_points)
    X[closest_idx] *= 0.1  # Make one point closer to origin
    distances = np.linalg.norm(X, axis=1)
    y = np.argmin(distances)  # Target is index of closest point
    return X, y

def generate_4d_data(num_samples=5000, num_points=10):
    """Generates dataset with parallel processing"""
    with Pool() as pool:
        data = pool.starmap(generate_4d_data_sample, 
                          [(num_points,) for _ in range(num_samples)])
    X, y = zip(*data)
    return np.array(X), np.array(y)

# =============================================
# 4D Processing Model
# =============================================
def build_4d_model(num_points=10):
    """Builds model that processes 4D geometric relationships"""
    inputs = tf.keras.Input(shape=(num_points, 4))
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    distance_estimates = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    x = tf.keras.layers.Flatten()(distance_estimates)
    outputs = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# =============================================
# Rotation Invariance Test
# =============================================
def generate_4d_rotation_matrix():
    """Generates random 4D rotation matrix using QR decomposition"""
    random_matrix = np.random.randn(4, 4)
    q, r = np.linalg.qr(random_matrix)
    q *= np.sign(np.diag(r))  # QR decomposition sign convention
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)

def rotate_4d_data(X, rotation_matrix):
    """Applies 4D rotation to dataset"""
    # Fix einsum indices to handle (samples, points, coordinates) format
    return np.einsum('ij,spj->spi', rotation_matrix, X)  # Changed from 'ij,sjk->sik' to 'ij,spj->spi'

def test_4d_rotation_invariance(model, X_test, y_test):
    """Tests model's invariance to 4D rotations"""
    rotation_accuracies = []
    for _ in range(10):
        rot_matrix = generate_4d_rotation_matrix()
        X_rotated = rotate_4d_data(X_test, rot_matrix)
        _, acc = model.evaluate(X_rotated, y_test, verbose=0)
        rotation_accuracies.append(acc)
    
    avg_rot_acc = np.mean(rotation_accuracies)
    std_rot_acc = np.std(rotation_accuracies)
    _, original_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nRotation Test Results:")
    print(f"Original Accuracy: {original_acc*100:.1f}%")
    print(f"Rotated Accuracy: {avg_rot_acc*100:.1f}% ±{std_rot_acc*100:.1f}%")
    assert np.isclose(original_acc, avg_rot_acc, atol=0.05), "Rotation invariance failed!"

# =============================================
# Main Experiment
# =============================================
def run_experiment():
    # Generate and split data
    X, y = generate_4d_data(num_samples=5000, num_points=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Build and train model
    model = build_4d_model(num_points=X.shape[1])
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # Run critical test
    test_4d_rotation_invariance(model, X_test, y_test)

if __name__ == "__main__":
    run_experiment()