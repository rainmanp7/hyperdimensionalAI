import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

# =============================================
# Data Generation Functions
# =============================================
def generate_4d_data_sample(num_points=10):
    """Generate a single 4D data sample with correct label type"""
    X = np.random.randn(num_points, 4) * 2
    closest_idx = np.random.randint(num_points)
    X[closest_idx] *= 0.1  # Make one point clearly closer
    distances = np.linalg.norm(X, axis=1)
    y = int(np.argmin(distances))  # Convert to native int
    return X, y

def generate_4d_data(num_samples=5000, num_points=10):
    """Generate batch of 4D data samples"""
    with Pool() as pool:
        data = pool.starmap(generate_4d_data_sample, [(num_points,) for _ in range(num_samples)])
    X, y = zip(*data)
    return np.array(X), np.array(y)

# =============================================
# Neural Network Model
# =============================================
def build_4d_model(num_points=10):
    """Build optimized 4D processing model"""
    inputs = tf.keras.Input(shape=(num_points, 4))
    
    # Feature extraction layers
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu'))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))(x)
    
    # Distance estimation head
    distance_estimates = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(x)
    x = tf.keras.layers.Flatten()(distance_estimates)
    
    # Classification layer
    outputs = tf.keras.layers.Dense(num_points, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =============================================
# Training Process
# =============================================
def run_experiment():
    """Optimized training process with callbacks"""
    X, y = generate_4d_data(num_samples=10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = build_4d_model()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        ]
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {acc*100:.1f}%")
    print(f"Random Baseline: {100/10}%")

# =============================================
# Unit Tests
# =============================================
class TestDataGeneration(unittest.TestCase):
    def test_generate_4d_data_sample_shape(self):
        num_points = 7
        X, y = generate_4d_data_sample(num_points)
        self.assertEqual(X.shape, (num_points, 4))
        self.assertIsInstance(y, int)

    def test_sample_label_correctness(self):
        for _ in range(100):  # Test multiple samples
            X, y = generate_4d_data_sample()
            distances = np.linalg.norm(X, axis=1)
            self.assertEqual(y, np.argmin(distances))

    def test_generate_4d_data_shape(self):
        X, y = generate_4d_data(num_samples=20, num_points=8)
        self.assertEqual(X.shape, (20, 8, 4))
        self.assertEqual(y.shape, (20,))

    def test_label_range(self):
        X, y = generate_4d_data(num_samples=50, num_points=15)
        self.assertTrue(all(0 <= label < 15 for label in y))

class TestModel(unittest.TestCase):
    def test_model_structure(self):
        model = build_4d_model(num_points=12)
        self.assertEqual(model.input_shape, (None, 12, 4))
        self.assertEqual(model.output_shape, (None, 12))

    def test_model_compilation(self):
        model = build_4d_model()
        self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
        self.assertEqual(model.loss, 'sparse_categorical_crossentropy')

class TestExperiment(unittest.TestCase):
    @patch('__main__.generate_4d_data')
    @patch('tensorflow.keras.Model.fit')
    def test_run_experiment(self, mock_fit, mock_generate):
        mock_X = np.random.randn(10, 10, 4)
        mock_y = np.random.randint(0, 10, size=10)
        mock_generate.return_value = (mock_X, mock_y)
        mock_fit.return_value = tf.keras.callbacks.History()
        
        run_experiment()
        mock_generate.assert_called_once()
        mock_fit.assert_called_once()

# =============================================
# Execution Control
# =============================================
if __name__ == "__main__":
    # Run tests first
    unittest.main(argv=[''], exit=False)
    
    # Then run the actual experiment
    run_experiment()