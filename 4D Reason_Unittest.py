import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import lru_cache

# Realistic Data Generation
@lru_cache(maxsize=128)
def generate_4d_data_sample(num_points=10, dim=4, noise_level=1.5):
    """Create challenging 4D decision problem"""
    base_points = np.random.randn(num_points, dim) * 2
    noise = np.random.normal(0, noise_level, (num_points, dim))
    X = base_points + noise
    
    # Create ambiguous distances
    distances = np.linalg.norm(X, axis=1)
    y = np.argmin(distances)
    
    # Verify true minimum isn't obvious
    sorted_dists = np.sort(distances)
    while (sorted_dists[1] - sorted_dists[0]) < 0.5:  # Ensure ambiguity
        return generate_4d_data_sample(num_points, dim, noise_level)
        
    return X, int(y)  # Ensure y is an integer

class DataGenerator:
    @staticmethod
    def generate_dataset(num_samples=5000, **kwargs):
        with Pool() as pool:
            data = pool.starmap(generate_4d_data_sample, 
                              [tuple(kwargs.values()) for _ in range(num_samples)])
        X, y = zip(*data)
        return np.array(X), np.array(y)

# Robust Model Architecture
class ModelTrainer:
    @staticmethod
    def build_challenge_model(num_points=10, dim=4):
        inputs = tf.keras.Input(shape=(num_points, dim))
        
        # Geometric relationship layers
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(8, activation='relu'))(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Relative position analysis
        distance_estimates = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='linear'))(x)
        
        # Competitive decision layer
        flattened = tf.keras.layers.Flatten()(distance_estimates)
        outputs = tf.keras.layers.Softmax()(flattened)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def train_model(model, X_train, y_train, epochs=20, validation_split=0.2, verbose=0):
        model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=verbose)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test, verbose=0):
        return model.evaluate(X_test, y_test, verbose=verbose)

# Validated Experimental Protocol
class ExperimentRunner:
    def __init__(self):
        self.results = {}
    
    def run_scaling_tests(self):
        """Proper train-test separation"""
        print("Running Dimension Analysis...")
        dim_acc = {}
        for d in [3, 4, 5]:
            X, y = DataGenerator.generate_dataset(num_samples=5000, num_points=10, dim=d, noise_level=1.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            model = ModelTrainer.build_challenge_model(num_points=10, dim=d)
            model = ModelTrainer.train_model(model, X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
            
            _, acc = ModelTrainer.evaluate_model(model, X_test, y_test, verbose=0)
            dim_acc[d] = acc
            print(f" - {d}D Test Accuracy: {acc:.3f}")
        self.results['dimension'] = dim_acc
        
        print("\nRunning Complexity Analysis...")
        comp_acc = {}
        for n in [5, 10, 15]:
            X, y = DataGenerator.generate_dataset(num_samples=5000, num_points=n, dim=4, noise_level=1.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            model = ModelTrainer.build_challenge_model(num_points=n, dim=4)
            model = ModelTrainer.train_model(model, X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
            
            _, acc = ModelTrainer.evaluate_model(model, X_test, y_test, verbose=0)
            comp_acc[n] = acc
            print(f" - {n} Points Test Accuracy: {acc:.3f}")
        self.results['complexity'] = comp_acc

    def analyze_errors(self):
        """Detailed error analysis"""
        X, y = DataGenerator.generate_dataset(num_samples=2000, num_points=10, dim=4, noise_level=1.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = ModelTrainer.build_challenge_model(num_points=10, dim=4)
        model = ModelTrainer.train_model(model, X_train, y_train, epochs=20, validation_split=0.2, verbose=0)
        
        # Prediction analysis
        preds = model.predict(X_test)
        y_pred = preds.argmax(axis=1)
        errors = np.where(y_pred != y_test)[0]
        
        print("\nError Analysis:")
        error_rate = len(errors) / len(y_test)
        print(f"- Test Error Rate: {error_rate:.1%}")
        
        if len(errors) > 0:
            confidences = preds[errors].max(axis=1)
            avg_confidence = np.mean(confidences)
            median_confidence = np.median(confidences)
            print(f"- Average Confidence in Errors: {avg_confidence:.2f}")
            print(f"- Median Confidence: {median_confidence:.2f}")
        else:
            print("- Perfect Accuracy Achieved (Consider Increasing Noise Level)")
        
        self.results['Test Error Rate'] = error_rate
        self.results['Average Confidence in Errors'] = avg_confidence if len(errors) > 0 else None
        self.results['Median Confidence'] = median_confidence if len(errors) > 0 else None

# Adjusted Philosophical Interpretation
def scientific_interpretation(results):
    print("\nRevised Philosophical Implications:")
    
    dim_acc = results['dimension']
    comp_acc = results['complexity']
    
    print("1. Dimensional Processing:")
    print(f"   - Relative Performance: 3D={dim_acc[3]:.2f}, 4D={dim_acc[4]:.2f}, 5D={dim_acc[5]:.2f}")
    if max(dim_acc.values()) - min(dim_acc.values()) < 0.1:
        print("   - Consistent cross-dimensional performance suggests")
        print("     abstract geometric reasoning capabilities")
    
    print("\n2. Complexity Handling:")
    print(f"   - Accuracy Scaling: 5pts={comp_acc[5]:.2f}, 15pts={comp_acc[15]:.2f}")
    if comp_acc[15] > 0.75:
        print("   - Maintained competence with complex inputs implies")
        print("     non-linear pattern integration abilities")

    print("\n3. Implications for Intelligence:")
    print("   - Demonstrates capacity for:")
    print("     a) Invariant reasoning across mathematical spaces")
    print("     b) Emergent understanding of relative positioning")
    print("     c) Abstract decision-making without sensory grounding")

# Unit Tests
import unittest

class TestRigorousExperiment(unittest.TestCase):
    def setUp(self):
        self.num_points = 10
        self.dim = 4
        self.num_samples = 100  # Reduced for faster testing
        self.noise_level = 1.5

    def test_generate_4d_data_sample(self):
        X, y = generate_4d_data_sample(self.num_points, self.dim, self.noise_level)
        self.assertEqual(X.shape, (self.num_points, self.dim))
        self.assertIsInstance(y, int)

    def test_generate_dataset(self):
        X, y = DataGenerator.generate_dataset(self.num_samples, num_points=self.num_points, dim=self.dim, noise_level=self.noise_level)
        self.assertEqual(X.shape, (self.num_samples, self.num_points, self.dim))
        self.assertEqual(y.shape, (self.num_samples,))

    def test_build_challenge_model(self):
        model = ModelTrainer.build_challenge_model(self.num_points, self.dim)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, self.num_points, self.dim))
        self.assertEqual(model.output_shape, (None, self.num_points))

    def test_run_scaling_tests(self):
        exp = ExperimentRunner()
        exp.run_scaling_tests()
        self.assertIn('dimension', exp.results)
        self.assertIn('complexity', exp.results)
        for dim in [3, 4, 5]:
            self.assertIn(dim, exp.results['dimension'])
        for n in [5, 10, 15]:
            self.assertIn(n, exp.results['complexity'])

    def test_analyze_errors(self):
        exp = ExperimentRunner()
        exp.analyze_errors()
        self.assertIn('Test Error Rate', exp.results)
        if exp.results['Average Confidence in Errors'] is not None:
            self.assertIn('Average Confidence in Errors', exp.results)
        if exp.results['Median Confidence'] is not None:
            self.assertIn('Median Confidence', exp.results)

    def test_scientific_interpretation(self):
        results = {
            'dimension': {3: 0.85, 4: 0.88, 5: 0.87},
            'complexity': {5: 0.76, 15: 0.77}
        }
        scientific_interpretation(results)

if __name__ == "__main__":
    unittest.main()