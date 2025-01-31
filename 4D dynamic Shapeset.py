import numpy as np
import tensorflow as tf
from multiprocessing import Pool, cpu_count, Manager

# ==============================================
# Core 4D Transformation System
# ==============================================
class DimensionalEngine:
    """Handles fundamental 4D scaling rules"""
    def __init__(self):
        self.scaling_bounds = (-0.5, 0.5)  # Allows 50% shrink/expand

    def transform(self, obj_3d, t):
        """Apply 4D scaling while maintaining proportions"""
        return obj_3d * (1 + t)

# ==============================================
# Smart Sampling Agents
# ==============================================
class ShapeAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.memory = []
        self.engine = DimensionalEngine()

    def create_shape_pair(self):
        """Generate 3D object with constrained 4D scaling"""
        base = np.random.uniform(1, 10, 3)
        current_size = np.linalg.norm(base)
        
        # Keep transformations within ±50%
        min_size = current_size * 0.5
        max_size = current_size * 1.5
        target_size = np.random.uniform(min_size, max_size)
        
        t = (target_size - current_size) / current_size
        self.memory.append((base.copy(), t))
        return base, t

    def share_knowledge(self, shared_mem):
        """Blend local and global understanding"""
        if not self.memory:
            return

        # Get wisdom from other agents
        global_ts = [t for _, t in shared_mem]
        if global_ts:
            avg_t = np.mean(global_ts)
            # Adjust personal memories
            self.memory = [
                (obj, 0.6*t + 0.4*avg_t) 
                for obj, t in self.memory
            ]

def agent_work(agent_id, shared_mem, samples=100):
    """Agent process with optimized sampling"""
    agent = ShapeAgent(agent_id)
    for _ in range(samples):
        obj, t = agent.create_shape_pair()
        shared_mem.append((obj, t))
    agent.share_knowledge(shared_mem)
    return agent.memory

# ==============================================
# Neural Transformation Model
# ==============================================
class SizeTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu', dtype=tf.float64)
        self.d2 = tf.keras.layers.Dense(32, activation='relu', dtype=tf.float64)
        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float64)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        return self.output_layer(x)

# ==============================================
# Training and Execution
# ==============================================
def generate_dataset(total_samples=500):
    """Generate collaborative dataset"""
    manager = Manager()
    shared_mem = manager.list()
    num_agents = cpu_count()
    samples_per = total_samples // num_agents
    extra_samples = total_samples % num_agents

    with Pool(num_agents) as pool:
        tasks = [(i, shared_mem, samples_per + (1 if i < extra_samples else 0)) 
                for i in range(num_agents)]
        results = pool.starmap(agent_work, tasks)

    # Combine results
    data = [item for sublist in results for item in sublist]
    X, y = zip(*data)
    return np.array(X), np.array(y).reshape(-1, 1)

def train_model():
    # Data setup
    X, y = generate_dataset()
    print(f"\nGenerated {len(X)} samples")
    print(f"Sample t range: {y.min():.2f} to {y.max():.2f}")

    # Model setup
    model = SizeTransformer()
    model.compile(optimizer='adam', loss='mse')
    
    # Efficient training
    history = model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    
    # Testing
    engine = DimensionalEngine()
    test_shapes = np.array([
        [5.0, 5.0, 5.0],
        [3.0, 4.0, 5.0],
        [8.0, 2.0, 2.0]
    ], dtype=np.float64)

    print("\n=== Transformation Results ===")
    for obj in test_shapes:
        original_norm = np.linalg.norm(obj)
        t = model.predict(obj.reshape(1, -1))[0][0]
        transformed = engine.transform(obj, t)
        new_norm = np.linalg.norm(transformed)
        
        print(f"\nOriginal: {obj}")
        print(f"4D Adjustment: {t:.4f}")
        print(f"Transformed: {transformed.round(2)}")
        print(f"Size Change: {original_norm:.1f} → {new_norm:.1f}")

if __name__ == "__main__":
    print("=== 4D DYNAMIC TRANSFORMATION SYSTEM ===")
    train_model()
    print("\n=== OPERATION COMPLETE ===")