import numpy as np
import tensorflow as tf

# ==============================================
# Dynamic Element Profiles with Adaptivity
# ==============================================
element_profiles = {
    'HCl': {'base_ph': -0.5, 'tau_decay': 0.1},
    'NaOH': {'base_ph': 14.0, 'tau_decay': 0.1},
    'NaCl': {'solubility': 360.0, 'tau_adapt': 0.2},
    'H2O': {'neutral_ph': 7.0, 'phase_adapt': 0.15}
}

class ChemicalEntity:
    def __init__(self, name, concentration):
        self.name = name
        self.concentration = max(concentration, 0.01)  # Prevent zero-lock
        self.tau = np.random.uniform(-0.1, 0.1)  # Initial randomness
        self.profile = element_profiles[name]
        self.history = []
        
    def dynamic_adjust(self, system_ph):
        """Self-adapting tau adjustment with system feedback"""
        # Dynamic sensitivity based on system state
        sensitivity = 0.2 * (1 + abs(system_ph - 7.0))
        delta = np.random.normal(0, 0.05) * sensitivity
        
        # Add decay from previous state
        if self.history:
            delta -= self.profile.get('tau_decay', 0.1) * self.history[-1]
            
        self.tau = np.clip(self.tau + delta, -1.0, 1.0)
        self.history.append(self.tau)
        return self.tau

# ==============================================
# Quantum Field System with Noise Injection
# ==============================================
class QuantumFieldSystem:
    def __init__(self):
        self.orbital_base = {
            'H': [-13.6], 
            'Cl': [-10.62, -8.11, -5.43],
            'Na': [-5.43, -3.62, -1.21],
            'O': [-13.61, -1.36, -0.54]
        }
        self.noise_level = 0.05
        
    def dynamic_fields(self, entities):
        """Generate fields with adaptive noise and feedback"""
        field_impact = sum(e.tau * e.concentration for e in entities)
        noise = np.random.normal(0, self.noise_level)
        
        return {
            'H': [e * (1 + 0.1 * field_impact + noise) for e in self.orbital_base['H']],
            'Cl': [e * (1 - 0.05 * field_impact + noise) for e in self.orbital_base['Cl']],
            'Na': [e * (1 + 0.15 * field_impact + noise) for e in self.orbital_base['Na']],
            'O': [e * (1 + 0.02 * (field_impact + noise)) for e in self.orbital_base['O']]
        }

# ==============================================
# Adaptive Neural Coordinator
# ==============================================
class AdaptiveCoordinator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='swish')
        self.dense2 = tf.keras.layers.Dense(16, activation='swish')
        self.tau_output = tf.keras.layers.Dense(1, activation='tanh')
        self.adaptation_layer = tf.keras.layers.Dense(8, activation='relu')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.adaptation_layer(x)
        x = self.dense2(x + tf.random.normal(tf.shape(x), stddev=0.1))  # Noise injection
        return self.tau_output(x)

# ==============================================
# Reactive Environment with Forced Dynamics
# ==============================================
class ReactiveEnvironment:
    def __init__(self):
        self.quantum = QuantumFieldSystem()
        self.coordinator = AdaptiveCoordinator()
        self.coordinator.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        
    def run_dynamic_simulation(self, entities, cycles=5):
        results = []
        previous_ph = 7.0
        
        for cycle in range(cycles):
            # Entity self-adaptation
            for entity in entities:
                entity.dynamic_adjust(previous_ph)
                
            # Neural coordination with noise
            inputs = self._build_dynamic_inputs(entities)
            global_tau = self.coordinator.predict(inputs, verbose=0)[0][0]
            
            # System updates
            ph = self._calculate_adaptive_ph(entities)
            rate = self._calculate_reaction_rate(entities, global_tau)
            orbitals = self.quantum.dynamic_fields(entities)
            
            # Force minimum activity
            if abs(rate) < 0.01:
                rate = np.random.uniform(0.1, 0.5) * np.sign(rate)
                
            results.append({
                'cycle': cycle,
                'ph': ph,
                'global_tau': global_tau,
                'reaction_rate': rate,
                'orbitals': orbitals
            })
            previous_ph = ph
            
        return results

    def _build_dynamic_inputs(self, entities):
        return np.array([[
            e.concentration, 
            e.tau, 
            np.random.normal(0, 0.1)  # Input noise
        ] for e in entities]).flatten().reshape(1, -1)

    def _calculate_adaptive_ph(self, entities):
        ph_values = []
        for e in entities:
            base = e.profile.get('base_ph', 7.0)
            ph = base + e.tau * (1 + e.concentration)
            ph_values.append(ph)
        return np.clip(np.mean(ph_values), 0.0, 14.0)

    def _calculate_reaction_rate(self, entities, global_tau):
        rate = 1.0
        for e in entities:
            rate *= e.concentration * (0.1 + abs(e.tau))  # Prevent zero
        return rate * np.sin(global_tau * np.pi)  # Oscillatory component

# ==============================================
# Enhanced Testing Framework
# ==============================================
def run_adaptive_test():
    # Initialize with safe concentrations
    hcl = ChemicalEntity('HCl', 1.2)
    naoh = ChemicalEntity('NaOH', 1.0)
    nacl = ChemicalEntity('NaCl', 0.1)  # Initial product
    h2o = ChemicalEntity('H2O', 0.1)
    
    environment = ReactiveEnvironment()
    results = environment.run_dynamic_simulation([hcl, naoh, nacl, h2o], cycles=5)
    
    print("Dynamic 4D Chemical Simulation Report\n")
    for result in results:
        print(f"Cycle {result['cycle']}:")
        print(f"→ System pH: {result['ph']:.2f}")
        print(f"→ Global τ: {result['global_tau']:.3f}")
        print(f"→ Reaction Rate: {result['reaction_rate']:.2f}x baseline")
        print("  Quantum Orbital Energies:")
        for element, energies in result['orbitals'].items():
            energy_str = [f"{e:+.2f}" for e in energies]
            print(f"    {element}: {energy_str}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_adaptive_test()