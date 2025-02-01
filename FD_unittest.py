import unittest
import numpy as np
import tensorflow as tf
from FD_Smart_Nanoscale import (  # Replace 'your_module' with the actual module name
    ChemicalEntity, QuantumFieldSystem, AdaptiveCoordinator, ReactiveEnvironment,
    element_profiles
)

class TestChemicalEntity(unittest.TestCase):
    def test_init(self):
        entity = ChemicalEntity('HCl', 1.2)
        self.assertEqual(entity.name, 'HCl')
        self.assertEqual(entity.concentration, 1.2)
        self.assertLess(abs(entity.tau), 0.1)  # Initial tau randomness
        self.assertEqual(entity.history, [])

    def test_dynamic_adjust(self):
        entity = ChemicalEntity('HCl', 1.2)
        initial_tau = entity.tau
        entity.dynamic_adjust(7.0)
        self.assertNotEqual(entity.tau, initial_tau)  # Tau should change
        self.assertEqual(len(entity.history), 1)  # History should update

    def test_concentration_floor(self):
        entity = ChemicalEntity('HCl', 0.0)
        self.assertEqual(entity.concentration, 0.01)  # Should be floored at 0.01

class TestQuantumFieldSystem(unittest.TestCase):
    def test_init(self):
        qfs = QuantumFieldSystem()
        self.assertIsNotNone(qfs.orbital_base)
        self.assertEqual(qfs.noise_level, 0.05)

    def test_dynamic_fields(self):
        qfs = QuantumFieldSystem()
        entities = [ChemicalEntity('HCl', 1.2), ChemicalEntity('NaOH', 1.0)]
        fields = qfs.dynamic_fields(entities)
        self.assertIn('H', fields)
        self.assertIn('Cl', fields)
        self.assertIn('Na', fields)
        self.assertIn('O', fields)
        for element, energies in fields.items():
            self.assertIsInstance(energies, list)
            for energy in energies:
                self.assertIsInstance(energy, float)

class TestAdaptiveCoordinator(unittest.TestCase):
    def test_init(self):
        ac = AdaptiveCoordinator()
        self.assertIsInstance(ac.dense1, tf.keras.layers.Dense)
        self.assertIsInstance(ac.tau_output, tf.keras.layers.Dense)
        self.assertIsInstance(ac.adaptation_layer, tf.keras.layers.Dense)

    def test_call(self):
        ac = AdaptiveCoordinator()
        inputs = np.array([[1.2, 0.05, 0.1]])  # Example input
        output = ac.call(inputs)
        self.assertEqual(output.shape, (1, 1))  # Output shape
        self.assertLessEqual(output.numpy()[0][0], 1.0)  # Tanh output
        self.assertGreaterEqual(output.numpy()[0][0], -1.0)

class TestReactiveEnvironment(unittest.TestCase):
    def test_init(self):
        re = ReactiveEnvironment()
        self.assertIsInstance(re.quantum, QuantumFieldSystem)
        self.assertIsInstance(re.coordinator, AdaptiveCoordinator)

    def test_run_dynamic_simulation(self):
        re = ReactiveEnvironment()
        entities = [ChemicalEntity('HCl', 1.2), ChemicalEntity('NaOH', 1.0)]
        results = re.run_dynamic_simulation(entities, cycles=3)
        self.assertEqual(len(results), 3)  # Number of cycles
        for result in results:
            self.assertIn('cycle', result)
            self.assertIn('ph', result)
            self.assertIn('global_tau', result)
            self.assertIn('reaction_rate', result)
            self.assertIn('orbitals', result)

    def test_build_dynamic_inputs(self):
        re = ReactiveEnvironment()
        entities = [ChemicalEntity('HCl', 1.2), ChemicalEntity('NaOH', 1.0)]
        inputs = re._build_dynamic_inputs(entities)
        self.assertEqual(inputs.shape[0], 1)  # Batch size
        self.assertEqual(inputs.shape[1], len(entities) * 3)  # Features (conc, tau, noise) per entity

    def test_calculate_adaptive_ph(self):
        re = ReactiveEnvironment()
        entities = [ChemicalEntity('HCl', 1.2), ChemicalEntity('NaOH', 1.0)]
        ph = re._calculate_adaptive_ph(entities)
        self.assertLessEqual(ph, 14.0)  # pH upper limit
        self.assertGreaterEqual(ph, 0.0)  # pH lower limit

    def test_calculate_reaction_rate(self):
        re = ReactiveEnvironment()
        entities = [ChemicalEntity('HCl', 1.2), ChemicalEntity('NaOH', 1.0)]
        rate = re._calculate_reaction_rate(entities, 0.5)  # Example global tau
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.0)  # Non-negative rate

if __name__ == '__main__':
    unittest.main()
