import unittest
import jax
import jax.numpy as jnp
import numpy as np  # For creating NumPy arrays for comparison
from src.bngmetric.core_calculations import (
    get_distinctiveness_value,
    get_condition_multiplier,
    calculate_baseline_bng_unit,
    calculate_batched_baseline_bng_units,
    calculate_bng_from_dataframe
)
from src.bngmetric.constants import (
    HABITAT_TYPE_TO_ID,
    ID_TO_DISTINCTIVENESS_VALUE,
    CONDITION_CATEGORY_TO_ID,
    CONDITION_MULTIPLIERS_MATRIX
)
import pandas as pd

class TestCoreCalculations(unittest.TestCase):

    def test_jax_working(self):
        """Simple test to ensure JAX is working correctly."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = a + b
        self.assertTrue(jnp.allclose(result, jnp.array([5.0, 7.0, 9.0])))

    def test_get_distinctiveness_value(self):
        """Test retrieval of distinctiveness value."""
        habitat_id = 1  # Example habitat ID
        expected_value = ID_TO_DISTINCTIVENESS_VALUE[habitat_id]
        actual_value = get_distinctiveness_value(habitat_id)
        self.assertEqual(actual_value, expected_value)

    def test_get_condition_multiplier(self):
        """Test retrieval of condition multiplier."""
        habitat_id = 2
        condition_id = 1
        expected_multiplier = CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]
        actual_multiplier = get_condition_multiplier(habitat_id, condition_id)
        self.assertEqual(actual_multiplier, expected_multiplier)

    def test_calculate_baseline_bng_unit(self):
        """Test calculation of baseline BNG unit."""
        habitat_id = 3
        condition_id = 2
        area = 1.5
        strategic_multiplier = 0.8
        distinctiveness = ID_TO_DISTINCTIVENESS_VALUE[habitat_id]
        condition_m = CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]
        expected_units = area * distinctiveness * condition_m * strategic_multiplier
        actual_units = calculate_baseline_bng_unit(habitat_id, condition_id, area, strategic_multiplier)
        self.assertAlmostEqual(actual_units, expected_units)

    def test_calculate_batched_baseline_bng_units(self):
        """Test batched calculation of baseline BNG units."""
        habitat_ids = jnp.array([1, 2, 3])
        condition_ids = jnp.array([0, 1, 2])
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([0.7, 0.9, 0.8])

        # Calculate expected units using the single-parcel function
        expected_units = jnp.array([
            calculate_baseline_bng_unit(h, c, a, s)
            for h, c, a, s in zip(habitat_ids, condition_ids, areas, strategic_multipliers)
        ])

        actual_units = calculate_batched_baseline_bng_units(habitat_ids, condition_ids, areas, strategic_multipliers)
        self.assertTrue(jnp.allclose(actual_units, expected_units))

    def test_calculate_bng_from_dataframe(self):
        """Test calculation of total BNG from a Pandas DataFrame."""
        data = {
            'Habitat': ['Woodland and forest - Upland birchwoods', 'Woodland and forest - Upland birchwoods', 'Woodland and forest - Upland birchwoods'],
            'Condition': ['Poor', 'Moderate', 'Good'],
            'Area': [1.0, 2.0, 1.5],
            'Strategic_Significance': [0.7, 0.9, 0.8]
        }
        df = pd.DataFrame(data)

        # Calculate expected units manually (using the mappings)
        expected_units = 0.0
        for index, row in df.iterrows():
            habitat_id = HABITAT_TYPE_TO_ID[row['Habitat']]
            condition_id = CONDITION_CATEGORY_TO_ID[row['Condition']]
            area = row['Area']
            strategic_multiplier = row['Strategic_Significance']
            distinctiveness = ID_TO_DISTINCTIVENESS_VALUE[habitat_id]
            condition_m = CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]
            expected_units += area * distinctiveness * condition_m * strategic_multiplier

        actual_units = calculate_bng_from_dataframe(df)
        self.assertAlmostEqual(actual_units, expected_units)

if __name__ == '__main__':
    unittest.main()