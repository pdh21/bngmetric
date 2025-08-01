import unittest
import pandas as pd
import numpy as np

import jax.numpy as jnp
from bngmetric.constants import (
    HABITAT_DISTINCTIVENESS_MAP,
    HABITAT_TYPE_TO_ID,
    ID_TO_DISTINCTIVENESS_VALUE,
    CONDITION_CATEGORIES_FOR_MATRIX,
    CONDITION_MULTIPLIERS_MATRIX,
    CONDITION_CATEGORY_TO_ID
)


class TestConstants(unittest.TestCase):
    def test_habitat_distinctiveness_map_exists(self):
        """Test that HABITAT_DISTINCTIVENESS_MAP is properly loaded."""
        self.assertIsInstance(HABITAT_DISTINCTIVENESS_MAP, dict)
        self.assertGreater(len(HABITAT_DISTINCTIVENESS_MAP), 0)
        
        # Test a sample of values to ensure they are numeric
        for habitat, score in list(HABITAT_DISTINCTIVENESS_MAP.items())[:5]:
            self.assertIsInstance(habitat, str)
            self.assertIsInstance(score, (int, float))
    
    def test_habitat_type_to_id_consistency(self):
        """Test that HABITAT_TYPE_TO_ID contains all habitats from the distinctiveness map."""
        self.assertEqual(
            set(HABITAT_DISTINCTIVENESS_MAP.keys()),
            set(HABITAT_TYPE_TO_ID.keys())
        )
        # Ensure IDs are unique and sequential
        self.assertEqual(
            set(HABITAT_TYPE_TO_ID.values()),
            set(range(len(HABITAT_TYPE_TO_ID)))
        )
    
    def test_id_to_distinctiveness_value(self):
        """Test that ID_TO_DISTINCTIVENESS_VALUE matches the HABITAT_DISTINCTIVENESS_MAP values."""
        self.assertIsInstance(ID_TO_DISTINCTIVENESS_VALUE, jnp.ndarray)
        self.assertEqual(len(ID_TO_DISTINCTIVENESS_VALUE), len(HABITAT_DISTINCTIVENESS_MAP))
        
        # Convert JAX array to numpy for testing
        values_array = np.array(ID_TO_DISTINCTIVENESS_VALUE)
        
        # Test values match for a few habitats
        for habitat, habitat_id in list(HABITAT_TYPE_TO_ID.items())[:5]:
            expected = HABITAT_DISTINCTIVENESS_MAP[habitat]
            actual = values_array[habitat_id]
            self.assertEqual(expected, actual)
    
    def test_condition_matrix_dimensions(self):
        """Test that condition matrix dimensions align with habitat and condition counts."""
        self.assertEqual(
            CONDITION_MULTIPLIERS_MATRIX.shape, 
            (len(HABITAT_TYPE_TO_ID), len(CONDITION_CATEGORIES_FOR_MATRIX))
        )
    
    def test_condition_category_to_id(self):
        """Test that condition category mapping is correct."""
        self.assertEqual(
            set(CONDITION_CATEGORY_TO_ID.keys()),
            set(CONDITION_CATEGORIES_FOR_MATRIX)
        )
        # Ensure IDs are unique and sequential
        self.assertEqual(
            set(CONDITION_CATEGORY_TO_ID.values()),
            set(range(len(CONDITION_CATEGORY_TO_ID)))
        )


if __name__ == "__main__":
    unittest.main()