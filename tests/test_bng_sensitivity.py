import unittest
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np


# Import necessary components from your library
# Assuming your library is installed or accessible via PYTHONPATH
from bngmetric.core_calculations import (
    calculate_total_bng_from_jax_arrays,
    calculate_batched_baseline_bng_units,
    calculate_baseline_bng_unit
)
from bngmetric.constants import HABITAT_TYPE_TO_ID, CONDITION_CATEGORY_TO_ID, ID_TO_DISTINCTIVENESS_VALUE, CONDITION_MULTIPLIERS_MATRIX

# --- Setup Test Data ---
# These remain as JAX arrays, as differentiation happens on these numerical inputs.
# Ensure these match values from your constants.py (e.g., Distinctiveness Values, Condition Multipliers)

# Example values based on previous discussions:
# Distinctiveness: High=6, Low=2
# Condition Multipliers (assuming for test): Good=3.0, Moderate=2.0

GOOD_M = 3.0
MODERATE_M = 2.0

# Define numerical IDs for habitats and conditions for consistent testing
HABITAT_IDS_HIGH_DISTINCTIVENESS = HABITAT_TYPE_TO_ID['Grassland - Lowland calcareous grassland']
HABITAT_IDS_LOW_DISTINCTIVENESS = HABITAT_TYPE_TO_ID['Grassland - Modified grassland']
HABITAT_IDS_WOODLAND = HABITAT_TYPE_TO_ID['Woodland and forest - Lowland mixed deciduous woodland']

CONDITION_ID_GOOD = CONDITION_CATEGORY_TO_ID['Good']
CONDITION_ID_MODERATE = CONDITION_CATEGORY_TO_ID['Moderate']

# Input arrays for the batched calculation
HABITAT_IDS_BATCH = jnp.array([
    HABITAT_IDS_HIGH_DISTINCTIVENESS,
    HABITAT_IDS_LOW_DISTINCTIVENESS,
    HABITAT_IDS_WOODLAND
])

CONDITION_IDS_BATCH = jnp.array([
    CONDITION_ID_GOOD,
    CONDITION_ID_MODERATE,
    CONDITION_ID_GOOD
])

AREAS_BATCH = jnp.array([5.0, 20.0, 8.0], dtype=jnp.float32)
STRATEGIC_MULTIPLIERS_BATCH = jnp.array([1.0, 1.0, 1.15], dtype=jnp.float32)

# Calculate expected base BNG units for reference
# This calculation needs to use the actual distinctiveness and condition multiplier values
# as loaded and ordered in your constants.py
P1_DIST = ID_TO_DISTINCTIVENESS_VALUE[HABITAT_IDS_HIGH_DISTINCTIVENESS]
P1_COND_M = CONDITION_MULTIPLIERS_MATRIX[HABITAT_IDS_HIGH_DISTINCTIVENESS, CONDITION_ID_GOOD]

P2_DIST = ID_TO_DISTINCTIVENESS_VALUE[HABITAT_IDS_LOW_DISTINCTIVENESS]
P2_COND_M = CONDITION_MULTIPLIERS_MATRIX[HABITAT_IDS_LOW_DISTINCTIVENESS, CONDITION_ID_MODERATE]

P3_DIST = ID_TO_DISTINCTIVENESS_VALUE[HABITAT_IDS_WOODLAND]
P3_COND_M = CONDITION_MULTIPLIERS_MATRIX[HABITAT_IDS_WOODLAND, CONDITION_ID_GOOD]


EXPECTED_P1_UNITS = AREAS_BATCH[0] * P1_DIST * P1_COND_M * STRATEGIC_MULTIPLIERS_BATCH[0]
EXPECTED_P2_UNITS = AREAS_BATCH[1] * P2_DIST * P2_COND_M * STRATEGIC_MULTIPLIERS_BATCH[1]
EXPECTED_P3_UNITS = AREAS_BATCH[2] * P3_DIST * P3_COND_M * STRATEGIC_MULTIPLIERS_BATCH[2]

TOTAL_EXPECTED_BNG = EXPECTED_P1_UNITS + EXPECTED_P2_UNITS + EXPECTED_P3_UNITS


class TestBNGSensitivity(unittest.TestCase):

    def test_gradient_wrt_area(self):
        # Differentiate calculate_total_bng_from_jax_arrays wrt 'areas' (3rd argument, index 2)
        grad_fn_area = jax.grad(calculate_total_bng_from_jax_arrays, argnums=2)
        
        # Compute the gradient
        gradients = grad_fn_area(
            HABITAT_IDS_BATCH, CONDITION_IDS_BATCH, AREAS_BATCH, STRATEGIC_MULTIPLIERS_BATCH
        )

        # Expected gradients for area: Distinctiveness * Condition_Multiplier * Strategic_Multiplier
        # dBNG/dArea_P1 = P1_DIST * P1_COND_M * 1.0
        # dBNG/dArea_P2 = P2_DIST * P2_COND_M * 1.0
        # dBNG/dArea_P3 = P3_DIST * P3_COND_M * 1.15

        expected_gradients_area = jnp.array([
            P1_DIST * P1_COND_M * STRATEGIC_MULTIPLIERS_BATCH[0],
            P2_DIST * P2_COND_M * STRATEGIC_MULTIPLIERS_BATCH[1],
            P3_DIST * P3_COND_M * STRATEGIC_MULTIPLIERS_BATCH[2]
        ], dtype=jnp.float32)

        np.testing.assert_allclose(gradients, expected_gradients_area, atol=1e-6)
        print(f"\nGradient wrt Area: {gradients}")


    def test_gradient_wrt_strategic_multiplier(self):
        # Differentiate wrt 'strategic_multipliers' (4th argument, index 3)
        grad_fn_strat = jax.grad(calculate_total_bng_from_jax_arrays, argnums=3)
        
        gradients = grad_fn_strat(
            HABITAT_IDS_BATCH, CONDITION_IDS_BATCH, AREAS_BATCH, STRATEGIC_MULTIPLIERS_BATCH
        )

        # Expected gradients for strategic_multipliers: Area * Distinctiveness * Condition_Multiplier
        # dBNG/dStrat_P1 = 5.0 * P1_DIST * P1_COND_M
        # dBNG/dStrat_P2 = 20.0 * P2_DIST * P2_COND_M
        # dBNG/dStrat_P3 = 8.0 * P3_DIST * P3_COND_M

        expected_gradients_strat = jnp.array([
            AREAS_BATCH[0] * P1_DIST * P1_COND_M,
            AREAS_BATCH[1] * P2_DIST * P2_COND_M,
            AREAS_BATCH[2] * P3_DIST * P3_COND_M
        ], dtype=jnp.float32)

        np.testing.assert_allclose(gradients, expected_gradients_strat, atol=1e-6)
        print(f"Gradient wrt Strategic Multiplier: {gradients}")


    def test_value_and_gradient(self):
        # Test getting both value and gradient for 'areas'
        value, gradients = jax.value_and_grad(calculate_total_bng_from_jax_arrays, argnums=2)(
            HABITAT_IDS_BATCH, CONDITION_IDS_BATCH, AREAS_BATCH, STRATEGIC_MULTIPLIERS_BATCH
        )
        
        self.assertAlmostEqual(value, float(TOTAL_EXPECTED_BNG), places=3)
        
        expected_gradients_area = jnp.array([
            P1_DIST * P1_COND_M * STRATEGIC_MULTIPLIERS_BATCH[0],
            P2_DIST * P2_COND_M * STRATEGIC_MULTIPLIERS_BATCH[1],
            P3_DIST * P3_COND_M * STRATEGIC_MULTIPLIERS_BATCH[2]
        ], dtype=jnp.float32)
        
        np.testing.assert_allclose(gradients, expected_gradients_area, atol=1e-3)
        print(f"Value and Gradient (Area): Value={value}, Gradients={gradients}")


if __name__ == '__main__':
    unittest.main()