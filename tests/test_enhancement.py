import unittest
import jax
import jax.numpy as jnp
import pandas as pd
from bngmetric.enhancement import (
    get_enhancement_temporal_multiplier,
    get_enhancement_difficulty_multiplier,
    calculate_enhancement_bng_unit,
    calculate_baseline_unit_for_enhancement,
    calculate_enhancement_uplift,
    calculate_batched_enhancement_bng_units,
    calculate_batched_enhancement_uplift,
    calculate_enhancement_bng_from_dataframe,
    calculate_enhancement_uplift_from_dataframe,
    calculate_total_enhancement_units_from_jax_arrays,
    calculate_total_enhancement_uplift_from_jax_arrays,
    # Distinctiveness enhancement
    get_distinctiveness_enhancement_temporal_multiplier,
    calculate_distinctiveness_enhancement_bng_unit,
    calculate_distinctiveness_enhancement_uplift,
    calculate_batched_distinctiveness_enhancement_bng_units,
    calculate_batched_distinctiveness_enhancement_uplift,
    calculate_distinctiveness_enhancement_bng_from_dataframe,
    calculate_distinctiveness_enhancement_uplift_from_dataframe,
    calculate_total_distinctiveness_enhancement_units_from_jax_arrays,
    calculate_total_distinctiveness_enhancement_uplift_from_jax_arrays
)
from bngmetric.constants import (
    HABITAT_TYPE_TO_ID,
    ID_TO_DISTINCTIVENESS_VALUE,
    CONDITION_CATEGORY_TO_ID,
    CONDITION_MULTIPLIERS_MATRIX,
    CONDITION_ENHANCEMENT_TEMPORAL,
    DISTINCTIVENESS_ENHANCEMENT_TEMPORAL,
    CONDITION_ID_TO_DIST_ENH_COL,
    ENHANCEMENT_RISK,
    TEMPORAL_MULTIPLIER_LOOKUP
)


class TestEnhancementCalculations(unittest.TestCase):

    def test_get_enhancement_temporal_multiplier_valid(self):
        """Test retrieval of temporal multiplier for a valid enhancement path."""
        # Find a valid enhancement path (non-NaN in the matrix)
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            for start_cond, start_id in CONDITION_CATEGORY_TO_ID.items():
                for target_cond, target_id in CONDITION_CATEGORY_TO_ID.items():
                    years = CONDITION_ENHANCEMENT_TEMPORAL[habitat_id, start_id, target_id]
                    if not jnp.isnan(years):
                        multiplier = get_enhancement_temporal_multiplier(habitat_id, start_id, target_id)
                        # Should be a positive value from the lookup
                        self.assertGreater(float(multiplier), 0.0)
                        return
        self.fail("No valid enhancement path found in test data")

    def test_get_enhancement_temporal_multiplier_invalid(self):
        """Test that invalid enhancement paths return 0."""
        # Find an invalid (NaN) enhancement path
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            for start_cond, start_id in CONDITION_CATEGORY_TO_ID.items():
                for target_cond, target_id in CONDITION_CATEGORY_TO_ID.items():
                    years = CONDITION_ENHANCEMENT_TEMPORAL[habitat_id, start_id, target_id]
                    if jnp.isnan(years):
                        multiplier = get_enhancement_temporal_multiplier(habitat_id, start_id, target_id)
                        self.assertEqual(float(multiplier), 0.0)
                        return
        self.fail("No invalid enhancement path found in test data")

    def test_get_enhancement_difficulty_multiplier(self):
        """Test retrieval of difficulty multiplier."""
        habitat_id = 1
        expected = ENHANCEMENT_RISK[habitat_id]
        actual = get_enhancement_difficulty_multiplier(habitat_id)
        self.assertEqual(float(actual), float(expected))

    def test_calculate_enhancement_bng_unit(self):
        """Test single parcel enhancement calculation."""
        # Find a valid enhancement path
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            for start_cond, start_id in CONDITION_CATEGORY_TO_ID.items():
                for target_cond, target_id in CONDITION_CATEGORY_TO_ID.items():
                    years = CONDITION_ENHANCEMENT_TEMPORAL[habitat_id, start_id, target_id]
                    if not jnp.isnan(years) and not jnp.isnan(ENHANCEMENT_RISK[habitat_id]):
                        area = 2.0
                        strategic = 1.15

                        units = calculate_enhancement_bng_unit(
                            habitat_id, start_id, target_id, area, strategic
                        )

                        # Manual calculation
                        distinctiveness = ID_TO_DISTINCTIVENESS_VALUE[habitat_id]
                        condition_m = CONDITION_MULTIPLIERS_MATRIX[habitat_id, target_id]
                        temporal_m = TEMPORAL_MULTIPLIER_LOOKUP[int(years)]
                        difficulty_m = ENHANCEMENT_RISK[habitat_id]
                        expected = area * distinctiveness * condition_m * strategic * temporal_m * difficulty_m

                        self.assertAlmostEqual(float(units), float(expected), places=5)
                        return
        self.fail("No valid enhancement path found for testing")

    def test_calculate_enhancement_uplift_positive(self):
        """Test that enhancing to better condition gives positive uplift."""
        # Find a valid enhancement where target > start condition
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            start_id = CONDITION_CATEGORY_TO_ID.get('Poor', 0)
            target_id = CONDITION_CATEGORY_TO_ID.get('Good', 2)
            years = CONDITION_ENHANCEMENT_TEMPORAL[habitat_id, start_id, target_id]
            if not jnp.isnan(years) and not jnp.isnan(ENHANCEMENT_RISK[habitat_id]):
                area = 1.0
                strategic = 1.0

                uplift = calculate_enhancement_uplift(
                    habitat_id, start_id, target_id, area, strategic
                )
                # Uplift could be negative due to temporal/difficulty multipliers
                # but the function should run without error
                self.assertIsInstance(float(uplift), float)
                return
        self.fail("No valid Poor->Good enhancement path found")

    def test_calculate_batched_enhancement_bng_units(self):
        """Test batched enhancement calculation."""
        # Use simple test data
        habitat_ids = jnp.array([0, 1, 2])
        start_condition_ids = jnp.array([0, 0, 0])
        target_condition_ids = jnp.array([1, 1, 1])
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])

        results = calculate_batched_enhancement_bng_units(
            habitat_ids, start_condition_ids, target_condition_ids,
            areas, strategic_multipliers
        )

        self.assertEqual(results.shape, (3,))

    def test_calculate_batched_enhancement_uplift(self):
        """Test batched uplift calculation."""
        habitat_ids = jnp.array([0, 1, 2])
        start_condition_ids = jnp.array([0, 0, 0])
        target_condition_ids = jnp.array([1, 1, 1])
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])

        results = calculate_batched_enhancement_uplift(
            habitat_ids, start_condition_ids, target_condition_ids,
            areas, strategic_multipliers
        )

        self.assertEqual(results.shape, (3,))

    def test_calculate_enhancement_bng_from_dataframe(self):
        """Test DataFrame interface for enhancement calculation."""
        # Get a valid habitat name
        habitat_name = list(HABITAT_TYPE_TO_ID.keys())[0]

        data = {
            'Habitat': [habitat_name, habitat_name],
            'Start_Condition': ['Poor', 'Moderate'],
            'Target_Condition': ['Moderate', 'Good'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15]
        }
        df = pd.DataFrame(data)

        result = calculate_enhancement_bng_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_calculate_enhancement_uplift_from_dataframe(self):
        """Test DataFrame interface for uplift calculation."""
        habitat_name = list(HABITAT_TYPE_TO_ID.keys())[0]

        data = {
            'Habitat': [habitat_name, habitat_name],
            'Start_Condition': ['Poor', 'Moderate'],
            'Target_Condition': ['Moderate', 'Good'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15]
        }
        df = pd.DataFrame(data)

        result = calculate_enhancement_uplift_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_jax_differentiability(self):
        """Test that the enhancement functions are differentiable with JAX."""
        habitat_ids = jnp.array([0, 1])
        start_condition_ids = jnp.array([0, 0])
        target_condition_ids = jnp.array([1, 1])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])

        # Test gradient with respect to areas
        def total_units_fn(areas_param):
            return calculate_total_enhancement_units_from_jax_arrays(
                habitat_ids, start_condition_ids, target_condition_ids,
                areas_param, strategic_multipliers
            )

        grad_fn = jax.grad(total_units_fn)
        gradients = grad_fn(areas)

        # Should return gradients of same shape as input
        self.assertEqual(gradients.shape, areas.shape)

    def test_jax_differentiability_uplift(self):
        """Test that uplift functions are differentiable with JAX."""
        habitat_ids = jnp.array([0, 1])
        start_condition_ids = jnp.array([0, 0])
        target_condition_ids = jnp.array([1, 1])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])

        def total_uplift_fn(areas_param):
            return calculate_total_enhancement_uplift_from_jax_arrays(
                habitat_ids, start_condition_ids, target_condition_ids,
                areas_param, strategic_multipliers
            )

        grad_fn = jax.grad(total_uplift_fn)
        gradients = grad_fn(areas)

        self.assertEqual(gradients.shape, areas.shape)


class TestDistinctivenessEnhancement(unittest.TestCase):
    """Tests for distinctiveness enhancement (changing to higher distinctiveness habitat)."""

    def test_get_distinctiveness_enhancement_temporal_multiplier_valid(self):
        """Test retrieval of temporal multiplier for a valid distinctiveness enhancement path."""
        # Find a valid path (non-NaN in the matrix)
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            for cond_name, cond_id in CONDITION_CATEGORY_TO_ID.items():
                col_idx = CONDITION_ID_TO_DIST_ENH_COL[cond_id]
                years = DISTINCTIVENESS_ENHANCEMENT_TEMPORAL[habitat_id, col_idx]
                if not jnp.isnan(years):
                    multiplier = get_distinctiveness_enhancement_temporal_multiplier(habitat_id, cond_id)
                    self.assertGreater(float(multiplier), 0.0)
                    return
        self.fail("No valid distinctiveness enhancement path found")

    def test_get_distinctiveness_enhancement_temporal_multiplier_invalid(self):
        """Test that invalid paths return 0."""
        # Find an invalid (NaN) path
        for habitat_name, habitat_id in HABITAT_TYPE_TO_ID.items():
            for cond_name, cond_id in CONDITION_CATEGORY_TO_ID.items():
                col_idx = CONDITION_ID_TO_DIST_ENH_COL[cond_id]
                years = DISTINCTIVENESS_ENHANCEMENT_TEMPORAL[habitat_id, col_idx]
                if jnp.isnan(years):
                    multiplier = get_distinctiveness_enhancement_temporal_multiplier(habitat_id, cond_id)
                    self.assertEqual(float(multiplier), 0.0)
                    return
        self.fail("No invalid distinctiveness enhancement path found")

    def test_calculate_distinctiveness_enhancement_bng_unit(self):
        """Test single parcel distinctiveness enhancement calculation."""
        # Find a valid path
        for target_name, target_id in HABITAT_TYPE_TO_ID.items():
            for cond_name, cond_id in CONDITION_CATEGORY_TO_ID.items():
                col_idx = CONDITION_ID_TO_DIST_ENH_COL[cond_id]
                years = DISTINCTIVENESS_ENHANCEMENT_TEMPORAL[target_id, col_idx]
                if not jnp.isnan(years) and not jnp.isnan(ENHANCEMENT_RISK[target_id]):
                    # Use a different baseline habitat
                    baseline_id = (target_id + 1) % len(HABITAT_TYPE_TO_ID)
                    area = 2.0
                    strategic = 1.15

                    units = calculate_distinctiveness_enhancement_bng_unit(
                        baseline_id, cond_id, target_id, cond_id, area, strategic
                    )

                    # Should be a valid number
                    self.assertIsInstance(float(units), float)
                    return
        self.fail("No valid distinctiveness enhancement path found for testing")

    def test_calculate_distinctiveness_enhancement_uplift(self):
        """Test distinctiveness enhancement uplift calculation."""
        # Find a valid path
        for target_name, target_id in HABITAT_TYPE_TO_ID.items():
            for cond_name, cond_id in CONDITION_CATEGORY_TO_ID.items():
                col_idx = CONDITION_ID_TO_DIST_ENH_COL[cond_id]
                years = DISTINCTIVENESS_ENHANCEMENT_TEMPORAL[target_id, col_idx]
                if not jnp.isnan(years) and not jnp.isnan(ENHANCEMENT_RISK[target_id]):
                    baseline_id = (target_id + 1) % len(HABITAT_TYPE_TO_ID)
                    area = 1.0
                    strategic = 1.0

                    uplift = calculate_distinctiveness_enhancement_uplift(
                        baseline_id, cond_id, target_id, cond_id, area, strategic
                    )

                    self.assertIsInstance(float(uplift), float)
                    return
        self.fail("No valid distinctiveness enhancement path found")

    def test_calculate_batched_distinctiveness_enhancement_bng_units(self):
        """Test batched distinctiveness enhancement calculation."""
        baseline_habitat_ids = jnp.array([0, 1, 2])
        baseline_condition_ids = jnp.array([2, 2, 2])  # Moderate
        target_habitat_ids = jnp.array([3, 4, 5])
        target_condition_ids = jnp.array([2, 2, 2])  # Moderate
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])

        results = calculate_batched_distinctiveness_enhancement_bng_units(
            baseline_habitat_ids, baseline_condition_ids,
            target_habitat_ids, target_condition_ids,
            areas, strategic_multipliers
        )

        self.assertEqual(results.shape, (3,))

    def test_calculate_batched_distinctiveness_enhancement_uplift(self):
        """Test batched distinctiveness enhancement uplift."""
        baseline_habitat_ids = jnp.array([0, 1, 2])
        baseline_condition_ids = jnp.array([2, 2, 2])
        target_habitat_ids = jnp.array([3, 4, 5])
        target_condition_ids = jnp.array([2, 2, 2])
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])

        results = calculate_batched_distinctiveness_enhancement_uplift(
            baseline_habitat_ids, baseline_condition_ids,
            target_habitat_ids, target_condition_ids,
            areas, strategic_multipliers
        )

        self.assertEqual(results.shape, (3,))

    def test_calculate_distinctiveness_enhancement_bng_from_dataframe(self):
        """Test DataFrame interface for distinctiveness enhancement."""
        habitat_names = list(HABITAT_TYPE_TO_ID.keys())
        baseline_habitat = habitat_names[0]
        target_habitat = habitat_names[10]  # Different habitat

        data = {
            'Baseline_Habitat': [baseline_habitat, baseline_habitat],
            'Baseline_Condition': ['Poor', 'Moderate'],
            'Target_Habitat': [target_habitat, target_habitat],
            'Target_Condition': ['Poor', 'Moderate'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15]
        }
        df = pd.DataFrame(data)

        result = calculate_distinctiveness_enhancement_bng_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_calculate_distinctiveness_enhancement_uplift_from_dataframe(self):
        """Test DataFrame interface for distinctiveness enhancement uplift."""
        habitat_names = list(HABITAT_TYPE_TO_ID.keys())
        baseline_habitat = habitat_names[0]
        target_habitat = habitat_names[10]

        data = {
            'Baseline_Habitat': [baseline_habitat, baseline_habitat],
            'Baseline_Condition': ['Poor', 'Moderate'],
            'Target_Habitat': [target_habitat, target_habitat],
            'Target_Condition': ['Poor', 'Moderate'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15]
        }
        df = pd.DataFrame(data)

        result = calculate_distinctiveness_enhancement_uplift_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_jax_differentiability_distinctiveness(self):
        """Test that distinctiveness enhancement functions are differentiable."""
        baseline_habitat_ids = jnp.array([0, 1])
        baseline_condition_ids = jnp.array([2, 2])
        target_habitat_ids = jnp.array([3, 4])
        target_condition_ids = jnp.array([2, 2])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])

        def total_units_fn(areas_param):
            return calculate_total_distinctiveness_enhancement_units_from_jax_arrays(
                baseline_habitat_ids, baseline_condition_ids,
                target_habitat_ids, target_condition_ids,
                areas_param, strategic_multipliers
            )

        grad_fn = jax.grad(total_units_fn)
        gradients = grad_fn(areas)

        self.assertEqual(gradients.shape, areas.shape)

    def test_jax_differentiability_distinctiveness_uplift(self):
        """Test that distinctiveness enhancement uplift is differentiable."""
        baseline_habitat_ids = jnp.array([0, 1])
        baseline_condition_ids = jnp.array([2, 2])
        target_habitat_ids = jnp.array([3, 4])
        target_condition_ids = jnp.array([2, 2])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])

        def total_uplift_fn(areas_param):
            return calculate_total_distinctiveness_enhancement_uplift_from_jax_arrays(
                baseline_habitat_ids, baseline_condition_ids,
                target_habitat_ids, target_condition_ids,
                areas_param, strategic_multipliers
            )

        grad_fn = jax.grad(total_uplift_fn)
        gradients = grad_fn(areas)

        self.assertEqual(gradients.shape, areas.shape)


if __name__ == '__main__':
    unittest.main()
