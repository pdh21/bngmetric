import unittest
import jax
import jax.numpy as jnp
import pandas as pd
from bngmetric.creation import (
    get_spatial_risk_multiplier,
    calculate_offsite_creation_bng_unit,
    calculate_batched_offsite_creation_bng_units,
    calculate_offsite_creation_bng_from_dataframe,
    calculate_creation_bng_unit,
    calculate_total_offsite_creation_units_from_jax_arrays
)
from bngmetric.enhancement import (
    calculate_offsite_enhancement_bng_unit,
    calculate_offsite_enhancement_uplift,
    calculate_batched_offsite_enhancement_bng_units,
    calculate_offsite_enhancement_bng_from_dataframe,
    calculate_offsite_enhancement_uplift_from_dataframe,
    calculate_total_offsite_enhancement_units_from_jax_arrays
)
from bngmetric.constants import (
    HABITAT_TYPE_TO_ID,
    CONDITION_CATEGORY_TO_ID,
    SPATIAL_RISK_CATEGORY_TO_ID,
    SPATIAL_RISK_MULTIPLIERS
)


class TestSpatialRiskMultipliers(unittest.TestCase):
    """Tests for spatial risk multiplier lookup."""

    def test_spatial_risk_categories_exist(self):
        """Test that spatial risk categories are defined."""
        self.assertEqual(len(SPATIAL_RISK_CATEGORY_TO_ID), 3)
        self.assertIn('Inside LPA/NCA', SPATIAL_RISK_CATEGORY_TO_ID)
        self.assertIn('Neighbouring LPA/NCA', SPATIAL_RISK_CATEGORY_TO_ID)
        self.assertIn('Beyond neighbouring LPA/NCA', SPATIAL_RISK_CATEGORY_TO_ID)

    def test_spatial_risk_multiplier_values(self):
        """Test that spatial risk multipliers have correct values."""
        self.assertEqual(float(SPATIAL_RISK_MULTIPLIERS[0]), 1.0)    # Inside LPA
        self.assertEqual(float(SPATIAL_RISK_MULTIPLIERS[1]), 0.75)   # Neighbouring
        self.assertEqual(float(SPATIAL_RISK_MULTIPLIERS[2]), 0.5)    # Beyond

    def test_get_spatial_risk_multiplier(self):
        """Test spatial risk multiplier retrieval."""
        self.assertEqual(float(get_spatial_risk_multiplier(0)), 1.0)
        self.assertEqual(float(get_spatial_risk_multiplier(1)), 0.75)
        self.assertEqual(float(get_spatial_risk_multiplier(2)), 0.5)


class TestOffsiteCreation(unittest.TestCase):
    """Tests for off-site habitat creation with spatial risk."""

    def test_offsite_creation_unit_inside_lpa(self):
        """Test off-site creation with Inside LPA spatial risk (multiplier=1.0)."""
        habitat_id = 0
        condition_id = 2  # Moderate
        area = 1.0
        strategic = 1.0
        spatial_risk = 0  # Inside LPA

        onsite_units = calculate_creation_bng_unit(
            habitat_id, condition_id, area, strategic
        )
        offsite_units = calculate_offsite_creation_bng_unit(
            habitat_id, condition_id, area, strategic, spatial_risk
        )

        # Should be equal when spatial risk = 1.0
        self.assertAlmostEqual(float(onsite_units), float(offsite_units), places=5)

    def test_offsite_creation_unit_neighbouring(self):
        """Test off-site creation with Neighbouring LPA spatial risk (multiplier=0.75)."""
        habitat_id = 0
        condition_id = 2
        area = 1.0
        strategic = 1.0
        spatial_risk = 1  # Neighbouring

        onsite_units = calculate_creation_bng_unit(
            habitat_id, condition_id, area, strategic
        )
        offsite_units = calculate_offsite_creation_bng_unit(
            habitat_id, condition_id, area, strategic, spatial_risk
        )

        # Offsite should be 75% of onsite
        self.assertAlmostEqual(float(offsite_units), float(onsite_units) * 0.75, places=5)

    def test_offsite_creation_unit_beyond(self):
        """Test off-site creation with Beyond LPA spatial risk (multiplier=0.5)."""
        habitat_id = 0
        condition_id = 2
        area = 1.0
        strategic = 1.0
        spatial_risk = 2  # Beyond

        onsite_units = calculate_creation_bng_unit(
            habitat_id, condition_id, area, strategic
        )
        offsite_units = calculate_offsite_creation_bng_unit(
            habitat_id, condition_id, area, strategic, spatial_risk
        )

        # Offsite should be 50% of onsite
        self.assertAlmostEqual(float(offsite_units), float(onsite_units) * 0.5, places=5)

    def test_batched_offsite_creation(self):
        """Test batched off-site creation calculation."""
        habitat_ids = jnp.array([0, 1, 2])
        condition_ids = jnp.array([2, 2, 2])
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])
        spatial_risk_ids = jnp.array([0, 1, 2])  # Different spatial risks

        results = calculate_batched_offsite_creation_bng_units(
            habitat_ids, condition_ids, areas, strategic_multipliers, spatial_risk_ids
        )

        self.assertEqual(results.shape, (3,))

    def test_offsite_creation_from_dataframe(self):
        """Test DataFrame interface for off-site creation."""
        habitat_name = list(HABITAT_TYPE_TO_ID.keys())[0]

        data = {
            'Habitat': [habitat_name, habitat_name],
            'Condition': ['Moderate', 'Good'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15],
            'Spatial_Risk': ['Inside LPA/NCA', 'Neighbouring LPA/NCA']
        }
        df = pd.DataFrame(data)

        result = calculate_offsite_creation_bng_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_offsite_creation_jax_differentiable(self):
        """Test that off-site creation is differentiable with JAX."""
        habitat_ids = jnp.array([0, 1])
        condition_ids = jnp.array([2, 2])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])
        spatial_risk_ids = jnp.array([0, 1])

        def total_fn(areas_param):
            return calculate_total_offsite_creation_units_from_jax_arrays(
                habitat_ids, condition_ids, areas_param,
                strategic_multipliers, spatial_risk_ids
            )

        grad_fn = jax.grad(total_fn)
        gradients = grad_fn(areas)

        self.assertEqual(gradients.shape, areas.shape)


class TestOffsiteEnhancement(unittest.TestCase):
    """Tests for off-site habitat enhancement with spatial risk."""

    def test_offsite_enhancement_unit(self):
        """Test off-site enhancement calculation."""
        habitat_id = 0
        start_condition_id = 4  # Poor
        target_condition_id = 2  # Moderate
        area = 1.0
        strategic = 1.0
        spatial_risk = 1  # Neighbouring

        units = calculate_offsite_enhancement_bng_unit(
            habitat_id, start_condition_id, target_condition_id,
            area, strategic, spatial_risk
        )

        self.assertIsInstance(float(units), float)

    def test_offsite_enhancement_uplift(self):
        """Test off-site enhancement uplift calculation."""
        habitat_id = 0
        start_condition_id = 4  # Poor
        target_condition_id = 2  # Moderate
        area = 1.0
        strategic = 1.0
        spatial_risk = 1  # Neighbouring

        uplift = calculate_offsite_enhancement_uplift(
            habitat_id, start_condition_id, target_condition_id,
            area, strategic, spatial_risk
        )

        self.assertIsInstance(float(uplift), float)

    def test_batched_offsite_enhancement(self):
        """Test batched off-site enhancement calculation."""
        habitat_ids = jnp.array([0, 1, 2])
        start_condition_ids = jnp.array([4, 4, 4])  # Poor
        target_condition_ids = jnp.array([2, 2, 2])  # Moderate
        areas = jnp.array([1.0, 2.0, 1.5])
        strategic_multipliers = jnp.array([1.0, 1.15, 1.0])
        spatial_risk_ids = jnp.array([0, 1, 2])

        results = calculate_batched_offsite_enhancement_bng_units(
            habitat_ids, start_condition_ids, target_condition_ids,
            areas, strategic_multipliers, spatial_risk_ids
        )

        self.assertEqual(results.shape, (3,))

    def test_offsite_enhancement_from_dataframe(self):
        """Test DataFrame interface for off-site enhancement."""
        habitat_name = list(HABITAT_TYPE_TO_ID.keys())[0]

        data = {
            'Habitat': [habitat_name, habitat_name],
            'Start_Condition': ['Poor', 'Moderate'],
            'Target_Condition': ['Moderate', 'Good'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15],
            'Spatial_Risk': ['Inside LPA/NCA', 'Neighbouring LPA/NCA']
        }
        df = pd.DataFrame(data)

        result = calculate_offsite_enhancement_bng_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_offsite_enhancement_uplift_from_dataframe(self):
        """Test DataFrame interface for off-site enhancement uplift."""
        habitat_name = list(HABITAT_TYPE_TO_ID.keys())[0]

        data = {
            'Habitat': [habitat_name, habitat_name],
            'Start_Condition': ['Poor', 'Moderate'],
            'Target_Condition': ['Moderate', 'Good'],
            'Area': [1.0, 2.0],
            'Strategic_Significance': [1.0, 1.15],
            'Spatial_Risk': ['Inside LPA/NCA', 'Beyond neighbouring LPA/NCA']
        }
        df = pd.DataFrame(data)

        result = calculate_offsite_enhancement_uplift_from_dataframe(df)
        self.assertIsInstance(result, float)

    def test_offsite_enhancement_jax_differentiable(self):
        """Test that off-site enhancement is differentiable with JAX."""
        habitat_ids = jnp.array([0, 1])
        start_condition_ids = jnp.array([4, 4])
        target_condition_ids = jnp.array([2, 2])
        areas = jnp.array([1.0, 2.0])
        strategic_multipliers = jnp.array([1.0, 1.15])
        spatial_risk_ids = jnp.array([0, 1])

        def total_fn(areas_param):
            return calculate_total_offsite_enhancement_units_from_jax_arrays(
                habitat_ids, start_condition_ids, target_condition_ids,
                areas_param, strategic_multipliers, spatial_risk_ids
            )

        grad_fn = jax.grad(total_fn)
        gradients = grad_fn(areas)

        self.assertEqual(gradients.shape, areas.shape)


if __name__ == '__main__':
    unittest.main()
