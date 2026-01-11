import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit, vmap
from .constants import (
    HABITAT_TYPE_TO_ID,
    CONDITION_CATEGORY_TO_ID,
    CREATION_MULTIPLIERS_MATRIX,
    CREATION_RISK,
    TEMPORAL_MULTIPLIER_LOOKUP,
    SPATIAL_RISK_CATEGORY_TO_ID,
    SPATIAL_RISK_MULTIPLIERS
)
from .core_calculations import get_distinctiveness_value, get_condition_multiplier

@jit
def get_temporal_multiplier(habitat_id: int
                            ,condition_id: int) -> float:
    """Retrieves the temporal multiplier for a given number of years
    by looking up in the creation multipliers matrix."""
    time_to_target_years=CREATION_MULTIPLIERS_MATRIX[habitat_id,condition_id]
    is_disallowed = jnp.isnan(time_to_target_years)
    safe_years_float = jnp.nan_to_num(time_to_target_years, nan=0.0)
    safe_year_indices = safe_years_float.astype(jnp.int32)
    all_multipliers = TEMPORAL_MULTIPLIER_LOOKUP[safe_year_indices,]
    final_temporal_multipliers = jnp.where(
        is_disallowed,  # Condition: Was this slot originally nan?
        0.0,            # If True: Set multiplier to 0.0
        all_multipliers # If False: Use the looked-up multiplier
        )
    # Assumes TEMPORAL_MULTIPLIER_LOOKUP is a JAX array indexed by years
    return final_temporal_multipliers

@jit
def get_difficulty_multiplier(habitat_id: int) -> float:
    """Retrieves the difficulty multiplier for creating a given habitat ID."""
    # Assumes CREATION_RISK is a JAX array indexed by habitat_id
    return CREATION_RISK[habitat_id]


@jit
def calculate_creation_bng_unit(
    habitat_id: int,
    condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the biodiversity units for a single habitat creation parcel.
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    condition_m = get_condition_multiplier(habitat_id, condition_id)
    temporal_m = get_temporal_multiplier(habitat_id,condition_id)
    difficulty_m = get_difficulty_multiplier(habitat_id)

    units = area * distinctiveness * condition_m * strategic_multiplier * temporal_m * difficulty_m
    return units

@jit
@vmap
def calculate_batched_creation_bng_units(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array) -> jax.Array:
    """
    Calculates creation biodiversity units for multiple habitat parcels in a batch.
    Inputs should be JAX arrays of the same length.
    """
    return calculate_creation_bng_unit(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )

def calculate_creation_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total creation biodiversity units from a pandas DataFrame.
    For on-site creation (no spatial risk multiplier).
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_units = calculate_batched_creation_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )

    total_bng = jnp.sum(parcel_units)
    return float(total_bng)


# =============================================================================
# OFF-SITE CREATION (includes spatial risk multiplier)
# =============================================================================

@jit
def get_spatial_risk_multiplier(spatial_risk_category_id: int) -> float:
    """Retrieves the spatial risk multiplier for off-site compensation."""
    return SPATIAL_RISK_MULTIPLIERS[spatial_risk_category_id]


@jit
def calculate_offsite_creation_bng_unit(
    habitat_id: int,
    condition_id: int,
    area: float,
    strategic_multiplier: float,
    spatial_risk_category_id: int
) -> float:
    """
    Calculates the biodiversity units for a single off-site habitat creation parcel.
    Includes spatial risk multiplier based on distance from impact site.

    Args:
        habitat_id: ID of the habitat being created
        condition_id: Target condition ID
        area: Area in hectares
        strategic_multiplier: Strategic significance multiplier
        spatial_risk_category_id: Spatial risk category (0=Inside LPA, 1=Neighbouring, 2=Beyond)
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    condition_m = get_condition_multiplier(habitat_id, condition_id)
    temporal_m = get_temporal_multiplier(habitat_id, condition_id)
    difficulty_m = get_difficulty_multiplier(habitat_id)
    spatial_risk_m = get_spatial_risk_multiplier(spatial_risk_category_id)

    units = (area * distinctiveness * condition_m * strategic_multiplier *
             temporal_m * difficulty_m * spatial_risk_m)
    return units


@jit
@vmap
def calculate_batched_offsite_creation_bng_units(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> jax.Array:
    """
    Calculates off-site creation biodiversity units for multiple parcels in a batch.
    Inputs should be JAX arrays of the same length.
    """
    return calculate_offsite_creation_bng_unit(
        habitat_ids, condition_ids, areas, strategic_multipliers, spatial_risk_category_ids
    )


def calculate_offsite_creation_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total off-site creation biodiversity units from a pandas DataFrame.

    Expected columns:
        - Habitat: habitat type string
        - Condition: target condition string
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
        - Spatial_Risk: spatial risk category string ('Inside LPA/NCA', 'Neighbouring LPA/NCA', 'Beyond neighbouring LPA/NCA')
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)
    spatial_risk_ids = jnp.array([SPATIAL_RISK_CATEGORY_TO_ID[sr] for sr in df['Spatial_Risk'].values])

    parcel_units = calculate_batched_offsite_creation_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers, spatial_risk_ids
    )

    total_bng = jnp.sum(parcel_units)
    return float(total_bng)


# Utility functions for JAX-based optimization
@jit
def calculate_total_creation_units_from_jax_arrays(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of on-site creation biodiversity units from JAX arrays.
    This is the function to differentiate for sensitivity analysis.
    """
    parcel_units = calculate_batched_creation_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )
    return jnp.sum(parcel_units)


@jit
def calculate_total_offsite_creation_units_from_jax_arrays(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> float:
    """
    Calculates the sum of off-site creation biodiversity units from JAX arrays.
    This is the function to differentiate for sensitivity analysis.
    """
    parcel_units = calculate_batched_offsite_creation_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers, spatial_risk_category_ids
    )
    return jnp.sum(parcel_units)