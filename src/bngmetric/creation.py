import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit, vmap
from .constants import (
    HABITAT_TYPE_TO_ID,
    ID_TO_DISTINCTIVENESS_VALUE,
    CONDITION_CATEGORY_TO_ID,
    CONDITION_MULTIPLIERS_MATRIX,
    CREATION_MULTIPLIERS_MATRIX, 
    CREATION_RISK,
    TEMPORAL_MULTIPLIER_LOOKUP
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