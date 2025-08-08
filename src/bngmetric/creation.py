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
    CREATION_RISK  
)
from .core_calculations import get_distinctiveness_value, get_condition_multiplier

@jit
def get_temporal_multiplier(time_to_target_years: int) -> float:
    """Retrieves the temporal multiplier for a given number of years."""
    # Assumes TEMPORAL_MULTIPLIER_LOOKUP is a JAX array indexed by years
    return CREATION_MULTIPLIERS_MATRIX[time_to_target_years,]

@jit
def get_difficulty_multiplier(habitat_id: int) -> float:
    """Retrieves the difficulty multiplier for creating a given habitat ID."""
    # Assumes DIFFICULTY_MULTIPLIER_LOOKUP is a JAX array indexed by habitat_id
    return DIFFICULTY_MULTIPLIER_LOOKUP[habitat_id]

@jit
def calculate_creation_bng_unit(
    habitat_id: int,
    condition_id: int,
    area: float,
    strategic_multiplier: float,
    time_to_target_years: int,
) -> float:
    """
    Calculates the biodiversity units for a single habitat creation parcel.
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    condition_m = get_condition_multiplier(habitat_id, condition_id)
    temporal_m = get_temporal_multiplier(time_to_target_years)
    difficulty_m = get_difficulty_multiplier(habitat_id)

    units = area * distinctiveness * condition_m * strategic_multiplier * temporal_m * difficulty_m
    return units

@jit
@vmap
def calculate_batched_creation_bng_units(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    times_to_target_years: jax.Array,
) -> jax.Array:
    """
    Calculates creation biodiversity units for multiple habitat parcels in a batch.
    Inputs should be JAX arrays of the same length.
    """
    return calculate_creation_bng_unit(
        habitat_ids, condition_ids, areas, strategic_multipliers, times_to_target_years
    )

def calculate_creation_bng_from_dataframe(df: pd.DataFrame) -> float:
    """

    Calculates total creation biodiversity units from a pandas DataFrame.
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)
    times_to_target_years = jnp.array(df['TimeToTarget'].values, dtype=jnp.int32)

    parcel_units = calculate_batched_creation_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers, times_to_target_years
    )
    
    total_bng = jnp.sum(parcel_units)
    return float(total_bng)