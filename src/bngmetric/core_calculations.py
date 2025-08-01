import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit, vmap
from .constants import (
    HABITAT_TYPE_TO_ID, # Master mapping for distinctiveness and condition matrix rows
    ID_TO_DISTINCTIVENESS_VALUE,
    CONDITION_CATEGORY_TO_ID,
    CONDITION_MULTIPLIERS_MATRIX # The 2D JAX array, already ordered correctly
)

@jit
def get_distinctiveness_value(habitat_id: int) -> float:
    """Retrieves distinctiveness value for a given habitat ID."""
    return ID_TO_DISTINCTIVENESS_VALUE[habitat_id]

@jit
def get_condition_multiplier(habitat_id: int, condition_id: int) -> float:
    """
    Retrieves condition multiplier for a given habitat ID (row) and condition ID (column)
    from the pre-loaded matrix.
    """
    # Direct indexing into the 2D matrix, which is now guaranteed to be ordered by habitat_id
    return CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]

@jit
def calculate_baseline_bng_unit(
    habitat_id: int,       # Master numerical ID for habitat type
    condition_id: int,     # Numerical ID for condition category
    area: float,           # Area in hectares
    strategic_multiplier: float
) -> float:
    """
    Calculates the baseline biodiversity units for a single habitat parcel.
    Uses habitat-specific condition multipliers.
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    condition_m = get_condition_multiplier(habitat_id, condition_id)

    units = area * distinctiveness * condition_m * strategic_multiplier
    return units

@jit
@vmap
def calculate_batched_baseline_bng_units(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> jax.Array:
    """
    Calculates baseline biodiversity units for multiple habitat parcels in a batch.
    Inputs should be JAX arrays of the same length.
    """
    # vmap correctly applies habitat_ids to both get_distinctiveness_value and get_condition_multiplier
    return calculate_baseline_bng_unit(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )

def calculate_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total baseline biodiversity units from a pandas DataFrame.
    Automatically maps string inputs to JAX-compatible numerical IDs.
    """
    # Convert string columns to numerical IDs using the master mappings
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Condition'].values])

    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_units = calculate_batched_baseline_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )
    
    total_bng = jnp.sum(parcel_units)
    return float(total_bng)

# New utility function in core_calculations.py for overall BNG sum from JAX arrays
@jit
def calculate_total_bng_from_jax_arrays(
    habitat_ids: jax.Array,
    condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of baseline biodiversity units from JAX arrays.
    This is the function we'll differentiate.
    """
    parcel_units = calculate_batched_baseline_bng_units(
        habitat_ids, condition_ids, areas, strategic_multipliers
    )
    return jnp.sum(parcel_units)