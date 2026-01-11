import jax
import jax.numpy as jnp
import pandas as pd
from jax import jit, vmap
from .constants import (
    HABITAT_TYPE_TO_ID,
    CONDITION_CATEGORY_TO_ID,
    CONDITION_ENHANCEMENT_TEMPORAL,
    DISTINCTIVENESS_ENHANCEMENT_TEMPORAL,
    CONDITION_ID_TO_DIST_ENH_COL,
    ENHANCEMENT_RISK,
    TEMPORAL_MULTIPLIER_LOOKUP,
    SPATIAL_RISK_CATEGORY_TO_ID,
    SPATIAL_RISK_MULTIPLIERS
)
from .core_calculations import get_distinctiveness_value, get_condition_multiplier


@jit
def get_enhancement_temporal_multiplier(
    habitat_id: int,
    start_condition_id: int,
    target_condition_id: int
) -> float:
    """
    Retrieves the temporal multiplier for enhancement within the same distinctiveness band.
    Looks up time to target years from the 3D enhancement matrix, then converts to multiplier.
    """
    time_to_target_years = CONDITION_ENHANCEMENT_TEMPORAL[habitat_id, start_condition_id, target_condition_id]
    is_disallowed = jnp.isnan(time_to_target_years)
    safe_years_float = jnp.nan_to_num(time_to_target_years, nan=0.0)
    safe_year_indices = safe_years_float.astype(jnp.int32)
    all_multipliers = TEMPORAL_MULTIPLIER_LOOKUP[safe_year_indices]
    final_temporal_multiplier = jnp.where(
        is_disallowed,
        0.0,
        all_multipliers
    )
    return final_temporal_multiplier


@jit
def get_enhancement_difficulty_multiplier(habitat_id: int) -> float:
    """Retrieves the difficulty multiplier for enhancing a given habitat ID."""
    return ENHANCEMENT_RISK[habitat_id]


@jit
def calculate_enhancement_bng_unit(
    habitat_id: int,
    start_condition_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the post-enhancement biodiversity units for a single habitat parcel.
    This represents the units after enhancement, not the uplift.

    For same-distinctiveness enhancement (improving condition within same habitat type).
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    target_condition_m = get_condition_multiplier(habitat_id, target_condition_id)
    temporal_m = get_enhancement_temporal_multiplier(habitat_id, start_condition_id, target_condition_id)
    difficulty_m = get_enhancement_difficulty_multiplier(habitat_id)

    units = area * distinctiveness * target_condition_m * strategic_multiplier * temporal_m * difficulty_m
    return units


@jit
def calculate_baseline_unit_for_enhancement(
    habitat_id: int,
    start_condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the baseline biodiversity units for a parcel before enhancement.
    Used to compute the uplift from enhancement.
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    condition_m = get_condition_multiplier(habitat_id, start_condition_id)
    units = area * distinctiveness * condition_m * strategic_multiplier
    return units


@jit
def calculate_enhancement_uplift(
    habitat_id: int,
    start_condition_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the net biodiversity uplift from enhancement.
    Uplift = post-enhancement units - baseline units
    """
    post_units = calculate_enhancement_bng_unit(
        habitat_id, start_condition_id, target_condition_id, area, strategic_multiplier
    )
    baseline_units = calculate_baseline_unit_for_enhancement(
        habitat_id, start_condition_id, area, strategic_multiplier
    )
    return post_units - baseline_units


@jit
@vmap
def calculate_batched_enhancement_bng_units(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> jax.Array:
    """
    Calculates post-enhancement biodiversity units for multiple habitat parcels in a batch.
    Inputs should be JAX arrays of the same length.
    """
    return calculate_enhancement_bng_unit(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )


@jit
@vmap
def calculate_batched_enhancement_uplift(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> jax.Array:
    """
    Calculates the net biodiversity uplift from enhancement for multiple parcels.
    Inputs should be JAX arrays of the same length.
    """
    return calculate_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )


# =============================================================================
# DISTINCTIVENESS ENHANCEMENT (changing to higher distinctiveness habitat)
# =============================================================================

@jit
def get_distinctiveness_enhancement_temporal_multiplier(
    target_habitat_id: int,
    target_condition_id: int
) -> float:
    """
    Retrieves the temporal multiplier for distinctiveness enhancement.

    This is used when enhancing from a lower distinctiveness habitat to a
    higher distinctiveness habitat (within the same broad habitat type).

    The temporal lookup is based on the target habitat and target condition.
    The starting condition of the lower distinctiveness habitat is assumed
    to match the target condition.
    """
    # Convert condition_id to the column index used in DISTINCTIVENESS_ENHANCEMENT_TEMPORAL
    col_idx = CONDITION_ID_TO_DIST_ENH_COL[target_condition_id]
    time_to_target_years = DISTINCTIVENESS_ENHANCEMENT_TEMPORAL[target_habitat_id, col_idx]

    is_disallowed = jnp.isnan(time_to_target_years)
    safe_years_float = jnp.nan_to_num(time_to_target_years, nan=0.0)
    safe_year_indices = safe_years_float.astype(jnp.int32)
    all_multipliers = TEMPORAL_MULTIPLIER_LOOKUP[safe_year_indices]

    final_temporal_multiplier = jnp.where(
        is_disallowed,
        0.0,
        all_multipliers
    )
    return final_temporal_multiplier


@jit
def calculate_distinctiveness_enhancement_bng_unit(
    baseline_habitat_id: int,
    baseline_condition_id: int,
    target_habitat_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the post-enhancement biodiversity units for distinctiveness enhancement.

    This is for enhancing from a lower distinctiveness habitat to a higher
    distinctiveness habitat (within the same broad habitat type).

    Args:
        baseline_habitat_id: ID of the original (lower distinctiveness) habitat
        baseline_condition_id: Condition ID of the baseline habitat
        target_habitat_id: ID of the target (higher distinctiveness) habitat
        target_condition_id: Condition ID of the target habitat
        area: Area in hectares
        strategic_multiplier: Strategic significance multiplier
    """
    target_distinctiveness = get_distinctiveness_value(target_habitat_id)
    target_condition_m = get_condition_multiplier(target_habitat_id, target_condition_id)
    temporal_m = get_distinctiveness_enhancement_temporal_multiplier(target_habitat_id, target_condition_id)
    difficulty_m = get_enhancement_difficulty_multiplier(target_habitat_id)

    units = area * target_distinctiveness * target_condition_m * strategic_multiplier * temporal_m * difficulty_m
    return units


@jit
def calculate_distinctiveness_enhancement_uplift(
    baseline_habitat_id: int,
    baseline_condition_id: int,
    target_habitat_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float
) -> float:
    """
    Calculates the net biodiversity uplift from distinctiveness enhancement.

    Uplift = post-enhancement units - baseline units

    Args:
        baseline_habitat_id: ID of the original (lower distinctiveness) habitat
        baseline_condition_id: Condition ID of the baseline habitat
        target_habitat_id: ID of the target (higher distinctiveness) habitat
        target_condition_id: Condition ID of the target habitat
        area: Area in hectares
        strategic_multiplier: Strategic significance multiplier
    """
    post_units = calculate_distinctiveness_enhancement_bng_unit(
        baseline_habitat_id, baseline_condition_id,
        target_habitat_id, target_condition_id,
        area, strategic_multiplier
    )
    baseline_units = calculate_baseline_unit_for_enhancement(
        baseline_habitat_id, baseline_condition_id, area, strategic_multiplier
    )
    return post_units - baseline_units


@jit
@vmap
def calculate_batched_distinctiveness_enhancement_bng_units(
    baseline_habitat_ids: jax.Array,
    baseline_condition_ids: jax.Array,
    target_habitat_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> jax.Array:
    """
    Calculates post-enhancement biodiversity units for multiple distinctiveness
    enhancement parcels in a batch.
    """
    return calculate_distinctiveness_enhancement_bng_unit(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )


@jit
@vmap
def calculate_batched_distinctiveness_enhancement_uplift(
    baseline_habitat_ids: jax.Array,
    baseline_condition_ids: jax.Array,
    target_habitat_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> jax.Array:
    """
    Calculates the net biodiversity uplift from distinctiveness enhancement
    for multiple parcels.
    """
    return calculate_distinctiveness_enhancement_uplift(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )


def calculate_distinctiveness_enhancement_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total post-enhancement biodiversity units from distinctiveness
    enhancement using a pandas DataFrame.

    Expected columns:
        - Baseline_Habitat: baseline (lower distinctiveness) habitat type string
        - Baseline_Condition: condition of baseline habitat
        - Target_Habitat: target (higher distinctiveness) habitat type string
        - Target_Condition: condition of target habitat
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
    """
    baseline_habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Baseline_Habitat'].values])
    baseline_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Baseline_Condition'].values])
    target_habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Target_Habitat'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_units = calculate_batched_distinctiveness_enhancement_bng_units(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )

    total_bng = jnp.sum(parcel_units)
    return float(total_bng)


def calculate_distinctiveness_enhancement_uplift_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total net biodiversity uplift from distinctiveness enhancement
    using a pandas DataFrame.

    Expected columns:
        - Baseline_Habitat: baseline (lower distinctiveness) habitat type string
        - Baseline_Condition: condition of baseline habitat
        - Target_Habitat: target (higher distinctiveness) habitat type string
        - Target_Condition: condition of target habitat
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
    """
    baseline_habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Baseline_Habitat'].values])
    baseline_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Baseline_Condition'].values])
    target_habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Target_Habitat'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_uplifts = calculate_batched_distinctiveness_enhancement_uplift(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )

    total_uplift = jnp.sum(parcel_uplifts)
    return float(total_uplift)


# Utility functions for JAX-based optimization (distinctiveness enhancement)
@jit
def calculate_total_distinctiveness_enhancement_units_from_jax_arrays(
    baseline_habitat_ids: jax.Array,
    baseline_condition_ids: jax.Array,
    target_habitat_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of post-enhancement biodiversity units from JAX arrays
    for distinctiveness enhancement.
    """
    parcel_units = calculate_batched_distinctiveness_enhancement_bng_units(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )
    return jnp.sum(parcel_units)


@jit
def calculate_total_distinctiveness_enhancement_uplift_from_jax_arrays(
    baseline_habitat_ids: jax.Array,
    baseline_condition_ids: jax.Array,
    target_habitat_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of net biodiversity uplift from JAX arrays
    for distinctiveness enhancement.
    """
    parcel_uplifts = calculate_batched_distinctiveness_enhancement_uplift(
        baseline_habitat_ids, baseline_condition_ids,
        target_habitat_ids, target_condition_ids,
        areas, strategic_multipliers
    )
    return jnp.sum(parcel_uplifts)


# =============================================================================
# CONDITION ENHANCEMENT DataFrame and utility functions
# =============================================================================

def calculate_enhancement_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total post-enhancement biodiversity units from a pandas DataFrame.

    Expected columns:
        - Habitat: habitat type string
        - Start_Condition: condition before enhancement
        - Target_Condition: condition after enhancement
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    start_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Start_Condition'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_units = calculate_batched_enhancement_bng_units(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )

    total_bng = jnp.sum(parcel_units)
    return float(total_bng)


def calculate_enhancement_uplift_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total net biodiversity uplift from enhancement from a pandas DataFrame.

    Expected columns:
        - Habitat: habitat type string
        - Start_Condition: condition before enhancement
        - Target_Condition: condition after enhancement
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    start_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Start_Condition'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)

    parcel_uplifts = calculate_batched_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )

    total_uplift = jnp.sum(parcel_uplifts)
    return float(total_uplift)


# Utility functions for JAX-based optimization and sensitivity analysis
@jit
def calculate_total_enhancement_units_from_jax_arrays(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of post-enhancement biodiversity units from JAX arrays.
    This is the function to differentiate for sensitivity analysis.
    """
    parcel_units = calculate_batched_enhancement_bng_units(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )
    return jnp.sum(parcel_units)


@jit
def calculate_total_enhancement_uplift_from_jax_arrays(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array
) -> float:
    """
    Calculates the sum of net biodiversity uplift from JAX arrays.
    This is the function to differentiate for sensitivity analysis on uplift.
    """
    parcel_uplifts = calculate_batched_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids, areas, strategic_multipliers
    )
    return jnp.sum(parcel_uplifts)


# =============================================================================
# OFF-SITE ENHANCEMENT (includes spatial risk multiplier)
# =============================================================================

@jit
def get_spatial_risk_multiplier(spatial_risk_category_id: int) -> float:
    """Retrieves the spatial risk multiplier for off-site compensation."""
    return SPATIAL_RISK_MULTIPLIERS[spatial_risk_category_id]


@jit
def calculate_offsite_enhancement_bng_unit(
    habitat_id: int,
    start_condition_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float,
    spatial_risk_category_id: int
) -> float:
    """
    Calculates the post-enhancement biodiversity units for a single off-site
    condition enhancement parcel. Includes spatial risk multiplier.
    """
    distinctiveness = get_distinctiveness_value(habitat_id)
    target_condition_m = get_condition_multiplier(habitat_id, target_condition_id)
    temporal_m = get_enhancement_temporal_multiplier(habitat_id, start_condition_id, target_condition_id)
    difficulty_m = get_enhancement_difficulty_multiplier(habitat_id)
    spatial_risk_m = get_spatial_risk_multiplier(spatial_risk_category_id)

    units = (area * distinctiveness * target_condition_m * strategic_multiplier *
             temporal_m * difficulty_m * spatial_risk_m)
    return units


@jit
def calculate_offsite_enhancement_uplift(
    habitat_id: int,
    start_condition_id: int,
    target_condition_id: int,
    area: float,
    strategic_multiplier: float,
    spatial_risk_category_id: int
) -> float:
    """
    Calculates the net biodiversity uplift from off-site enhancement.
    Uplift = post-enhancement units - baseline units
    Note: Baseline units do NOT include spatial risk (they represent existing value).
    """
    post_units = calculate_offsite_enhancement_bng_unit(
        habitat_id, start_condition_id, target_condition_id,
        area, strategic_multiplier, spatial_risk_category_id
    )
    baseline_units = calculate_baseline_unit_for_enhancement(
        habitat_id, start_condition_id, area, strategic_multiplier
    )
    return post_units - baseline_units


@jit
@vmap
def calculate_batched_offsite_enhancement_bng_units(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> jax.Array:
    """
    Calculates off-site enhancement biodiversity units for multiple parcels.
    """
    return calculate_offsite_enhancement_bng_unit(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_category_ids
    )


@jit
@vmap
def calculate_batched_offsite_enhancement_uplift(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> jax.Array:
    """
    Calculates off-site enhancement uplift for multiple parcels.
    """
    return calculate_offsite_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_category_ids
    )


def calculate_offsite_enhancement_bng_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total off-site enhancement biodiversity units from a DataFrame.

    Expected columns:
        - Habitat: habitat type string
        - Start_Condition: condition before enhancement
        - Target_Condition: condition after enhancement
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
        - Spatial_Risk: spatial risk category string
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    start_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Start_Condition'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)
    spatial_risk_ids = jnp.array([SPATIAL_RISK_CATEGORY_TO_ID[sr] for sr in df['Spatial_Risk'].values])

    parcel_units = calculate_batched_offsite_enhancement_bng_units(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_ids
    )

    total_bng = jnp.sum(parcel_units)
    return float(total_bng)


def calculate_offsite_enhancement_uplift_from_dataframe(df: pd.DataFrame) -> float:
    """
    Calculates total off-site enhancement uplift from a DataFrame.

    Expected columns:
        - Habitat: habitat type string
        - Start_Condition: condition before enhancement
        - Target_Condition: condition after enhancement
        - Area: area in hectares
        - Strategic_Significance: strategic significance multiplier
        - Spatial_Risk: spatial risk category string
    """
    habitat_ids = jnp.array([HABITAT_TYPE_TO_ID[ht] for ht in df['Habitat'].values])
    start_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Start_Condition'].values])
    target_condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID[c] for c in df['Target_Condition'].values])
    areas = jnp.array(df['Area'].values)
    strategic_multipliers = jnp.array(df['Strategic_Significance'].values)
    spatial_risk_ids = jnp.array([SPATIAL_RISK_CATEGORY_TO_ID[sr] for sr in df['Spatial_Risk'].values])

    parcel_uplifts = calculate_batched_offsite_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_ids
    )

    total_uplift = jnp.sum(parcel_uplifts)
    return float(total_uplift)


# Off-site JAX utility functions
@jit
def calculate_total_offsite_enhancement_units_from_jax_arrays(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> float:
    """
    Calculates the sum of off-site enhancement biodiversity units from JAX arrays.
    """
    parcel_units = calculate_batched_offsite_enhancement_bng_units(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_category_ids
    )
    return jnp.sum(parcel_units)


@jit
def calculate_total_offsite_enhancement_uplift_from_jax_arrays(
    habitat_ids: jax.Array,
    start_condition_ids: jax.Array,
    target_condition_ids: jax.Array,
    areas: jax.Array,
    strategic_multipliers: jax.Array,
    spatial_risk_category_ids: jax.Array
) -> float:
    """
    Calculates the sum of off-site enhancement uplift from JAX arrays.
    """
    parcel_uplifts = calculate_batched_offsite_enhancement_uplift(
        habitat_ids, start_condition_ids, target_condition_ids,
        areas, strategic_multipliers, spatial_risk_category_ids
    )
    return jnp.sum(parcel_uplifts)
