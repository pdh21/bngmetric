import pandas as pd
import jax.numpy as jnp
import importlib.resources as pkg_resources
from . import data as bngmetric_data # Import the 'data' subpackage


# --- Load Distinctiveness and Create Master Habitat ID Mapping ---
def _load_distinctiveness_data():
    """Loads distinctiveness from CSV and returns a dictionary for mapping."""
    with pkg_resources.files(bngmetric_data).joinpath('habitat_LUT.csv').open('r') as f:
        df = pd.read_csv(f)
    return dict(zip(df['Labelling column'], df['Distinctiveness Score']))

HABITAT_DISTINCTIVENESS_MAP = _load_distinctiveness_data()

# Master mapping from habitat name to a unique integer ID
# This will be used consistently for both distinctiveness and condition matrix lookups.
# Ensures that the integer ID for 'Lowland calcareous grassland' is the same everywhere.
HABITAT_TYPE_TO_ID = {name: i for i, name in enumerate(HABITAT_DISTINCTIVENESS_MAP.keys())}

# JAX array for distinctiveness values, ordered by master HABITAT_TYPE_TO_ID
ID_TO_DISTINCTIVENESS_VALUE = jnp.array([
    HABITAT_DISTINCTIVENESS_MAP[name] for name in HABITAT_TYPE_TO_ID.keys()
], dtype=jnp.float32)


# --- Load and Process Habitat-Specific Condition Multiplier Matrix ---
def _load_condition_multipliers_matrix():
    """
    Loads condition multipliers from CSV as a matrix and reorders rows
    according to the master HABITAT_TYPE_TO_ID.
    """
    with pkg_resources.files(bngmetric_data).joinpath('condition_lookup.csv').open('r') as f:
        df_condition = pd.read_csv(f)

    # Get condition categories (column names excluding 'Habitat_Type')
    condition_categories = [col for col in df_condition.columns if col != 'Habitat Description']

    # Create a DataFrame that is reindexed/reordered by the master HABITAT_TYPE_TO_ID order.
    # This is crucial for direct indexing in JAX.
    # fillna(0) handles NaNs for non-applicable conditions.
    ordered_condition_values = []
    for habitat_name in HABITAT_TYPE_TO_ID.keys():
        if habitat_name in df_condition['Habitat Description'].values:
            # Get the row for this habitat from the loaded CSV, and extract the condition columns
            row_data = df_condition[df_condition['Habitat Description'] == habitat_name][condition_categories].iloc[0]
            ordered_condition_values.append(row_data.fillna(0).values)
        else:
            # If a habitat from master list is not in condition table, assume all 0s for conditions
            # (or raise an error if this should be impossible)
            ordered_condition_values.append(jnp.zeros(len(condition_categories), dtype=jnp.float32))

    CONDITION_MULTIPLIERS_MATRIX_JAX = jnp.array(ordered_condition_values, dtype=jnp.float32)
    
    return condition_categories, CONDITION_MULTIPLIERS_MATRIX_JAX

# Load the condition data and get the JAX-compatible matrixs
CONDITION_CATEGORIES_FOR_MATRIX, CONDITION_MULTIPLIERS_MATRIX = _load_condition_multipliers_matrix()

# Condition Category String to ID for Condition Matrix Columns
CONDITION_CATEGORY_TO_ID = {
    name: i for i, name in enumerate(CONDITION_CATEGORIES_FOR_MATRIX)
}

def _load_level2_mapping():
    """Loads Level 2 habitat labels from CSV and creates a mapping."""
    with pkg_resources.files(bngmetric_data).joinpath('habitat_LUT.csv').open('r') as f:
        df = pd.read_csv(f)
    return dict(zip(df['Labelling column'], df['Level 2 Label']))

HABITAT_TO_LEVEL2 = _load_level2_mapping()
# Level 2 Label to ID mapping for consistent testing
LEVEL2_TO_ID = {name: i for i, name in enumerate(pd.Series(HABITAT_TO_LEVEL2).unique())}
# JAX array for Level 2 IDs, ordered by master HABITAT_TYPE_TO_ID
LEVEL2_IDS_ARRAY = jnp.array([LEVEL2_TO_ID[HABITAT_TO_LEVEL2[i]] for i in HABITAT_TYPE_TO_ID.keys()], dtype=jnp.int32)