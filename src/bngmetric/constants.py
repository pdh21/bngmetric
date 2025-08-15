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

#------Creation tables----
def _load_creation_table():
    """Loads the creation table from CSV and returns a DataFrame."""
    with pkg_resources.files(bngmetric_data).joinpath('creation_multiplier.csv').open('r') as f:
        df_creation=pd.read_csv(f)
    ordered_creation_values=[]
    # Get condition categories (column names excluding 'Habitat_Type')
    condition_categories = [col for col in df_creation.columns if col != 'Habitat Description']
    for habitat_name in HABITAT_TYPE_TO_ID.keys():
        if habitat_name in df_creation['Habitat Description'].values:
            # Get the row for this habitat from the loaded CSV, and extract the condition columns
            row_data = df_creation[df_creation['Habitat Description'] == habitat_name][condition_categories].iloc[0]
            # Convert any "30+" strings to numeric 32
            row_data = row_data.apply(lambda x: 32 if isinstance(x, str) and "30+" in x else x)
            ordered_creation_values.append(row_data.values)
        else:
            # If a habitat from master list is not in condition table, assume all 0s for conditions
            # (or raise an error if this should be impossible)
            ordered_creation_values.append(jnp.full(len(condition_categories),jnp.nan, dtype=jnp.float32))
    ordered_creation_values = jnp.array(ordered_creation_values, dtype=jnp.float32)

    return ordered_creation_values

# Load the creation data and get the JAX-compatible matrix
CREATION_MULTIPLIERS_MATRIX = _load_creation_table()

def _load_temporal_multiplier():
    """Loads the temporal multiplier from CSV and returns a JAX array."""
    with pkg_resources.files(bngmetric_data).joinpath('temporal_multiplier_LUT.csv').open('r') as f:
        df_temporal = pd.read_csv(f)
    # Convert the 'Years' column to a JAX array of multipliers
    return jnp.array(df_temporal.set_index('Year')['Time to target Multiplier'].values, dtype=jnp.float32)

TEMPORAL_MULTIPLIER_LOOKUP = _load_temporal_multiplier()


def _load_difficulty_multiplier():
    """Loads the difficulty multiplier from CSV and returns a dictionary for mapping."""
    with pkg_resources.files(bngmetric_data).joinpath('risk_multiplier.csv').open('r') as f:
        df_difficulty = pd.read_csv(f)
    ordered_creation_values=[]
    ordered_enhancement_values = []
    for habitat_name in HABITAT_TYPE_TO_ID.keys():
        if habitat_name in df_difficulty['Habitat Description'].values:
            # Get the row for this habitat from the loaded CSV, and extract the condition columns
            row_data = df_difficulty[df_difficulty['Habitat Description'] == habitat_name]['Multiplier'].iloc[0]
            
            ordered_creation_values.append(row_data)
            row_data = df_difficulty[df_difficulty['Habitat Description'] == habitat_name]['Multiplier.1'].iloc[0]
            ordered_enhancement_values.append(row_data)
        else:
            # If a habitat from master list is not in condition table, assume all 0s for conditions
            # (or raise an error if this should be impossible)
            ordered_creation_values.append(jnp.nan)
            ordered_enhancement_values.append(jnp.nan)
    ordered_creation_values = jnp.array(ordered_creation_values, dtype=jnp.float32)
    ordered_enhancement_values = jnp.array(ordered_enhancement_values, dtype=jnp.float32)
    return ordered_creation_values, ordered_enhancement_values

CREATION_RISK,ENHANCEMENT_RISK = _load_difficulty_multiplier()


#--load the temporal multiplier for enhancement
def _load_temporal_enhancement_multiplier():
    """Loads the temporal multiplier for enhancement from CSV and returns a JAX array."""
    with pkg_resources.files(bngmetric_data).joinpath('enhancement_multiplier.csv').open('r') as f:
        df_temporal_enhance = pd.read_csv(f)

    ## ----Same distinctiveness values----
    same_distinct=df_temporal_enhance[df_temporal_enhance.columns[[0,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,]]]
    same_distinct = same_distinct.drop(index=[1], inplace=False)

    # Create a dictionary mapping for column renaming
    column_mapping = {}
    for col in same_distinct.columns:
        # Handle columns with dots by extracting the base name
        if '.' in col:
            base_name = col.split('.')[0]
            column_mapping[col] = base_name
        else:
            column_mapping[col] = col

    # Apply the renaming using the dictionary
    renamed_distinct = same_distinct.rename(columns=column_mapping)

    # Create tuples for the MultiIndex
    index_tuples = list(zip(renamed_distinct.columns[1:], renamed_distinct.iloc[0, 1:].values))
    final_enhance=renamed_distinct.iloc[1:,:]
    final_enhance.rename(columns={'Start condition': 'Habitat'}, inplace=True)
    final_enhance.set_index('Habitat', inplace=True)
    final_enhance.columns=pd.MultiIndex.from_tuples(index_tuples, names=['Start Condition', 'Target Condition'])


    # ---  Get values and reshape ---
    # The .values will be a 1D array ordered correctly by the new index
    n_habitats = len(HABITAT_TYPE_TO_ID)
    conditions = list(CONDITION_CATEGORY_TO_ID.keys())
    n_conditions = len(conditions)
    s=np.full((n_habitats, n_conditions, n_conditions), jnp.nan, dtype=np.float32)
    # Reshape the values into a 3D array 
    for habitat in HABITAT_TYPE_TO_ID.keys():
        for start_condition in CONDITION_CATEGORY_TO_ID.keys():
            try:
                for tc in final_enhance.loc[habitat, (start_condition,)].index:
                    # Extract the habitat ID and condition IDs
                    habitat_id = HABITAT_TYPE_TO_ID[habitat]
                    current_id= CONDITION_CATEGORY_TO_ID[start_condition]
                    target_id = CONDITION_CATEGORY_TO_ID[tc]
                    s[habitat_id, current_id, target_id] = final_enhance.loc[habitat, (start_condition, tc)]
            except Exception as e:
                #print(f"Error processing habitat '{habitat}' with start condition '{start_condition}': {e}")
                continue


        #start_condition_id = CONDITION_CATEGORY_TO_ID[start_condition]
        #target_condition_id = CONDITION_CATEGORY_TO_ID[target_condition]
        #s[habitat_id, start_condition_id, target_condition_id] = s.values[i]

    # --- 2. Convert to JAX array ---
    lookup_jax_same = jnp.array(s, dtype=jnp.float32)

    ## ----Different distinctiveness values----
    diff_distinct=df_temporal_enhance[df_temporal_enhance.columns[0:8]]
    target=diff_distinct.iloc[0,1:]
    current=[i.split('Lower Distinctiveness Habitat - ')[1] for i in diff_distinct.iloc[1,1:]]
    diff_distinct_enhance=diff_distinct.iloc[2:,:]
    diff_distinct_enhance.rename(columns={'Start condition': 'Habitat'}, inplace=True)
    diff_distinct_enhance.set_index('Habitat', inplace=True)
    diff_distinct_enhance.columns=pd.MultiIndex.from_tuples(list(zip(current, target.values.tolist())), names=['Start Condition', 'Target Condition'])
    ordered_enhance_values = []
    for habitat_name in HABITAT_TYPE_TO_ID.keys():
        if habitat_name in diff_distinct_enhance.index.values:
            # Get the row for this habitat from the loaded CSV, and extract the condition columns
            row_data = diff_distinct_enhance.loc[habitat_name]
            # Convert any "30+" strings to numeric 32
            row_data = row_data.apply(lambda x: 32 if isinstance(x, str) and "30+" in x else x)
            ordered_enhance_values.append(row_data.values)
        else:
            # If a habitat from master list is not in condition table, assume all 0s for conditions
            # (or raise an error if this should be impossible)
            ordered_enhance_values.append(jnp.full(len(diff_distinct_enhance.columns),jnp.nan, dtype=jnp.float32))
    ordered_enhance_values = jnp.array(ordered_enhance_values, dtype=jnp.float32)
    # Convert the 'Years' column to a JAX array of multipliers
    return ordered_enhance_values,lookup_jax_same

DISTINCTIVENESS_ENHANCEMENT_TEMPORAL,CONDITION_ENHANCEMENT_TEMPORAL = _load_temporal_enhancement_multiplier()



