Basic Usage
===========

This guide covers the fundamental operations for calculating biodiversity units
using bngmetric.

Understanding the BNG Metric
----------------------------

The Biodiversity Net Gain metric calculates biodiversity value in standardised
"biodiversity units". The core formula is:

.. math::

   \text{Units} = \text{Area} \times \text{Distinctiveness} \times \text{Condition} \times \text{Strategic Significance}

Where:

- **Area**: Size in hectares
- **Distinctiveness**: Ecological value of the habitat type (V.Low to V.High)
- **Condition**: Current health/quality of the habitat (Poor to Good)
- **Strategic Significance**: Alignment with local biodiversity priorities


Calculating Baseline Units
--------------------------

The first step in any BNG assessment is calculating the baseline biodiversity
value of the site before development.

Using DataFrames
^^^^^^^^^^^^^^^^

The simplest approach is to use pandas DataFrames:

.. code-block:: python

   import pandas as pd
   from bngmetric.core_calculations import calculate_bng_from_dataframe

   # Define your baseline habitats
   baseline = pd.DataFrame({
       'Habitat': [
           'Grassland - Lowland meadows',
           'Woodland - Lowland mixed deciduous woodland',
           'Heathland and shrub - Lowland heathland'
       ],
       'Condition': ['Moderate', 'Good', 'Poor'],
       'Area': [2.5, 1.0, 0.5],
       'Strategic_Significance': [1.15, 1.0, 1.1]
   })

   total_units = calculate_bng_from_dataframe(baseline)
   print(f"Total baseline units: {total_units:.2f}")


Using JAX Arrays Directly
^^^^^^^^^^^^^^^^^^^^^^^^^

For performance or when integrating with JAX workflows:

.. code-block:: python

   import jax.numpy as jnp
   from bngmetric.core_calculations import calculate_batched_baseline_bng_units
   from bngmetric.constants import HABITAT_TYPE_TO_ID, CONDITION_CATEGORY_TO_ID

   # Convert to numerical IDs
   habitat_ids = jnp.array([
       HABITAT_TYPE_TO_ID['Grassland - Lowland meadows'],
       HABITAT_TYPE_TO_ID['Woodland - Lowland mixed deciduous woodland']
   ])
   condition_ids = jnp.array([
       CONDITION_CATEGORY_TO_ID['Moderate'],
       CONDITION_CATEGORY_TO_ID['Good']
   ])
   areas = jnp.array([2.5, 1.0])
   strategic = jnp.array([1.15, 1.0])

   # Calculate units for each parcel
   parcel_units = calculate_batched_baseline_bng_units(
       habitat_ids, condition_ids, areas, strategic
   )
   print(f"Per-parcel units: {parcel_units}")
   print(f"Total: {jnp.sum(parcel_units):.2f}")


Available Habitat Types
-----------------------

bngmetric includes all habitat types from the UK Habitat Classification:

.. code-block:: python

   from bngmetric.constants import HABITAT_TYPE_TO_ID

   # List all available habitats
   for habitat in sorted(HABITAT_TYPE_TO_ID.keys()):
       print(habitat)


Condition Categories
--------------------

The following condition categories are available:

- Good
- Fairly Good
- Moderate
- Fairly Poor
- Poor
- Condition Assessment N/A
- N/A - Other

.. code-block:: python

   from bngmetric.constants import CONDITION_CATEGORY_TO_ID

   print(CONDITION_CATEGORY_TO_ID)
   # {'Good': 0, 'Fairly Good': 1, 'Moderate': 2, ...}


Strategic Significance
----------------------

Strategic significance multipliers reflect alignment with local biodiversity
strategies:

- **1.15**: High strategic significance (formally identified in local strategy)
- **1.10**: Medium strategic significance (ecologically desirable location)
- **1.00**: Low strategic significance (not in local strategy)
