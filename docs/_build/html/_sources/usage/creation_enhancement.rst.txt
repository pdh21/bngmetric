Habitat Creation & Enhancement
==============================

This guide covers calculating biodiversity units for habitat creation and
enhancement interventions.


Habitat Creation
----------------

Habitat creation involves establishing new habitats where none existed before
(or replacing a lower-value habitat). Creation calculations include additional
multipliers for:

- **Temporal risk**: Time required to reach target condition
- **Difficulty**: Technical difficulty of creating the habitat

Formula
^^^^^^^

.. math::

   \text{Creation Units} = \text{Area} \times \text{Distinctiveness} \times \text{Condition} \times \text{Strategic} \times \text{Temporal} \times \text{Difficulty}


Basic Creation Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from bngmetric.creation import calculate_creation_bng_from_dataframe

   creation = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows', 'Wetland - Reedbeds'],
       'Condition': ['Good', 'Moderate'],
       'Area': [3.0, 1.5],
       'Strategic_Significance': [1.15, 1.0]
   })

   units = calculate_creation_bng_from_dataframe(creation)
   print(f"Creation units: {units:.2f}")


Habitat Enhancement
-------------------

Enhancement improves existing habitat, either by:

1. **Condition enhancement**: Improving the condition of the same habitat type
   (e.g., Poor grassland to Good grassland)
2. **Distinctiveness enhancement**: Converting to a higher distinctiveness
   habitat within the same broad habitat type


Condition Enhancement
^^^^^^^^^^^^^^^^^^^^^

For improving condition within the same habitat:

.. code-block:: python

   import pandas as pd
   from bngmetric.enhancement import (
       calculate_enhancement_bng_from_dataframe,
       calculate_enhancement_uplift_from_dataframe
   )

   enhancement = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows'],
       'Start_Condition': ['Poor'],
       'Target_Condition': ['Good'],
       'Area': [2.0],
       'Strategic_Significance': [1.15]
   })

   # Post-enhancement units
   post_units = calculate_enhancement_bng_from_dataframe(enhancement)

   # Net uplift (post - baseline)
   uplift = calculate_enhancement_uplift_from_dataframe(enhancement)
   print(f"Post-enhancement units: {post_units:.2f}")
   print(f"Net uplift: {uplift:.2f}")


Distinctiveness Enhancement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For converting to a higher distinctiveness habitat (within the same broad type):

.. code-block:: python

   import pandas as pd
   from bngmetric.enhancement import (
       calculate_distinctiveness_enhancement_bng_from_dataframe,
       calculate_distinctiveness_enhancement_uplift_from_dataframe
   )

   enhancement = pd.DataFrame({
       'Baseline_Habitat': ['Grassland - Modified grassland'],
       'Baseline_Condition': ['Poor'],
       'Target_Habitat': ['Grassland - Lowland meadows'],
       'Target_Condition': ['Moderate'],
       'Area': [2.0],
       'Strategic_Significance': [1.15]
   })

   units = calculate_distinctiveness_enhancement_bng_from_dataframe(enhancement)
   uplift = calculate_distinctiveness_enhancement_uplift_from_dataframe(enhancement)


Understanding Uplift
--------------------

The **uplift** represents the net biodiversity gain from an intervention:

.. math::

   \text{Uplift} = \text{Post-intervention Units} - \text{Baseline Units}

This is the key metric for demonstrating biodiversity net gain.


Using JAX Arrays
----------------

For integration with JAX workflows or when processing large datasets:

.. code-block:: python

   import jax.numpy as jnp
   from bngmetric.creation import calculate_batched_creation_bng_units
   from bngmetric.constants import HABITAT_TYPE_TO_ID, CONDITION_CATEGORY_TO_ID

   habitat_ids = jnp.array([HABITAT_TYPE_TO_ID['Grassland - Lowland meadows']])
   condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID['Good']])
   areas = jnp.array([3.0])
   strategic = jnp.array([1.15])

   units = calculate_batched_creation_bng_units(
       habitat_ids, condition_ids, areas, strategic
   )


Disallowed Combinations
-----------------------

Not all habitat/condition combinations are valid for creation or enhancement.
Invalid combinations return 0 units (the temporal multiplier is set to 0).

.. code-block:: python

   from bngmetric.creation import get_temporal_multiplier

   # Check if a creation pathway is valid
   temporal_m = get_temporal_multiplier(habitat_id, condition_id)
   if temporal_m == 0:
       print("This habitat/condition combination is not valid for creation")
