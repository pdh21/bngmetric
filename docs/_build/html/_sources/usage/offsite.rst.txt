Off-site Compensation
=====================

When biodiversity gains cannot be achieved on-site, compensation can be
delivered at off-site locations. Off-site compensation includes an additional
**spatial risk multiplier** that accounts for the reduced ecological benefit
of delivering gains further from the impact site.


Spatial Risk Categories
-----------------------

The spatial risk multiplier depends on the distance between the impact site
and the compensation site:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Category
     - Multiplier
     - ID
   * - Inside LPA/NCA boundary
     - 1.00
     - 0
   * - Neighbouring LPA/NCA
     - 0.75
     - 1
   * - Beyond neighbouring LPA/NCA
     - 0.50
     - 2

Where:

- **LPA**: Local Planning Authority
- **NCA**: National Character Area


Off-site Creation
-----------------

.. code-block:: python

   import pandas as pd
   from bngmetric.creation import calculate_offsite_creation_bng_from_dataframe

   offsite_creation = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows', 'Wetland - Reedbeds'],
       'Condition': ['Good', 'Moderate'],
       'Area': [3.0, 2.0],
       'Strategic_Significance': [1.15, 1.0],
       'Spatial_Risk': ['Inside LPA/NCA', 'Neighbouring LPA/NCA']
   })

   units = calculate_offsite_creation_bng_from_dataframe(offsite_creation)
   print(f"Off-site creation units: {units:.2f}")


Comparing On-site vs Off-site
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from bngmetric.creation import (
       calculate_creation_bng_from_dataframe,
       calculate_offsite_creation_bng_from_dataframe
   )

   # Same habitat, same conditions
   habitat_data = {
       'Habitat': ['Grassland - Lowland meadows'],
       'Condition': ['Good'],
       'Area': [3.0],
       'Strategic_Significance': [1.15]
   }

   onsite = pd.DataFrame(habitat_data)
   onsite_units = calculate_creation_bng_from_dataframe(onsite)

   # Off-site in neighbouring LPA
   offsite = pd.DataFrame({
       **habitat_data,
       'Spatial_Risk': ['Neighbouring LPA/NCA']
   })
   offsite_units = calculate_offsite_creation_bng_from_dataframe(offsite)

   print(f"On-site units: {onsite_units:.2f}")
   print(f"Off-site units: {offsite_units:.2f}")
   print(f"Reduction: {(1 - offsite_units/onsite_units)*100:.0f}%")


Off-site Enhancement
--------------------

.. code-block:: python

   import pandas as pd
   from bngmetric.enhancement import (
       calculate_offsite_enhancement_bng_from_dataframe,
       calculate_offsite_enhancement_uplift_from_dataframe
   )

   offsite_enhancement = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows'],
       'Start_Condition': ['Poor'],
       'Target_Condition': ['Good'],
       'Area': [2.0],
       'Strategic_Significance': [1.15],
       'Spatial_Risk': ['Neighbouring LPA/NCA']
   })

   units = calculate_offsite_enhancement_bng_from_dataframe(offsite_enhancement)
   uplift = calculate_offsite_enhancement_uplift_from_dataframe(offsite_enhancement)

   print(f"Post-enhancement units: {units:.2f}")
   print(f"Net uplift: {uplift:.2f}")


Using JAX Arrays
----------------

For programmatic access with JAX:

.. code-block:: python

   import jax.numpy as jnp
   from bngmetric.creation import (
       calculate_batched_offsite_creation_bng_units,
       get_spatial_risk_multiplier
   )
   from bngmetric.constants import (
       HABITAT_TYPE_TO_ID,
       CONDITION_CATEGORY_TO_ID,
       SPATIAL_RISK_CATEGORY_TO_ID
   )

   habitat_ids = jnp.array([HABITAT_TYPE_TO_ID['Grassland - Lowland meadows']])
   condition_ids = jnp.array([CONDITION_CATEGORY_TO_ID['Good']])
   areas = jnp.array([3.0])
   strategic = jnp.array([1.15])
   spatial_risk = jnp.array([SPATIAL_RISK_CATEGORY_TO_ID['Neighbouring LPA/NCA']])

   units = calculate_batched_offsite_creation_bng_units(
       habitat_ids, condition_ids, areas, strategic, spatial_risk
   )


Intertidal Habitats
-------------------

For intertidal habitats, spatial risk is based on Marine Plan Areas rather
than LPA/NCA boundaries, but the multipliers are the same:

- Inside Marine Plan Area: 1.00
- Neighbouring Marine Plan Area: 0.75
- Beyond neighbouring Marine Plan Area: 0.50

Use the same ``Spatial_Risk`` column values; the multipliers are identical.
