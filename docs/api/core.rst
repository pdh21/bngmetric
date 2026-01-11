Core Calculations API
=====================

The ``core_calculations`` module provides functions for calculating baseline
biodiversity units.

.. module:: bngmetric.core_calculations

Functions
---------

.. autofunction:: get_distinctiveness_value

.. autofunction:: get_condition_multiplier

.. autofunction:: calculate_baseline_bng_unit

.. autofunction:: calculate_batched_baseline_bng_units

.. autofunction:: calculate_bng_from_dataframe

.. autofunction:: calculate_total_bng_from_jax_arrays


Usage Examples
--------------

Single Parcel Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from bngmetric.core_calculations import calculate_baseline_bng_unit

   units = calculate_baseline_bng_unit(
       habitat_id=5,
       condition_id=2,
       area=1.5,
       strategic_multiplier=1.15
   )

Batched Calculation
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp
   from bngmetric.core_calculations import calculate_batched_baseline_bng_units

   units = calculate_batched_baseline_bng_units(
       habitat_ids=jnp.array([0, 1, 2]),
       condition_ids=jnp.array([2, 0, 4]),
       areas=jnp.array([1.0, 2.0, 0.5]),
       strategic_multipliers=jnp.array([1.0, 1.15, 1.1])
   )

DataFrame Interface
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from bngmetric.core_calculations import calculate_bng_from_dataframe

   df = pd.DataFrame({
       'Habitat': ['Grassland - Lowland meadows'],
       'Condition': ['Good'],
       'Area': [2.0],
       'Strategic_Significance': [1.15]
   })

   total = calculate_bng_from_dataframe(df)
