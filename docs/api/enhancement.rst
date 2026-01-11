Enhancement API
===============

The ``enhancement`` module provides functions for calculating biodiversity
units from habitat enhancement, including condition enhancement, distinctiveness
enhancement, and off-site scenarios.

.. module:: bngmetric.enhancement


Condition Enhancement (On-site)
-------------------------------

For improving condition within the same habitat type.

.. autofunction:: get_enhancement_temporal_multiplier

.. autofunction:: get_enhancement_difficulty_multiplier

.. autofunction:: calculate_enhancement_bng_unit

.. autofunction:: calculate_baseline_unit_for_enhancement

.. autofunction:: calculate_enhancement_uplift

.. autofunction:: calculate_batched_enhancement_bng_units

.. autofunction:: calculate_batched_enhancement_uplift

.. autofunction:: calculate_enhancement_bng_from_dataframe

.. autofunction:: calculate_enhancement_uplift_from_dataframe

.. autofunction:: calculate_total_enhancement_units_from_jax_arrays

.. autofunction:: calculate_total_enhancement_uplift_from_jax_arrays


Distinctiveness Enhancement
---------------------------

For enhancing to a higher distinctiveness habitat within the same broad type.

.. autofunction:: get_distinctiveness_enhancement_temporal_multiplier

.. autofunction:: calculate_distinctiveness_enhancement_bng_unit

.. autofunction:: calculate_distinctiveness_enhancement_uplift

.. autofunction:: calculate_batched_distinctiveness_enhancement_bng_units

.. autofunction:: calculate_batched_distinctiveness_enhancement_uplift

.. autofunction:: calculate_distinctiveness_enhancement_bng_from_dataframe

.. autofunction:: calculate_distinctiveness_enhancement_uplift_from_dataframe

.. autofunction:: calculate_total_distinctiveness_enhancement_units_from_jax_arrays

.. autofunction:: calculate_total_distinctiveness_enhancement_uplift_from_jax_arrays


Off-site Enhancement
--------------------

For off-site enhancement with spatial risk multiplier.

.. autofunction:: get_spatial_risk_multiplier

.. autofunction:: calculate_offsite_enhancement_bng_unit

.. autofunction:: calculate_offsite_enhancement_uplift

.. autofunction:: calculate_batched_offsite_enhancement_bng_units

.. autofunction:: calculate_batched_offsite_enhancement_uplift

.. autofunction:: calculate_offsite_enhancement_bng_from_dataframe

.. autofunction:: calculate_offsite_enhancement_uplift_from_dataframe

.. autofunction:: calculate_total_offsite_enhancement_units_from_jax_arrays

.. autofunction:: calculate_total_offsite_enhancement_uplift_from_jax_arrays


DataFrame Column Requirements
-----------------------------

Condition Enhancement
^^^^^^^^^^^^^^^^^^^^^

- ``Habitat``: Habitat type string
- ``Start_Condition``: Condition before enhancement
- ``Target_Condition``: Condition after enhancement
- ``Area``: Area in hectares
- ``Strategic_Significance``: Strategic significance multiplier

Distinctiveness Enhancement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``Baseline_Habitat``: Original (lower distinctiveness) habitat type
- ``Baseline_Condition``: Condition of baseline habitat
- ``Target_Habitat``: Target (higher distinctiveness) habitat type
- ``Target_Condition``: Condition of target habitat
- ``Area``: Area in hectares
- ``Strategic_Significance``: Strategic significance multiplier

Off-site Enhancement
^^^^^^^^^^^^^^^^^^^^

All condition enhancement columns plus:

- ``Spatial_Risk``: Spatial risk category string
