Constants API
=============

The ``constants`` module provides lookup tables and mappings loaded from
the official BNG metric data files.

.. module:: bngmetric.constants


Habitat Mappings
----------------

.. py:data:: HABITAT_TYPE_TO_ID
   :type: dict[str, int]

   Maps habitat type names to unique integer IDs for JAX array indexing.

   .. code-block:: python

      from bngmetric.constants import HABITAT_TYPE_TO_ID

      habitat_id = HABITAT_TYPE_TO_ID['Grassland - Lowland meadows']

.. py:data:: HABITAT_DISTINCTIVENESS_MAP
   :type: dict[str, float]

   Maps habitat type names to their distinctiveness scores.

.. py:data:: ID_TO_DISTINCTIVENESS_VALUE
   :type: jax.Array

   JAX array of distinctiveness values indexed by habitat ID.


Condition Mappings
------------------

.. py:data:: CONDITION_CATEGORY_TO_ID
   :type: dict[str, int]

   Maps condition category names to integer IDs.

   .. code-block:: python

      from bngmetric.constants import CONDITION_CATEGORY_TO_ID

      condition_id = CONDITION_CATEGORY_TO_ID['Good']  # Returns 0

.. py:data:: CONDITION_MULTIPLIERS_MATRIX
   :type: jax.Array

   2D JAX array of condition multipliers.
   Shape: ``(n_habitats, n_conditions)``

   Usage:

   .. code-block:: python

      multiplier = CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]


Creation Multipliers
--------------------

.. py:data:: CREATION_MULTIPLIERS_MATRIX
   :type: jax.Array

   2D JAX array of time-to-target years for habitat creation.
   Shape: ``(n_habitats, n_conditions)``
   NaN values indicate disallowed combinations.

.. py:data:: CREATION_RISK
   :type: jax.Array

   1D JAX array of difficulty multipliers for habitat creation.
   Indexed by habitat ID.

.. py:data:: TEMPORAL_MULTIPLIER_LOOKUP
   :type: jax.Array

   1D JAX array converting years to temporal multipliers.
   Index by year to get the multiplier.


Enhancement Multipliers
-----------------------

.. py:data:: CONDITION_ENHANCEMENT_TEMPORAL
   :type: jax.Array

   3D JAX array of years for condition enhancement.
   Shape: ``(n_habitats, n_conditions, n_conditions)``
   Index: ``[habitat_id, start_condition_id, target_condition_id]``

.. py:data:: DISTINCTIVENESS_ENHANCEMENT_TEMPORAL
   :type: jax.Array

   2D JAX array of years for distinctiveness enhancement.
   Shape: ``(n_habitats, n_condition_transitions)``

.. py:data:: ENHANCEMENT_RISK
   :type: jax.Array

   1D JAX array of difficulty multipliers for enhancement.
   Indexed by habitat ID.


Spatial Risk Multipliers
------------------------

.. py:data:: SPATIAL_RISK_CATEGORY_TO_ID
   :type: dict[str, int]

   Maps spatial risk category names to integer IDs.

   .. code-block:: python

      SPATIAL_RISK_CATEGORY_TO_ID = {
          'Inside LPA/NCA': 0,
          'Neighbouring LPA/NCA': 1,
          'Beyond neighbouring LPA/NCA': 2,
      }

.. py:data:: SPATIAL_RISK_MULTIPLIERS
   :type: jax.Array

   JAX array of spatial risk multipliers: ``[1.0, 0.75, 0.5]``


Level 2 Habitat Labels
----------------------

.. py:data:: HABITAT_TO_LEVEL2
   :type: dict[str, str]

   Maps specific habitat types to their Level 2 broad habitat category.

.. py:data:: LEVEL2_TO_ID
   :type: dict[str, int]

   Maps Level 2 habitat labels to integer IDs.


Usage Example
-------------

.. code-block:: python

   from bngmetric.constants import (
       HABITAT_TYPE_TO_ID,
       CONDITION_CATEGORY_TO_ID,
       CONDITION_MULTIPLIERS_MATRIX,
       ID_TO_DISTINCTIVENESS_VALUE
   )

   # Get IDs
   habitat_id = HABITAT_TYPE_TO_ID['Grassland - Lowland meadows']
   condition_id = CONDITION_CATEGORY_TO_ID['Good']

   # Look up values
   distinctiveness = ID_TO_DISTINCTIVENESS_VALUE[habitat_id]
   condition_multiplier = CONDITION_MULTIPLIERS_MATRIX[habitat_id, condition_id]

   print(f"Distinctiveness: {distinctiveness}")
   print(f"Condition multiplier: {condition_multiplier}")
