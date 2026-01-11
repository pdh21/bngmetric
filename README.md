![BNG Metric](bngmetric.png)

# bngmetric

An open source Python implementation of the UK's Biodiversity Net Gain (BNG) Statutory Metric.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://github.com/google/jax)

## Overview

**bngmetric** provides two complementary capabilities:

### 1. Python Implementation of the BNG Metric

A straightforward, validated implementation of the UK's Statutory Biodiversity Metric. Use it as a drop-in replacement for spreadsheet-based calculations, with the benefits of programmatic access, automation, and integration with existing data pipelines.

```python
import pandas as pd
from bngmetric.core_calculations import calculate_bng_from_dataframe

baseline = pd.DataFrame({
    'Habitat': ['Grassland - Lowland meadows'],
    'Condition': ['Moderate'],
    'Area': [2.5],
    'Strategic_Significance': [1.15]
})

units = calculate_bng_from_dataframe(baseline)
print(f"Baseline units: {units:.2f}")
```

### 2. JAX-Powered Advanced Analysis

Built on JAX, enabling gradient-based methods for sophisticated analyses that aren't possible with traditional spreadsheet tools:

- **Sensitivity analysis**: Compute gradients to understand how changes in area, condition, or habitat type affect biodiversity units
- **Uncertainty propagation**: Propagate classification uncertainty (e.g., from remote sensing or ecologist assessments) through the metric using probabilistic methods
- **Optimisation**: Find optimal habitat creation/enhancement strategies subject to constraints

```python
import jax
from bngmetric.core_calculations import calculate_total_bng_from_jax_arrays

# Compute gradient of total units with respect to area
grad_fn = jax.grad(lambda areas: calculate_total_bng_from_jax_arrays(
    habitat_ids, condition_ids, areas, strategic
))
sensitivities = grad_fn(areas)
```

## Features

- **Baseline calculations**: Calculate biodiversity units for existing habitats
- **Habitat creation**: Includes temporal and difficulty multipliers
- **Habitat enhancement**: Both condition and distinctiveness pathways
- **Off-site compensation**: Spatial risk multipliers for off-site delivery
- **JAX integration**: JIT compilation, automatic differentiation, vectorisation

## Installation

```bash
pip install bngmetric
```

Or install from source:

```bash
git clone https://github.com/pdh21/bngmetric.git
cd bngmetric
pip install -e .
```

## Documentation

Full documentation is available at [pdh21.github.io/bngmetric](https://pdh21.github.io/bngmetric/).

## Quick Start

### Habitat Creation

```python
from bngmetric.creation import calculate_creation_bng_from_dataframe

creation = pd.DataFrame({
    'Habitat': ['Grassland - Lowland meadows'],
    'Condition': ['Good'],
    'Area': [3.0],
    'Strategic_Significance': [1.15]
})

units = calculate_creation_bng_from_dataframe(creation)
```

### Habitat Enhancement

```python
from bngmetric.enhancement import calculate_enhancement_uplift_from_dataframe

enhancement = pd.DataFrame({
    'Habitat': ['Grassland - Lowland meadows'],
    'Start_Condition': ['Poor'],
    'Target_Condition': ['Good'],
    'Area': [2.0],
    'Strategic_Significance': [1.15]
})

uplift = calculate_enhancement_uplift_from_dataframe(enhancement)
```

### Off-site with Spatial Risk

```python
from bngmetric.creation import calculate_offsite_creation_bng_from_dataframe

offsite = pd.DataFrame({
    'Habitat': ['Grassland - Lowland meadows'],
    'Condition': ['Good'],
    'Area': [3.0],
    'Strategic_Significance': [1.15],
    'Spatial_Risk': ['Neighbouring LPA/NCA']  # 0.75 multiplier
})

units = calculate_offsite_creation_bng_from_dataframe(offsite)
```

## License

This project is open source under the MIT License.

## Contributing

Contributions are welcome! Please see the documentation for development guidelines.
