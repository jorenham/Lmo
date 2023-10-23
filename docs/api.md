# Lmo reference


## High-level API

### Sample L-moments

::: lmo
    options:
      filters: 
      - "!l_weights"
      - "!^l_co"
      - "!^l_rv"
      heading_level: 4

### Sample L-comoments

::: lmo
    options:
      filters: ["^l_co"]
      heading_level: 4


### Distributions

::: lmo
    options:
      members:
      - l_rv_generic
      - l_rv_nonparametric
      heading_level: 4

### Statistical test and tools 

::: lmo.diagnostic
    options:
      heading_level: 4

## Low-level API

::: lmo
    options:
      members:
      - l_weights
      heading_level: 3

### `inference`

::: lmo.inference
    options:
      heading_level: 4

### `theoretical`

::: lmo.theoretical
    options:
      heading_level: 4

### `linalg`

::: lmo.linalg
    options:
      heading_level: 4
      show_root_full_path: true

### Order statistics

::: lmo.ostats
    options:
      heading_level: 4

### β PWMs

::: lmo.pwm_beta
    options:
      heading_level: 4
