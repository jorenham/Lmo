# Lmo reference


## High-level API

### Sample L-moments

::: lmo
    options:
      filters: 
      - "!l_weights"
      - "!^l_co"
      heading_level: 4

### Sample L-comoments

::: lmo
    options:
      filters: ["^l_co"]
      heading_level: 4

### Population L-moments

::: lmo.theoretical
    options:
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

### `linalg`

::: lmo.linalg
    options:
      heading_level: 4
      show_root_full_path: true
