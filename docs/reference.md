# Lmo reference


## High-level API

### L-moments

::: lmo
    options:
      filters: 
      - "!l_weights"
      - "!^l_co"
      heading_level: 4

### Mixed L-(co)moments

::: lmo
    options:
      filters: ["^l_co"]
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
