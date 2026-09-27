set positional-arguments := true
set shell := ["pwsh.exe", "-NoProfile", "-CommandWithArgs"]

[private]
default:
    @just --list

# Capture reconstruct merge and compare multiple placements
inspect *args:
    pixi run inspect @($args | Select-Object -Skip 1)

# Trace one frame from a selected placement
trace *args:
    pixi run trace @($args | Select-Object -Skip 1)

# Rebuild and compare an existing inspection run
merge *args:
    pixi run merge @($args | Select-Object -Skip 1)

# Show the latest or selected result
show *args:
    pixi run show @($args | Select-Object -Skip 1)

# Remove regenerable capture artifacts
compact *args:
    pixi run compact @($args | Select-Object -Skip 1)

# Check camera and turntable connectivity without changing device state
doctor:
    pixi run doctor
