# Power and Cooling

## Thermal Throttling
Servers throttle CPU frequency when inlet temperature exceeds 35°C. Sustained throttling shows up as unexplained latency spikes rather than errors. Check rack inlet sensors and confirm blanking panels are installed so hot exhaust air does not recirculate.

## PSU Redundancy Alerts
A `PSU redundancy lost` alert means one of the two power supplies has failed or lost input. The server keeps running on one PSU, but a second failure causes an immediate hard power-off. Replace the failed unit within the same maintenance window.
