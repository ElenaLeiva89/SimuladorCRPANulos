# Banco de pruebas del simulador CRPA

La suite está adaptada a la versión actual del proyecto:

- `simulation_config.doa_mode`: valor único por ejecución (`fixed` o `variable`).
- `beamforming_config.algorithm`: valor único por ejecución (`lcmv` o `power_inversion`).
- `jammer_config.num_jammers` y `jammer_config.jnr_dB`: valores únicos.
- `compute_weights(config, snapshot_matrix, element_positions_m, jammer_list)`.
- Métricas por jammer exportadas con contexto de Monte Carlo, algoritmo, modo DoA y tipo de jammer.

## Instalación

Desde la raíz del proyecto:

```bash
python -m pip install -r tests/requirements-dev.txt
python -m pytest tests
```

## Cobertura funcional

Los tests cubren:

- parseo, defaults y validaciones de `input_config.json`;
- geometría `hexagonal_7` y steering vectors;
- generación de ruido, tonos, gaussianos complejos y chirps;
- covarianza muestral, diagonal loading y pseudoinversa;
- cálculo de pesos `power_inversion` y `lcmv`;
- sensibilidad de nulos ante error DoA, calibración y acoplo simulado;
- cortes de patrón, malla 2D y FFT temporal;
- métricas de profundidad, anchura 1D y región 2D;
- escritura de CSV, NPZ, logs y plots con Matplotlib en backend `Agg`;
- integración ligera de `run_project`.
- ejecución end-to-end con `doa_mode = variable` para `lcmv` y `power_inversion`.

## Casos aún no modelados

- Steering medido real (`steering_model = measured`).
- Uso físico de `bandwidth_hz` en señales de jammer.
- Geometrías distintas de `hexagonal_7`.
- Validación de calidad visual de plots más allá de que se generen ficheros.
