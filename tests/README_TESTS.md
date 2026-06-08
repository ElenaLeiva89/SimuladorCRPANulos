# Banco de pruebas del simulador CRPA

La suite esta adaptada a la version actual del proyecto:

- `simulation_config.doa_mode`: valor unico por ejecucion (`fixed` o `variable`).
- `beamforming_config.algorithm`: valor unico por ejecucion (`lcmv` o `power_inversion`).
- `jammer_config.num_jammers` y `jammer_config.jnr_dB`: valores comunes por corrida.
- `generate_received_snapshot_matrix(...)`: devuelve matriz total, tabla de jammers, ruido y contribucion jammer.
- `compute_weights(config, snapshot_matrix, element_positions_m, jammer_list)`.
- Metricas por jammer exportadas con contexto de Monte Carlo, algoritmo, modo DoA, tipo de jammer y region 2D de nulo.

## Instalacion

Desde la raiz del proyecto:

```bash
python -m pip install -r tests/requirements-dev.txt
python -m pytest tests
```

## Cobertura funcional

Los tests cubren:

- parseo, defaults y validaciones de `input_config.json`;
- geometria `hexagonal_7` y steering vectors ideal/medido;
- generacion de ruido, tonos, gaussianos complejos y chirps;
- covarianza muestral, diagonal loading y pseudoinversa;
- calculo de pesos `power_inversion` y `lcmv`;
- sensibilidad de nulos ante error DoA, calibracion y acoplo simulado;
- cortes de patron, malla 2D y PSD temporal;
- metricas de profundidad y region 2D;
- escritura de CSV, NPZ, logs y plots con Matplotlib en backend `Agg`;
- integracion ligera de `run_project`;
- ejecucion end-to-end con `doa_mode = variable` para `lcmv` y `power_inversion`.

## Casos aun no modelados

- Interpolacion del steering medido entre muestras reales.
- Uso fisico de `bandwidth_hz` en senales de jammer.
- Geometrias distintas de `hexagonal_7`.
- Validacion de calidad visual de plots mas alla de que se generen ficheros.
