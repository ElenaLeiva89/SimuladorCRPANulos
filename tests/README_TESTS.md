# Banco de pruebas del simulador CRPA

La suite esta adaptada a la version actual del proyecto:

- `simulation_config.doa_mode`: valor unico por ejecucion (`fixed` o `variable`).
- `beamforming_config.algorithm`: valor unico por ejecucion
  (`power_inversion`, `lcmv` o `lcmvq`).
- `array_config.steering_model`: `ideal` o `measured`.
- El modelo `measured` se prueba con MAT sinteticos de fase/amplitud, sin
  depender de ficheros locales externos.
- `jammer_config.num_jammers` y `jammer_config.jnr_dB`: valores comunes por
  corrida.
- Los jammers pueden definir `center_frequency_hz`; si no, se deriva desde la
  portadora GNSS y `normalized_frequency`.
- La ejecucion principal usa generadores aleatorios sin semilla de
  configuracion; la suite no exige reproducibilidad exacta entre corridas
  Monte Carlo.

## Instalacion

Desde la raiz del proyecto:

```bash
python -m pip install -r tests/requirements-dev.txt
python -m pytest tests
```

## Cobertura funcional

Los tests cubren:

- parseo, defaults y validaciones de configuracion;
- geometria `hexagonal_7` y steering vector ideal;
- carga, decodificacion e interpolacion de steering medido desde MAT;
- generacion de ruido, tonos, gaussianos complejos y chirps;
- frecuencia RF y longitud de onda por jammer;
- covarianza muestral, diagonal loading y pseudoinversa;
- calculo de pesos `power_inversion`, `lcmv` y `lcmvq`;
- sensibilidad de nulos ante error DoA, calibracion y acoplo simulado;
- cortes de patron, malla 2D y PSD temporal;
- metricas de profundidad y region 2D;
- escritura de CSV, NPZ, logs y plots con Matplotlib en backend `Agg`;
- integracion ligera de `run_project`;
- ejecucion end-to-end con `doa_mode = variable` para los tres algoritmos.

## Casos aun no modelados

- Interpolacion angular del steering medido entre muestras reales.
- Uso fisico de `bandwidth_hz` en senales de jammer.
- Geometrias distintas de `hexagonal_7`.
- Validacion visual de plots mas alla de que se generen ficheros.
