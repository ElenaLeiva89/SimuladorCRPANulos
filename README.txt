# Simulador CRPA ideal de 7 elementos para Nullforming/Beamforming

Este proyecto simula una antena CRPA ideal de 7 elementos en geometría
hexagonal, orientada al cenit, para estudiar formación de haces y generación
de nulos frente a interferencias GNSS. El simulador genera ruido, jammers,
snapshots complejos, pesos adaptativos y métricas de profundidad, anchura y
área de nulo.

La versión actual ejecuta una combinación concreta por corrida:

- un único modo DoA: `fixed` o `variable`;
- un único algoritmo de beamforming: `power_inversion` o `lcmv`;
- un valor común de JNR para todos los jammers activos;
- una CRPA ideal hexagonal de 7 elementos.

## Flujo de ejecución

`main.py` ejecuta la simulación completa:

1. Lee `input_config.json`.
2. Valida rangos físicos y opciones soportadas.
3. Construye la geometría CRPA con `crpa_sim/array_model.py`.
4. Genera jammers y ruido con `crpa_sim/jammers.py`.
5. Calcula la matriz de snapshots recibidos `X`.
6. Estima la covarianza espacial `R = X X^H / L`.
7. Calcula pesos con `power_inversion` o `lcmv`.
8. Evalúa cortes azimut/elevación y malla 2D del patrón.
9. Calcula métricas de nulo por jammer, umbral y Monte Carlo.
10. Guarda logs, CSV, NPZ y figuras según `output_config`.

## Instalación y ejecución

Dependencias mínimas:

```bash
pip install numpy pandas matplotlib
```

Ejecución normal desde la raíz del proyecto:

```bash
python main.py
```

La configuración usada por defecto es:

```text
input_config.json
```

## Estructura del proyecto

```text
main.py                         Orquestación de la simulación.
input_config.json               Configuración principal de entrada.
README.txt                      Documentación del proyecto.

crpa_sim/
  array_model.py                Geometría CRPA y steering vectors.
  beamformers.py                Pesos Power Inversion y LCMV.
  config.py                     Dataclasses y normalización del JSON.
  covariance.py                 Covarianza, diagonal loading e inversión.
  fft_tools.py                  FFT temporal media de snapshots.
  io_utils.py                   Carga, validación y guardado de artefactos.
  jammers.py                    Ruido, señales jammer y matriz recibida.
  null_metrics.py               Profundidad, anchura y área de nulos.
  patterns.py                   Cortes y mallas de patrón espacial.
  plots.py                      Figuras de geometría, patrones y espectro.

tests/
  pytest.ini                    Configuración de pytest.
  requirements-dev.txt          Dependencias de desarrollo.
  README_TESTS.md               Resumen del banco de pruebas.
  tests/                        Tests unitarios e integración ligera.
```

## Especificación del modelo

### Array CRPA

- Geometría soportada: `hexagonal_7`.
- Número de elementos soportado: `7`.
- Elemento 1: centro del array.
- Elementos 2 a 7: anillo hexagonal exterior.
- Plano del array: `XY`.
- Boresight esperado: elevación `90.0` grados.
- Steering soportado: `ideal` isotropico y `measured` desde CSV real.

La separación física se calcula como:

```text
element_spacing_m = element_spacing_over_lambda * wavelength_m
```

### Convención angular

- `azimuth_deg = 0`: dirección `+X`.
- El azimut crece hacia `+Y`.
- `elevation_deg = 0`: horizonte.
- `elevation_deg = 90`: cenit.
- Los barridos configurados usan azimut `[0, 360]` y elevacion `[0, 90]`.

### Señal GNSS

Bandas soportadas:

```text
E5: 1.19179e9 Hz
E6: 1.27875e9 Hz
E1: 1.57542e9 Hz
```

También se admiten códigos históricos:

```text
1 -> E5
2 -> E6
3 -> E1
```

La longitud de onda se calcula como:

```text
wavelength_m = speed_of_light_m_s / carrier_frequency_hz
```

### Jammers

Tipos de señal soportados en `base_jammers[].signal_type`:

- `tone`: tono complejo con frecuencia normalizada.
- `complex_gaussian`: interferencia compleja gaussiana circular.
- `chirp`: chirp lineal complejo con frecuencia central normalizada.

La potencia lineal de cada jammer se calcula a partir del JNR:

```text
jammer_power_linear = noise_power_linear * 10^(jnr_dB / 10)
```

Para `chirp`, `chirp_frequency` es obligatorio y debe estar expresado como
frecuencia normalizada en ciclos por muestra.

### Algoritmos de beamforming

`power_inversion`:

- Estima la covarianza espacial a partir de `snapshot_matrix`.
- Aplica diagonal loading.
- Invierte con pseudoinversa.
- Fuerza una restricción sobre `power_inversion_reference_element`.

`lcmv`:

- Fuerza ganancia unitaria en la dirección deseada.
- Fuerza nulos en las direcciones de los jammers.
- Requiere como máximo `num_elements - 1` jammers.

Los pesos convencionales `delay-and-sum` se calculan en `patterns.py` para
comparativas internas, pero no son un algoritmo seleccionable en el JSON.

## Configuración completa

El fichero `input_config.json` contiene los siguientes bloques.

### `array_config`

```json
{
  "num_elements": 7,
  "geometry": "hexagonal_7",
  "element_type": "isotropic",
  "element_spacing_over_lambda": 0.5,
  "array_boresight_elevation_deg": 90.0,
  "steering_model": "ideal",
  "measured_steering_file": "data/crpa_measured_steering.csv"
}
```

Variables:

- `num_elements`: debe ser `7`.
- `geometry`: debe ser `hexagonal_7`.
- `element_type`: actualmente informativo, se usa `isotropic`.
- `element_spacing_over_lambda`: separación radial en longitudes de onda.
- `array_boresight_elevation_deg`: debe ser `90.0`.
- `steering_model`: `ideal` o `measured`.
- `measured_steering_file`: ruta CSV obligatoria cuando `steering_model = measured`.

### `signal_config`

```json
{
  "gnss_band": "E1",
  "speed_of_light_m_s": 299792458.0,
  "sample_rate_hz": 64000000.0,
  "num_snapshots": 4096,
  "fft_size": 8192
}
```

Variables:

- `gnss_band`: `E1`, `E5`, `E6`, o códigos `1`, `2`, `3`.
- `speed_of_light_m_s`: velocidad de la luz, debe ser mayor que cero.
- `sample_rate_hz`: frecuencia de muestreo, debe ser mayor que cero.
- `num_snapshots`: número de muestras temporales por canal.
- `fft_size`: tamaño de FFT; puede ser `null` para usar `num_snapshots`.

### `simulation_config`

```json
{
  "num_montecarlo": 10,
  "random_seed": 12345,
  "doa_mode": "fixed"
}
```

Variables:

- `num_montecarlo`: número de iteraciones, debe ser al menos `1`.
- `random_seed`: semilla base reproducible.
- `doa_mode`: `fixed` o `variable`.

En modo `fixed`, cada jammer usa el azimut/elevación de su plantilla. En modo
`variable`, cada iteración sortea las direcciones dentro de los rangos
configurados.

### `beamforming_config`

```json
{
  "algorithm": "power_inversion",
  "desired_azimuth_deg": 0.0,
  "desired_elevation_deg": 90.0,
  "diagonal_loading_factor": 0.001,
  "power_inversion_reference_element": 1
}
```

Variables:

- `algorithm`: `power_inversion` o `lcmv`.
- `desired_azimuth_deg`: azimut de la dirección deseada.
- `desired_elevation_deg`: elevación de la dirección deseada.
- `diagonal_loading_factor`: factor de regularización, debe ser `>= 0`.
- `power_inversion_reference_element`: índice del elemento de referencia.

Nota: el código usa índices Python de base cero. Por tanto, el elemento central
normalmente es `0`. Si se configura `1`, se usa el segundo elemento del vector
de posiciones.

### `scan_config`

```json
{
  "azimuth_scan_min_deg": 0.0,
  "azimuth_scan_max_deg": 360.0,
  "azimuth_scan_step_deg": 0.5,
  "elevation_scan_min_deg": 0.0,
  "elevation_scan_max_deg": 90.0,
  "elevation_scan_step_deg": 0.5,
  "null_thresholds_dB": [-10, -20, -30, -40, -50]
}
```
Los parametros de "azimuth_scan_step_deg": 0.5 y "elevation_scan_step_deg": 0.5 son valores 
muy pequeños que permiten el calculo para la CRPA ideal pero que hacen muy complejo el calculo
del steering vector de la CRPA real.
Se recomienda asignar valores mas altos para la CRPA real y no realentizar el calculo.

Variables:

- `azimuth_scan_min_deg`: mínimo de azimut.
- `azimuth_scan_max_deg`: máximo de azimut.
- `azimuth_scan_step_deg`: paso de azimut, debe ser mayor que cero.
- `elevation_scan_min_deg`: mínimo de elevación.
- `elevation_scan_max_deg`: máximo de elevación.
- `elevation_scan_step_deg`: paso de elevación, debe ser mayor que cero.
- `null_thresholds_dB`: umbrales usados para medir anchura/área de nulo.

### `noise_config`

```json
{
  "noise_power_linear": 1.0
}
```

Variables:

- `noise_power_linear`: potencia de ruido por canal en escala lineal.

### `jammer_config`

```json
{
  "num_jammers": 1,
  "jnr_dB": 20.0,
  "variable_doa_azimuth_range_deg": [0.0, 360.0],
  "variable_doa_elevation_range_deg": [0.0, 90.0],
  "base_jammers": [
    {
      "name": "Jammer_1",
      "azimuth_deg": 40.0,
      "elevation_deg": 20.0,
      "signal_type": "tone",
      "normalized_frequency": 0.05,
      "chirp_frequency": 0.05
    }
  ]
}
```

Variables:

- `num_jammers`: número de jammers activos. Debe cumplir
  `1 <= num_jammers <= num_elements - 1`.
- `jnr_dB`: relación jammer-ruido común para los jammers activos.
- `variable_doa_azimuth_range_deg`: rango usado si `doa_mode = variable`.
- `variable_doa_elevation_range_deg`: rango usado si `doa_mode = variable`.
- `base_jammers`: lista de plantillas. Se usan las primeras `num_jammers`.

Campos de cada jammer:

- `name`: nombre usado en tablas y figuras.
- `azimuth_deg`: azimut fijo si `doa_mode = fixed`. Rango de 0º a 360º
- `elevation_deg`: elevación fija si `doa_mode = fixed`. Rango de 0º a 90º
- `signal_type`: `tone`, `complex_gaussian` o `chirp`.
- `normalized_frequency`: frecuencia del tono en ciclos por muestra.
- `bandwidth_hz`: campo reservado para modelos futuros.
- `chirp_frequency`: frecuencia central normalizada para `chirp`.

### `output_config`

```json
{
  "output_dir": "results_{algorithm}_{doa_mode}_{steering_model}",
  "save_csv": true,
  "save_npz": true,
  "save_plots": true,
  "csv_separator": ";",
  "csv_decimal": ","
}
```

Variables:

- `output_dir`: directorio raíz de resultados. Puede ser una ruta literal o
  una plantilla con `{algorithm}`, `{doa_mode}` y `{steering_model}`.
- `save_csv`: activa tablas CSV.
- `save_npz`: activa matrices complejas comprimidas.
- `save_plots`: activa figuras PNG.
- `csv_separator`: separador de columnas.
- `csv_decimal`: carácter decimal.

Ejemplo de nombre dinámico de carpeta:

```text
results_{algorithm}_{doa_mode}_{steering_model}
```

Con `algorithm = lcmv`, `doa_mode = fixed` y `steering_model = ideal`,
la salida se crea en:

```text
results_lcmv_fixed_ideal
```

## Salidas generadas

En `output_dir`:

```text
config_used.json                Copia exacta de la configuración parseada.
run_log.txt                     Resumen textual de la ejecución.
null_metrics_by_jammer.csv      Métricas por Monte Carlo, jammer y umbral.
pattern_global_azimuth_dB.png   Corte global de azimut, si save_plots=true.
pattern_global_elevation_dB.png Corte global de elevación, si save_plots=true.
array_geometry.png              Geometría del array, si save_plots=true.
array_factor_heatmap.png        Mapa 2D azimut/elevación, si save_plots=true.
pattern_3d_comparison.png       Superficie 3D del patrón, si save_plots=true.
temporal_fft_snapshot_spectrum.png Espectro medio, si save_plots=true.
jammer_plots/                   Cortes individuales por jammer.
```

En `output_dir/output_data`, si corresponde:

```text
element_positions_m.csv         Posiciones XYZ de los elementos.
jammer_table.csv                Jammers generados en la primera iteración.
matrices_complex.npz            Snapshots, covarianza y pesos complejos.
temporal_fft_snapshot_spectrum.csv Espectro temporal medio.
jammer_cuts/                    Cortes azimut/elevación por jammer.
```

`matrices_complex.npz` contiene:

- `snapshot_matrix`
- `covariance_matrix`
- `selected_weights`
- `conventional_weights`

## Métricas de nulo

`null_metrics_by_jammer.csv` contiene una fila por:

```text
Monte Carlo x jammer x umbral de atenuación
```

Columnas principales:

- `montecarlo_index`
- `algorithm`
- `doa_mode`
- `jammer_azimuth_deg`
- `jammer_elevation_deg`
- `jammer_jnr_dB`
- `jammer_signal_type`
- `null_depth_dB`
- `attenuation_threshold_dB`
- `null_area_cells`
- `null_area_deg2`
- `null_width_azimuth`
- `null_width_elevation`

La tabla conserva las metricas 2D alrededor de la direccion del jammer para
cada umbral configurado.

## Tests

Instalación de dependencias de desarrollo:

```bash
python -m pip install -r tests/requirements-dev.txt
```

Ejecución:

```bash
python -m pytest tests
```

La suite cubre:

- parseo y validación de configuración;
- geometría CRPA y steering vector ideal;
- generación de ruido y señales jammer;
- covarianza, diagonal loading y pseudoinversa;
- pesos `power_inversion` y `lcmv`;
- evaluación de patrones 1D/2D;
- métricas de profundidad, anchura y área 2D de nulo;
- creación de salidas principales;
- plots con backend `Agg` de Matplotlib.

## Limitaciones actuales

- Solo se implementa geometría `hexagonal_7`.
- Se soporta steering `ideal` y steering `measured` por vecino mas cercano
  sobre un CSV con amplitud/fase o real/imag por elemento.
- El modelo `measured` no interpola entre muestras; usa la direccion medida
  mas cercana, por lo que el paso de scan no debe ser mas fino que la malla
  real salvo que se quiera sobremuestrear para visualizacion.
- `bandwidth_hz` existe como campo de configuración, pero no se usa todavía
  en la generación de señal.
- LCMV no admite más de `num_elements - 1` jammers.

## Sustitución futura por CRPA real

El punto principal de sustitución está en:

```text
crpa_sim/array_model.py -> steering_vector()
```

Actualmente se selecciona con:

```text
steering_model = "ideal" | "measured"
```

Una integracion con datos reales mas completa podria extender esa funcion
para interpolar steering vectors medidos, patrones de elemento, errores de
calibracion o acoplo mutuo.
