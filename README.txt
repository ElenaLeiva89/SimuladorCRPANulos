# Simulador CRPA de 7 elementos para nullforming y beamforming

Este proyecto simula una antena CRPA de 7 elementos en geometria hexagonal
para estudiar formacion de haces y generacion de nulos frente a interferencias
GNSS. El simulador genera ruido, jammers, snapshots complejos, pesos
adaptativos y metricas de profundidad, anchura y area de nulo.

La version actual ejecuta una combinacion concreta por corrida:

- un modo DoA: `fixed` o `variable`;
- un algoritmo: `power_inversion`, `lcmv` o `lcmvq`;
- un modelo de steering: `ideal` o `measured`;
- un valor comun de JNR para todos los jammers activos;
- una geometria CRPA hexagonal de 7 elementos.

## Flujo de ejecucion

`main.py` ejecuta la simulacion completa:

1. Lee y actualiza el JSON de configuracion seleccionado.
2. Valida rangos fisicos y opciones soportadas.
3. Construye la geometria CRPA con `crpa_sim/array_model.py`.
4. Genera jammers y ruido con `crpa_sim/jammers.py`.
5. Calcula la matriz de snapshots recibidos `X`.
6. Estima la covarianza espacial `R = X X^H / L` cuando el algoritmo la usa.
7. Calcula pesos con `power_inversion`, `lcmv` o `lcmvq`.
8. Evalua cortes azimut/elevacion y mallas 2D del patron.
9. Calcula metricas de nulo por jammer, umbral y Monte Carlo.
10. Guarda logs, CSV, NPZ y figuras segun `output_config`.

## Instalacion

Dependencias minimas:

```bash
python -m pip install numpy pandas matplotlib scipy
```

Dependencias de desarrollo y tests:

```bash
python -m pip install -r tests/requirements-dev.txt
```

## Ejecucion

La CLI exige seleccionar explicitamente el modelo de steering.

Modo ideal:

```bash
python main.py --steering-model ideal
```

Modo medido desde tablas MATLAB:

```bash
python main.py --steering-model measured --phase-mat data/TABLASFASE_E1_C_LBADICIONALES_ALT.mat --amplitude-mat data/TABLASAMPL_E1_C_LBADICIONALES_ALT.mat
```

Con un JSON alternativo:

```bash
python main.py --config input_config.json --steering-model ideal
```

Notas importantes:

- `--steering-model measured` requiere siempre `--phase-mat` y
  `--amplitude-mat`.
- Antes de lanzar la simulacion, la CLI actualiza `array_config` dentro del
  JSON indicado: escribe `steering_model` y, en modo medido, las rutas
  absolutas de los dos MAT.
- En modo ideal, la CLI deja los campos MAT a `null`.

## Estructura del proyecto

```text
main.py                         Orquestacion de la simulacion.
input_config.json               Configuracion principal de entrada.
README.txt                      Documentacion del proyecto.

crpa_sim/
  array_model.py                Geometria CRPA y steering vectors.
  beamformers.py                Pesos Power Inversion, LCMV y LCMVQ.
  config.py                     Dataclasses y normalizacion del JSON.
  covariance.py                 Covarianza, diagonal loading e inversion.
  fft_tools.py                  PSD temporal media de snapshots.
  io_utils.py                   Carga, validacion y guardado de artefactos.
  jammers.py                    Ruido, senales jammer y matriz recibida.
  measured_mat.py               Lectura de tablas MAT medidas.
  null_metrics.py               Profundidad, anchura y area de nulos.
  patterns.py                   Cortes y mallas de patron espacial.
  plots.py                      Figuras de geometria, patrones y espectro.

tests/
  pytest.ini                    Configuracion de pytest.
  requirements-dev.txt          Dependencias de desarrollo.
  README_TESTS.md               Resumen del banco de pruebas.
  tests/                        Tests unitarios e integracion ligera.
```

## Modelo de array

- Geometria soportada: `hexagonal_7`.
- Numero de elementos soportado: `7`.
- Elemento 1: centro del array.
- Elementos 2 a 7: anillo hexagonal exterior.
- Plano del array: `XY`.
- Boresight esperado: elevacion `90.0` grados.
- Separacion fisica radial: `array_config.element_spacing_m`.
- Steering soportado: `ideal` analitico y `measured` desde MAT de fase y
  amplitud.

### Convencion angular

- `azimuth_deg = 0`: direccion `+X`.
- El azimut crece hacia `+Y`.
- `elevation_deg = 0`: horizonte.
- `elevation_deg = 90`: cenit.
- Los barridos configurados usan azimut `[0, 360]` y elevacion `[0, 90]`.

### Senal GNSS

Bandas soportadas:

```text
E5: 1.19179e9 Hz
E6: 1.27875e9 Hz
E1: 1.57542e9 Hz
```

Tambien se admiten codigos historicos:

```text
1 -> E5
2 -> E6
3 -> E1
```

La longitud de onda GNSS se calcula como:

```text
wavelength_m = speed_of_light_m_s / carrier_frequency_hz
```

## Steering medido

El modelo `measured` usa dos ficheros MATLAB:

- `measured_phase_mat_file`: tabla de fase con `TablasAOAFase`.
- `measured_amplitude_mat_file`: tabla de amplitud con `TablasAOAAmpli`.

Variables esperadas en el MAT de fase:

```text
TablasAOAFase
FRECSTAB_MHz
AOAsTab_Grad
ELEVSTAB_GRAD
NBITSFASE
NBITSREGI
```

Variables esperadas en el MAT de amplitud:

```text
TablasAOAAmpli
FRECSTAB_MHz
AOAsTab_Grad
ELEVSTAB_GRAD
MINDIFPA
PASODIFAMP
```

Ambos ficheros deben compartir ejes de azimut, elevacion y frecuencia, y el
nombre debe incluir la misma polarizacion mediante `_C_`, `_V_` o `_H_`.

Las tablas originales contienen 12 diferencias entre antenas. Para
beamforming se usan las seis primeras, correspondientes a las diferencias de
los elementos 2..7 respecto al elemento central. El elemento central se toma
como referencia de amplitud 1 y fase 0.

La seleccion angular usa vecino mas cercano en azimut/elevacion. La frecuencia
se interpola linealmente entre las dos muestras medidas mas cercanas:

- amplitud en dB con interpolacion lineal;
- fase por el camino angular mas corto para evitar saltos en +/-180 grados.

## Jammers

Tipos de senal soportados en `base_jammers[].signal_type`:

- `tone`: tono complejo con frecuencia normalizada.
- `complex_gaussian`: interferencia compleja gaussiana circular.
- `chirp`: chirp lineal complejo con frecuencia central normalizada.

La potencia lineal de cada jammer se calcula a partir del JNR:

```text
jammer_power_linear = noise_power_linear * 10^(jnr_dB / 10)
```

Frecuencia RF de cada jammer:

- si `center_frequency_hz` existe, se usa directamente;
- si no existe, se deriva como
  `carrier_frequency_hz + normalized_frequency * sample_rate_hz`.

La longitud de onda usada para steering, patrones y metricas de ese jammer se
calcula desde su frecuencia RF central. Esto permite evaluar jammers fuera de
la portadora GNSS nominal.

Para `chirp`, `chirp_frequency` es obligatorio y el generador actual lo trata
como frecuencia normalizada en ciclos por muestra. El campo `bandwidth_hz`
permanece reservado para modelos futuros.

## Algoritmos

`power_inversion`:

- estima la covarianza espacial a partir de `snapshot_matrix`;
- aplica diagonal loading;
- invierte con pseudoinversa;
- fuerza una restriccion sobre `power_inversion_reference_element`.

`lcmv`:

- estima la covarianza espacial;
- fuerza ganancia unitaria en la direccion deseada;
- fuerza nulos en las direcciones y frecuencias RF de los jammers;
- requiere como maximo `num_elements - 1` jammers.

`lcmvq`:

- usa restricciones geometricas sin covarianza;
- fuerza una restriccion sobre el elemento central;
- fuerza ganancia unitaria en la direccion deseada;
- fuerza nulos en las direcciones y frecuencias RF de los jammers.

Los pesos convencionales `delay-and-sum` se calculan en `patterns.py` para
comparativas internas y plots, pero no son un algoritmo seleccionable en el
JSON.

## Configuracion

El fichero `input_config.json` contiene los siguientes bloques.

### `array_config`

```json
{
  "num_elements": 7,
  "geometry": "hexagonal_7",
  "element_type": "isotropic",
  "element_spacing_m": 0.095,
  "array_boresight_elevation_deg": 90.0,
  "steering_model": "measured",
  "measured_phase_mat_file": "data/TABLASFASE_E1_C_LBADICIONALES_ALT.mat",
  "measured_amplitude_mat_file": "data/TABLASAMPL_E1_C_LBADICIONALES_ALT.mat"
}
```

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

### `simulation_config`

```json
{
  "num_montecarlo": 10,
  "doa_mode": "fixed"
}
```

En modo `fixed`, cada jammer usa el azimut/elevacion de su plantilla. En modo
`variable`, cada iteracion sortea las direcciones dentro de los rangos
configurados. El programa crea generadores aleatorios nuevos sin semilla de
configuracion, por lo que los resultados no son bit a bit reproducibles entre
ejecuciones.

### `beamforming_config`

```json
{
  "algorithm": "lcmvq",
  "desired_azimuth_deg": 0.0,
  "desired_elevation_deg": 90.0,
  "diagonal_loading_factor": 0.001,
  "power_inversion_reference_element": 1
}
```

`algorithm` admite `power_inversion`, `lcmv` o `lcmvq`.

Nota: `power_inversion_reference_element` usa indices Python de base cero. El
elemento central es `0`.

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

Con steering medido, pasos muy finos pueden multiplicar el coste de lectura y
evaluacion del patron. Conviene usar pasos mas grandes si la malla medida no
justifica el sobremuestreo.

### `noise_config`

```json
{
  "noise_power_linear": 1.0
}
```

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
      "elevation_deg": 60.0,
      "signal_type": "tone",
      "center_frequency_hz": 1578620000.0,
      "normalized_frequency": 0.05,
      "chirp_frequency": 0.05
    }
  ]
}
```

`num_jammers` debe cumplir `1 <= num_jammers <= num_elements - 1` y no puede
superar el numero de plantillas definidas en `base_jammers`.

### `output_config`

```json
{
  "output_dir": "results_{algorithm}_DOA_{doa_mode}_CRPA_{steering_model}_1Jammers",
  "save_csv": true,
  "save_npz": false,
  "save_plots": true,
  "csv_separator": ";",
  "csv_decimal": ","
}
```

`output_dir` puede incluir las claves dinamicas `{algorithm}`, `{doa_mode}` y
`{steering_model}`.

## Salidas generadas

En `output_dir`:

```text
config_used.json                  Copia de la configuracion parseada.
run_log.txt                       Resumen textual de la ejecucion.
null_metrics_by_jammer.csv        Metricas por Monte Carlo, jammer y umbral.
pattern_global_azimuth_dB.png     Corte global de azimut, si save_plots=true.
pattern_global_elevation_dB.png   Corte global de elevacion, si save_plots=true.
array_geometry.png                Geometria del array, si save_plots=true.
pattern_3d_comparison.png         Superficie 3D del patron, si save_plots=true.
temporal_psd_spectrum.png         PSD temporal media, si save_plots=true.
jammer_plots/                     Cortes y heatmaps individuales por jammer.
```

En `output_dir/output_data`, si corresponde:

```text
element_positions_m.csv           Posiciones XYZ de los elementos.
jammer_table.csv                  Jammers generados en la primera iteracion.
matrices_complex.npz              Snapshots, covarianza y pesos complejos.
jammer_cuts/                      Cortes azimut/elevacion por jammer.
```

`matrices_complex.npz` contiene:

- `snapshot_matrix`
- `covariance_matrix`
- `selected_weights`
- `conventional_weights`

## Metricas de nulo

`null_metrics_by_jammer.csv` contiene una fila por:

```text
Monte Carlo x jammer x umbral de atenuacion
```

Columnas principales:

- `montecarlo_index`
- `algorithm`
- `doa_mode`
- `jammer_azimuth_deg`
- `jammer_elevation_deg`
- `jammer_jnr_dB`
- `jammer_signal_type`
- `jammer_center_frequency_hz`
- `jammer_wavelength_m`
- `null_depth_dB`
- `attenuation_threshold_dB`
- `null_area_cells`
- `null_area_deg2`
- `null_width_azimuth`
- `null_width_elevation`

## Tests

Ejecutar la suite:

```bash
python -m pytest tests
```

La suite cubre:

- parseo y validacion de configuracion;
- geometria CRPA y steering ideal;
- carga y reconstruccion de steering medido desde MAT sinteticos;
- generacion de ruido y senales jammer;
- frecuencia RF y longitud de onda por jammer;
- covarianza, diagonal loading y pseudoinversa;
- pesos `power_inversion`, `lcmv` y `lcmvq`;
- patrones 1D/2D y PSD temporal;
- metricas de profundidad, anchura y area 2D de nulo;
- creacion de salidas principales;
- plots con backend `Agg` de Matplotlib;
- integracion ligera de `run_project` para combinaciones de steering, DoA y
  algoritmo.

## Limitaciones actuales

- Solo se implementa geometria `hexagonal_7`.
- El modelo `measured` usa vecino mas cercano en azimut/elevacion.
- `bandwidth_hz` existe como campo de configuracion, pero no se usa todavia
  en la generacion de senal.
- `lcmv` no admite mas de `num_elements - 1` jammers.
- El flujo activo de steering medido usa MAT de fase/amplitud; cualquier helper
  historico de CSV no forma parte de la ejecucion principal.
