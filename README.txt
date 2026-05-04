Simulación CRPA de 7 elementos con jammers, ruido y pesos conventional/LCMV/power_inversion.

Ejecución:
  pip install numpy pandas matplotlib
  python main.py

Ficheros principales:
  main.py                         Punto de entrada.
  input_config.json               Configuración editable.
  crpa_sim/config.py              Dataclasses y bandas GNSS.
  crpa_sim/crpa_array.py          Geometría, steering vector y patrones.
  crpa_sim/covariance.py          Covarianza espacial, carga diagonal e inversión estable.
  crpa_sim/jammers.py             Ruido e interferencias.
  crpa_sim/beamformers.py         Pesos conventional, LCMV y Power Inversion.
  crpa_sim/fft_tools.py           FFT temporal y FFT espacial ULA opcional.
  crpa_sim/plots.py               Figuras.
  crpa_sim/io_utils.py            Guardado/carga/logs.

Nota sobre FFT:
  - La FFT temporal se aplica a snapshot_matrix sobre el eje temporal para ver tonos o espectro.
  - La FFT espacial 1D solo es directa para arrays lineales uniformes ULA.
  - Para la CRPA hexagonal 2D, el patrón principal debe calcularse con barrido angular: B=w^H a(az,el).

## Formato de `input_config.json`

El fichero debe contener tres secciones principales:

- `simulation_config`: parámetros de la simulación global.
- `scenario_config`: parámetros de la CRPA y del barrido.
- `jammer_list`: lista de jammers a simular.

### simulation_config

- `nsimulations`: Número de iteraciones de Monte Carlo. Ejemplo: `10000`.
- `gnssBand`: Banda GNSS. Puede ser un número `1`, `2`, `3`, o la etiqueta `"E5"`, `"E6"`, `"E1"`.
- `maxPhaseNoise_deg`: Ruido de fase máximo en grados.
- `maxAmplNoise_dB`: Ruido de amplitud máximo en dB.
- `interferenceType`: Lista de tipos de interferencia. Actualmente sólo se usa como etiqueta interna, por ejemplo `[1]`.
- `algorithmType`: Tipo de algoritmo a aplicar para calculo de los pesos. Puede ser "conventional", "LCMW", "power_inversion". 

### scenario_config

- `speed_of_light_m_s`: Velocidad de la luz en m/s. Normalmente `299792458.0`.
- `num_elements`: Número de elementos de la CRPA. Debe ser `7` para este proyecto.
- `element_spacing_over_lambda`: Separación radial exterior en longitudes de onda.
- `num_snapshots`: Número de snapshots temporales.
- `noise_power_linear`: Potencia de ruido lineal por elemento.
- `desired_azimuth_deg`: Azimut deseado del haz principal en grados.
- `desired_elevation_deg`: Elevación deseada del haz principal en grados.
- `azimuth_scan_min_deg`: Ángulo mínimo del barrido de azimut.
- `azimuth_scan_max_deg`: Ángulo máximo del barrido de azimut.
- `azimuth_scan_step_deg`: Paso del barrido de azimut.
- `fixed_azimuth_cut_deg`: Azimut fijo para el corte de elevación.
- `elevation_scan_min_deg`: Elevación mínima del barrido.
- `elevation_scan_max_deg`: Elevación máxima del barrido.
- `elevation_scan_step_deg`: Paso del barrido de elevación.
- `random_seed`: Semilla aleatoria para reproducibilidad.
- `output_dir`: Carpeta de salida.

### jammer_list

Cada elemento de la lista define un jammer:

- `name`: Nombre identificador.
- `azimuth_deg`: Azimut de llegada del jammer en grados.
- `elevation_deg`: Elevación de llegada del jammer en grados.
- `jnr_dB`: Jamming-to-noise ratio en dB.
- `signal_type`: Tipo de señal. Puede ser `"complex_gaussian"` o `"tone"`.
- `normalized_frequency`: Frecuencia normalizada para señales de tipo `tone`.
