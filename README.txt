# Simulador CRPA ideal de 7 elementos para Nullforming/Beamforming

Versión limpia y conservadora del simulador.

## Qué hace

- Simula una CRPA ideal hexagonal de 7 elementos, orientada al cenit.
- Genera jammers y ruido térmico no coherente.
- Ejecuta **un único modo DoA por ejecución**: `fixed` o `variable`.
- Ejecuta **un único algoritmo por ejecución**: `power_inversion` o `lcmv`.
- Genera logs y ficheros de salida similares a la versión anterior.
- Genera plots globales y plots por jammer.
- Genera un CSV final con profundidad y anchuras de nulo por jammer y umbral.

## Ejecución

```bash
pip install numpy pandas matplotlib
python main.py
```

## Configuración principal

En `input_config.json`:

```json
"simulation_config": {
  "num_montecarlo": 20,
  "random_seed": 12345,
  "doa_mode": "fixed"
}
```

`doa_mode` solo acepta:

- `fixed`
- `variable`

```json
"beamforming_config": {
  "algorithm": "lcmv"
}
```

`algorithm` solo acepta:

- `lcmv`
- `power_inversion`

El número de jammers se controla aquí:

```json
"jammer_config": {
  "num_jammers": 3,
  "jnr_dB": 40.0
}
```

`num_jammers` debe ser `1 <= num_jammers <= num_elements - 1`.

## Salidas principales

En `results_crpa_nullforming/`:

- `run_log.txt`
- `config_used.json`
- `element_positions_m.csv`
- `jammer_table.csv`
- `matrices_complex.npz`
- `null_metrics_by_jammer.csv`
- `null_metrics_summary.csv`
- `pattern_global_azimuth_dB.png`
- `pattern_global_elevation_dB.png`
- `array_factor_heatmap.png`
- `pattern_3d_comparison.png`
- `temporal_fft_snapshot_spectrum.csv`
- `temporal_fft_snapshot_spectrum.png`
- `jammer_cuts/`
- `jammer_plots/`

## Plots por jammer

Para cada jammer se genera:

```text
jammer_plots/jammer_X_NAME/
  pattern_azimuth_dB.png
  pattern_elevation_dB.png
  array_factor_heatmap.png
  pattern_3d_comparison.png
```

Los cortes por jammer atraviesan el nulo:

- corte de azimut con elevación fija igual a la elevación del jammer;
- corte de elevación con azimut fijo igual al azimut del jammer.

## CSV de métricas

`null_metrics_by_jammer.csv` contiene:

- `montecarlo_index`
- `algorithm`
- `doa_mode`
- `num_jammers`
- `jammer_index`
- `jammer_name`
- `jammer_azimuth_deg`
- `jammer_elevation_deg`
- `jammer_jnr_dB`
- `jammer_signal_type`
- `null_depth_dB`
- `attenuation_threshold_dB`
- `null_width_azimuth_deg`
- `null_width_elevation_deg`

## Sustitución futura por CRPA real

El punto de sustitución está en:

```text
crpa_sim/array_model.py -> steering_vector()
```

Actualmente `steering_model = "ideal"`.
