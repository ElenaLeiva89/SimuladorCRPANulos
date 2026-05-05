# Banco de pruebas adaptado - versión limpia Navshield CRPA

Adaptado a la versión actual:
- `simulation_config.doa_mode`: valor único (`fixed` o `variable`)
- `beamforming_config.algorithm`: valor único (`lcmv` o `power_inversion`)
- `jammer_config.num_jammers` y `jammer_config.jnr_dB`: valores únicos
- `compute_weights(config, snapshot_matrix, element_positions_m, jammer_list)`

Copia `tests/`, `pytest.ini` y `requirements-dev.txt` en la raíz del proyecto y ejecuta:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```
