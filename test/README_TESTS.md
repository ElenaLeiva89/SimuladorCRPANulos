# Banco de pruebas Navshield CRPA

Coloca la carpeta `tests/`, `pytest.ini` y `requirements-dev.txt` en la raíz del proyecto, al mismo nivel que `main.py`, `input_config.json` y la carpeta `crpa_sim/`.

## Instalar dependencias

```bash
python -m pip install -r requirements-dev.txt
```

## Ejecutar todo

```bash
python -m pytest
```

## Ejecutar un fichero concreto

```bash
python -m pytest tests/test_beamformers.py
```

## Nota importante

Estos tests están pensados para tu estructura actual:

```text
main.py
input_config.json
crpa_sim/
  beamformers.py
  config.py
  covariance.py
  crpa_array.py
  fft_tools.py
  io_utils.py
  jammers.py
  plots.py
tests/
```
