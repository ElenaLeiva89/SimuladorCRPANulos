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

Actualización fase 2:
  - Se añade crpa_sim/covariance.py para separar R = X X^H / L de la geometría.
  - crpa_sim/beamformers.py calcula pesos conventional, LCMV y Power Inversion.
  - main.py calcula patrones siempre como B=w^H a(az,el).
  - plot_array_geometry numera centro=1, derecha(+X)=2 y resto en sentido horario.
