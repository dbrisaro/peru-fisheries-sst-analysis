# Análisis de Pérdidas por Cierres de Puertos por Anomalías SST

Este proyecto analiza el impacto económico de los cierres de puertos peruanos causados por anomalías en la temperatura superficial del mar (SST).

## Estructura del Proyecto

```
peru_produccion/
├── data/                    # Datos del proyecto (no incluidos en el repositorio)
│   ├── MODIST/             # Datos de temperatura superficial del mar
│   ├── imarpe/             # Datos de producción pesquera
│   └── puertos/            # Datos de puertos
├── results/                 # Resultados del análisis (no incluidos en el repositorio)
├── src/                     # Código fuente
│   ├── port_sst_analysis_functions.py
│   ├── port_sst_analysis_quick.py
│   └── port_sst_analysis_weekly.py
├── requirements.txt         # Dependencias del proyecto
└── README.md               # Este archivo
```

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/peru_produccion.git
cd peru_produccion
```

2. Crear y activar un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Datos Requeridos

Para ejecutar el análisis, necesitarás los siguientes archivos de datos:

1. **Datos de SST**:
   - Ubicación: `data/MODIS/processed/sst_anomaly_daily_2002_2025.nc`
   - Contenido: Anomalías diarias de SST para el período 2002-2025
   - Formato: NetCDF (.nc)

2. **Datos de Producción**:
   - Ubicación: `data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv`
   - Contenido: Datos de producción pesquera diaria por puerto
   - Formato: CSV

3. **Datos de Puertos**:
   - Ubicación: `data/puertos/processed/ports_with_normal_angles_corrected.xlsx`
   - Contenido: Información de puertos y sus ángulos normales
   - Formato: Excel (.xlsx)

Por favor, contacta al administrador del proyecto para obtener acceso a estos archivos de datos.

## Uso

1. Asegúrate de tener todos los archivos de datos necesarios en las ubicaciones correctas.

2. Para análisis diario:
```bash
python src/port_sst_analysis_quick.py
```

3. Para análisis semanal:
```bash
python src/port_sst_analysis_weekly.py
```

## Resultados

Los resultados se guardarán en el directorio `results/` con la siguiente estructura:

```
results/
├── port_sst_analysis_daily/
│   ├── annual_losses.csv
│   └── annual_losses_exceedance.png
└── port_sst_analysis_weekly/
    ├── annual_losses.csv
    └── annual_losses_exceedance.png
```

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 