#!/bin/bash

# ========== CONFIGURACIÓN ==========
# Directorios de entrada y salida
INPUT_DIR="../data/ocean_data_sst/raw"      # Carpeta con archivos originales
OUTPUT_DIR="../data/ocean_data_sst/processed"  # Carpeta donde guardamos los resultados

# Rango de años para el dataset completo
START_YEAR=1982
END_YEAR=2024

# Rango de años para la climatología
CLIM_START_YEAR=1990
CLIM_END_YEAR=2020

# Crear directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Archivos de salida (Diarios)
MERGED_FILE="$OUTPUT_DIR/sst_merged_daily_${START_YEAR}_${END_YEAR}.nc"
FULL_YEARS_FILE="$OUTPUT_DIR/sst_merged_full_years_daily_${START_YEAR}_${END_YEAR}.nc"
CLIMATOLOGY_FILE="$OUTPUT_DIR/sst_climatology_daily_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
ANOMALY_FILE="$OUTPUT_DIR/sst_anomaly_daily_${START_YEAR}_${END_YEAR}.nc"

# Archivos de salida (Mensuales)
MERGED_MONTHLY_FILE="$OUTPUT_DIR/sst_merged_monthly_${START_YEAR}_${END_YEAR}.nc"
CLIMATOLOGY_MONTHLY_FILE="$OUTPUT_DIR/sst_climatology_monthly_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
ANOMALY_MONTHLY_FILE="$OUTPUT_DIR/sst_anomaly_monthly_${START_YEAR}_${END_YEAR}.nc"

echo "🚀 Iniciando procesamiento de SST..."

# 1. Unir todos los archivos .nc de la carpeta raw
echo "📌 Uniendo archivos diarios de $INPUT_DIR..."
cdo cat $(ls "$INPUT_DIR"/sst.day.mean.*.subset.nc | sort -V) "$MERGED_FILE"
echo ""

# 2. Seleccionar solo los años completos
echo "📌 Filtrando años completos (${START_YEAR}-${END_YEAR})..."
cdo seldate,${START_YEAR}-01-01,${END_YEAR}-12-31 "$MERGED_FILE" "$FULL_YEARS_FILE"
echo ""

# 3. Extraer el período para la climatología diaria
echo "📌 Extrayendo período ${CLIM_START_YEAR}-${CLIM_END_YEAR} para climatología diaria..."
CLIM_TEMP_FILE="$OUTPUT_DIR/sst_clim_temp_daily_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
cdo seldate,${CLIM_START_YEAR}-01-01,${CLIM_END_YEAR}-12-31 "$FULL_YEARS_FILE" "$CLIM_TEMP_FILE"
echo ""

# 4. Calcular la climatología diaria
echo "📌 Calculando climatología diaria (${CLIM_START_YEAR}-${CLIM_END_YEAR})..."
cdo ydaymean "$CLIM_TEMP_FILE" "$CLIMATOLOGY_FILE"
rm "$CLIM_TEMP_FILE"  # Borrar archivo temporal
echo ""

# 5. Calcular la anomalía diaria
echo "📌 Calculando anomalía diaria (${START_YEAR}-${END_YEAR})..."
cdo ydaysub "$FULL_YEARS_FILE" "$CLIMATOLOGY_FILE" "$ANOMALY_FILE"
echo ""

echo "📌 Archivos generados: $MERGED_FILE, $FULL_YEARS_FILE, $CLIMATOLOGY_FILE, $ANOMALY_FILE."
echo ""
echo ""


# =============================================
# ============ CÁLCULOS MENSUALES =============
# =============================================

echo "📌 Calculando valores mensuales..."

# 6. Calcular la media mensual de los datos originales
echo "📌 Calculando datos mensuales promedio (${START_YEAR}-${END_YEAR})..."
cdo monmean "$FULL_YEARS_FILE" "$MERGED_MONTHLY_FILE"
echo ""

# 7. Extraer el período de referencia para la climatología mensual
echo "📌 Extrayendo período ${CLIM_START_YEAR}-${CLIM_END_YEAR} para climatología mensual..."
CLIM_TEMP_MONTHLY_FILE="$OUTPUT_DIR/sst_clim_temp_monthly_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
cdo seldate,${CLIM_START_YEAR}-01-01,${CLIM_END_YEAR}-12-31 "$MERGED_MONTHLY_FILE" "$CLIM_TEMP_MONTHLY_FILE"
echo ""

# 8. Calcular la climatología mensual
echo "📌 Calculando climatología mensual (${CLIM_START_YEAR}-${CLIM_END_YEAR})..."
cdo ymonmean "$CLIM_TEMP_MONTHLY_FILE" "$CLIMATOLOGY_MONTHLY_FILE"
rm "$CLIM_TEMP_MONTHLY_FILE"  # Borrar archivo temporal
echo ""

# 9. Calcular la anomalía mensual
echo "📌 Calculando anomalía mensual (${START_YEAR}-${END_YEAR})..."
cdo ymonsub "$MERGED_MONTHLY_FILE" "$CLIMATOLOGY_MONTHLY_FILE" "$ANOMALY_MONTHLY_FILE"
echo ""

echo "📌 Archivos generados: $MERGED_MONTHLY_FILE, $CLIMATOLOGY_MONTHLY_FILE, $ANOMALY_MONTHLY_FILE."
echo ""

echo "✅ ¡Procesamiento completado! Resultados guardados en: $OUTPUT_DIR"

