#!/bin/bash

# ========== CONFIGURACIÓN ==========
# Directorios de entrada y salida
INPUT_DIR="../data/MODIS_chl/raw"      # Carpeta con archivos originales
OUTPUT_DIR="../data/MODIS_chl/processed"  # Carpeta donde guardamos los resultados

# Rango de años para el dataset completo
START_YEAR=2003
END_YEAR=2024

# Rango de años para la climatología
CLIM_START_YEAR=2003
CLIM_END_YEAR=2023

# Crear directorio de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Archivos de salida (Diarios)
MERGED_FILE="$OUTPUT_DIR/chl_merged_daily_complete.nc"
FULL_YEARS_FILE="$OUTPUT_DIR/chl_merged_full_years_daily_${START_YEAR}_${END_YEAR}.nc"
CLIMATOLOGY_FILE="$OUTPUT_DIR/chl_climatology_daily_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"

# Archivos de salida (Mensuales)
MERGED_MONTHLY_FILE="$OUTPUT_DIR/chl_merged_monthly_${START_YEAR}_${END_YEAR}.nc"
CLIMATOLOGY_MONTHLY_FILE="$OUTPUT_DIR/chl_climatology_monthly_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"

echo "🚀 Iniciando procesamiento de CHL..."

# 2. Seleccionar solo los años completos
echo "📌 Filtrando años completos (${START_YEAR}-${END_YEAR})..."
cdo seldate,${START_YEAR}-01-01,${END_YEAR}-12-31 "$MERGED_FILE" "$FULL_YEARS_FILE"
echo ""

# 3. Extraer el período para la climatología diaria
echo "📌 Extrayendo período ${CLIM_START_YEAR}-${CLIM_END_YEAR} para climatología diaria..."
CLIM_TEMP_FILE="$OUTPUT_DIR/chl_clim_temp_daily_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
cdo seldate,${CLIM_START_YEAR}-01-01,${CLIM_END_YEAR}-12-31 "$FULL_YEARS_FILE" "$CLIM_TEMP_FILE"
echo ""

# 4. Calcular la climatología diaria
echo "📌 Calculando climatología diaria (${CLIM_START_YEAR}-${CLIM_END_YEAR})..."
cdo ydaymean "$CLIM_TEMP_FILE" "$CLIMATOLOGY_FILE"
cdo runmean,10 "$CLIMATOLOGY_FILE" "$CLIMATOLOGY_FILE_RUNNING"

rm "$CLIM_TEMP_FILE"  # Borrar archivo temporal
echo ""

echo "📌 Archivos generados: $MERGED_FILE, $FULL_YEARS_FILE, $CLIMATOLOGY_FILE"
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
CLIM_TEMP_MONTHLY_FILE="$OUTPUT_DIR/chl_clim_temp_monthly_${CLIM_START_YEAR}_${CLIM_END_YEAR}.nc"
cdo seldate,${CLIM_START_YEAR}-01-01,${CLIM_END_YEAR}-12-31 "$MERGED_MONTHLY_FILE" "$CLIM_TEMP_MONTHLY_FILE"
echo ""

# 8. Calcular la climatología mensual
echo "📌 Calculando climatología mensual (${CLIM_START_YEAR}-${CLIM_END_YEAR})..."
cdo ymonmean "$CLIM_TEMP_MONTHLY_FILE" "$CLIMATOLOGY_MONTHLY_FILE"
rm "$CLIM_TEMP_MONTHLY_FILE"  # Borrar archivo temporal
echo ""


echo "📌 Archivos generados: $MERGED_MONTHLY_FILE, $CLIMATOLOGY_MONTHLY_FILE"
echo ""

echo "✅ ¡Procesamiento completado! Resultados guardados en: $OUTPUT_DIR"

