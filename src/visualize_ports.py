import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Leer los datos de puertos
df = pd.read_csv('results/pa_analysis/puertos_maritimos_con_coordenadas.csv')

# Filtrar puertos con coordenadas válidas (excluyendo OTROS y OTROS PUERTOS)
df = df[(df['LATITUD'] != 0) & (df['LONGITUD'] != 0)]

# Crear figura
fig = plt.figure(figsize=(12, 15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Añadir características del mapa
ax.add_feature(cfeature.LAND, color='lightgrey')
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

# Plot puertos
scatter = ax.scatter(df['LONGITUD'], df['LATITUD'], 
                    s=50, c='blue', marker='o', alpha=0.6,
                    transform=ccrs.PlateCarree())

# Añadir etiquetas de puertos
for i, row in df.iterrows():
    ha = 'right' if i % 2 == 0 else 'left'  # Alternar alineación
    ax.text(row['LONGITUD'], row['LATITUD'], row['PUERTO'],
            fontsize=8, ha=ha, color='black', alpha=0.7,
            transform=ccrs.PlateCarree())

# Configurar leyenda
ports_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                            markersize=8, label="Puertos")

# Configurar ejes
ax.set_extent([-85, -70, -20, -2])
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
ax.set_title("Puertos Marítimos del Perú", loc='left', pad=20, fontsize=14)
ax.legend(handles=[ports_legend], frameon=False, loc='lower left')

# Añadir grid
ax.gridlines(draw_labels=True, alpha=0.3)

# Guardar figura
plt.tight_layout()
plt.savefig('results/pa_analysis/mapa_puertos.png', dpi=300, bbox_inches='tight')
plt.close()

print("Mapa guardado en: results/pa_analysis/mapa_puertos.png") 