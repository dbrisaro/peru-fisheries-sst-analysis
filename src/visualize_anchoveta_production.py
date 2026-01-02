import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines

df = pd.read_csv('data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv')
df['year'] = pd.to_datetime(df['date']).dt.year
df_2024 = df[df['year'] == 2024]

port_names = {
    'Chicama': 'CHICAMA', 'Chimbote': 'CHIMBOTE', 'Callao': 'CALLAO',
    'Chancay': 'CHANCAY', 'Supe': 'SUPE', 'Tambo de Mora': 'TAMBO DE MORA',
    'Vegueta': 'VEGUETA', 'Huacho': 'HUACHO', 'Pisco': 'PISCO',
    'Parachique': 'PARACHIQUE', 'Ilo': 'ILO', 'Samanco': 'SAMANCO',
    'Mollendo': 'MOLLENDO', 'Salaverry': 'SALAVERRY', 'Atico': 'ATICO',
    'Quilca': 'QUILCA', 'Planchada': 'LA PLANCHADA', 'Huarmey': 'HUARMEY',
    'Casma': 'CASMA', 'Paita': 'PAITA'
}

production_data = df_2024.iloc[:, 1:].sum()
production_data = production_data[production_data.index != 'year']

print("\nProducción total por puerto en 2024:")
for port, tons in sorted(zip(production_data.index, production_data.values), key=lambda x: x[1], reverse=True):
    print(f"{port}: {tons:,.2f} toneladas")

anchoveta = pd.DataFrame({
    'LUGAR_DESEMBARQUE': [port_names[col] for col in production_data.index],
    'produccion_total': production_data.values
}).reset_index(drop=True)

coords = pd.read_csv('data/imarpe/processed/coordenadas_puertos.csv')
data = pd.merge(anchoveta, coords.rename(columns={'PUERTO': 'LUGAR_DESEMBARQUE'}), 
                on='LUGAR_DESEMBARQUE', how='inner')
data = data[(data['LATITUD'] != 0) & (data['LONGITUD'] != 0)]

fig = plt.figure(figsize=(12, 15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color='lightgrey')
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

ranges = [0, 100000, 500000, 1000000, float('inf')]
sizes = [50, 400, 1600, 3200]
labels = ['< 100,000', '100,000-500,000', '500,000-1,000,000', '> 1,000,000']
legend_elements = []

for i in range(len(ranges)-1):
    mask = (data['produccion_total'] > ranges[i]) & (data['produccion_total'] <= ranges[i+1])
    if any(mask):
        ax.scatter(data.loc[mask, 'LONGITUD'], data.loc[mask, 'LATITUD'],
                  s=sizes[i], c='red', alpha=0.6, transform=ccrs.PlateCarree())
        ax.scatter(data.loc[mask, 'LONGITUD'], data.loc[mask, 'LATITUD'],
                  s=10, c='black', alpha=1, transform=ccrs.PlateCarree())
        legend_elements.append(mlines.Line2D([], [], marker='o', color='red',
                                           label=f'{labels[i]} ton',
                                           markersize=np.sqrt(sizes[i]/10),
                                           linestyle='None', alpha=0.6))

data_sorted = data[data['produccion_total'] > 0].sort_values('LATITUD')
for i, row in data_sorted.iterrows():
    ha = 'left' if i % 2 == 0 else 'right'
    x_offset = 0.3 if ha == 'left' else -0.3
    y_offset = 0
    
    if row['LUGAR_DESEMBARQUE'] == 'TAMBO DE MORA':
        ha = 'left'
        x_offset = 0.3
    elif row['LUGAR_DESEMBARQUE'] == 'PISCO':
        ha = 'right'
        x_offset = -0.3
    elif row['LUGAR_DESEMBARQUE'] == 'SUPE':
        y_offset = 0.2
    elif row['LUGAR_DESEMBARQUE'] == 'VEGUETA':
        y_offset = -0.2
    elif row['LUGAR_DESEMBARQUE'] == 'CALLAO':
        y_offset = -0.4
    elif row['LUGAR_DESEMBARQUE'] == 'CHANCAY':
        y_offset = -0.2
    
    ax.text(row['LONGITUD'] + x_offset, row['LATITUD'] + y_offset,
            f"{row['LUGAR_DESEMBARQUE']}\n({row['produccion_total']:,.0f} ton)",
            fontsize=10, transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            ha=ha)

print("\nLímites actuales del mapa:")
extent = [-83, -70, -18, -4]
print(f"Longitud: {extent[0]} a {extent[1]}")
print(f"Latitud: {extent[2]} a {extent[3]}")

ax.set_extent(extent)
ax.legend(handles=legend_elements, title='Producción Total', frameon=False, loc='lower left')
gl = ax.gridlines(draw_labels=True, alpha=0.3)
gl.right_labels = False
gl.top_labels = False
gl.xlines = True
gl.ylines = True

plt.savefig('results/anchoveta_production_map.png', dpi=300, bbox_inches='tight')
plt.close() 