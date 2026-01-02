import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import Set2

df = pd.read_csv('data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['week'] = df['date'].dt.isocalendar().week

df_2024 = df[df['year'] == 2024]
selected_ports = ['Chimbote', 'Callao', 'Supe', 'Tambo de Mora']
df_selected = df_2024[['date', 'week'] + selected_ports]

weekly_data = df_selected.groupby('week')[selected_ports].sum().reset_index()

fig, ax1 = plt.subplots(figsize=(15, 8))
sns.set_style("whitegrid")

colors = [Set2(0), Set2(1), Set2(2), Set2(3)]
for port, color in zip(selected_ports, colors):
    ax1.plot(weekly_data['week'], weekly_data[port], 
             label=port, linewidth=1.5, color=color, marker='o', markersize=4)

ax1.set_xlabel('Semana del año', fontsize=12)
ax1.set_ylabel('Toneladas', fontsize=12)

month_weeks = [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52]
month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic', 'Ene']

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(month_weeks)
ax2.set_xticklabels(month_names)

week_ticks = list(range(0, 53, 4))
ax1.set_xticks(week_ticks)
ax1.grid(True, axis='y', alpha=0.3)

for week in month_weeks:
    ax1.axvline(x=week, color='gray', alpha=0.3, linestyle='-')

ax1.legend(fontsize=12, title='Puertos', title_fontsize=14, frameon=False, loc='upper left')

ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax1.set_xlim(0, weekly_data['week'].max() + 1)
ax2.set_xlim(0, weekly_data['week'].max() + 1)

print("\nProducción total por puerto en 2024 (por semanas):")
for port in selected_ports:
    total = weekly_data[port].sum()
    print(f"{port}: {total:,.2f} toneladas")
    max_week = weekly_data.loc[weekly_data[port].idxmax(), 'week']
    max_prod = weekly_data[port].max()
    print(f"  Máxima producción: {max_prod:,.2f} toneladas (semana {max_week})")
    print()

plt.tight_layout()
plt.savefig('results/weekly_production_2024.png', dpi=300, bbox_inches='tight')
plt.close() 