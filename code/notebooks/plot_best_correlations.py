import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_strongest_correlation(df_combined, results_df):
    """
    Creates a scatter plot for the strongest negative correlation found in the results.
    """
    # Sort by R-value ascending (most negative first)
    sorted_results = results_df.sort_values('R-value', ascending=True)
    
    # Get the strongest negative correlation
    best_correlation = sorted_results.iloc[0]
    
    # Extract the relevant data
    cluster = best_correlation['Cluster']
    variable = best_correlation['Variable']
    r_value = best_correlation['R-value']
    p_value = best_correlation['P-value']
    
    # Prepare the data for plotting
    plot_data = df_combined[[cluster, variable]].dropna()
    plot_data = plot_data[plot_data[variable] != 0]
    plot_data = plot_data[plot_data[variable] != np.inf]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x=cluster, y=variable, alpha=0.6)
    
    # Add regression line
    sns.regplot(data=plot_data, x=cluster, y=variable, 
                scatter=False, color='red', line_kws={'linestyle': '--'})
    
    # Customize the plot
    plt.title(f'Strongest Negative Correlation\n{cluster} vs {variable}\nR = {r_value:.3f}, p = {p_value:.3f}')
    plt.xlabel(f'Cluster Time Series ({cluster})')
    plt.ylabel(variable)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    return plt.gcf()

# Example usage:
# fig = plot_strongest_correlation(df_combined, results_df)
# plt.show() 