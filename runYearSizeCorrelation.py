import pandas as pd
import numpy as np
from scipy.stats import spearmanr

file_path_new = r'C:\Users\GM\Downloads\tDCS PEC Python\Classical Meta-Analysis\allData.csv'
data_new = pd.read_csv(file_path_new)
meta_data = data_new[['Name', 'Mean tDCS', 'SD tDCS', 'Mean Sham', 'SD Sham', 'Number tDSC', 'Number Sham', 'Polarity', 'Type', 'Source', 'Year']]
meta_data['Sample Size'] = meta_data['Number tDSC'] + meta_data['Number Sham']
# meta_data[['EffectSize', 'Variance']] = meta_data.apply(compute_effect_size, axis=1)
filtered_meta_data = meta_data[meta_data['Type'].isin(['ToM', 'Altruism', 'Empathy'])]

def compute_correlations(data, type_name, source):
    type_source_data = data[(data['Type'] == type_name) & (data['Source'] == source)]
    
    corr_sample_effect_size, p_value_sample_effect_size = spearmanr(type_source_data['Sample Size'], type_source_data['EffectSize'])
    corr_year_effect_size, p_value_year_effect_size = spearmanr(type_source_data['Year'], type_source_data['EffectSize'])
    corr_year_sample_size, p_value_year_sample_size = spearmanr(type_source_data['Year'], type_source_data['Sample Size'])
    
    return {
        'sample_size_vs_effect_size': (corr_sample_effect_size, p_value_sample_effect_size),
        'year_vs_effect_size': (corr_year_effect_size, p_value_year_effect_size),
        'year_vs_sample_size': (corr_year_sample_size, p_value_year_sample_size)
    }
correlation_results = {}
for type_name in filtered_meta_data['Type'].unique():
    for source in filtered_meta_data['Source'].unique():
        if len(filtered_meta_data[(filtered_meta_data['Type'] == type_name) & (filtered_meta_data['Source'] == source)]) > 0:
            correlations = compute_correlations(filtered_meta_data, type_name, source)
            correlation_results[(type_name, source)] = correlations
            print(f"\nCorrelation Results for Type: {type_name}, Source: {source}")
            print(f"Sample Size vs. Effect Size: Spearman's rho = {correlations['sample_size_vs_effect_size'][0]}, p-value = {correlations['sample_size_vs_effect_size'][1]}")
            print(f"Year vs. Effect Size: Spearman's rho = {correlations['year_vs_effect_size'][0]}, p-value = {correlations['year_vs_effect_size'][1]}")
            print(f"Year vs. Sample Size: Spearman's rho = {correlations['year_vs_sample_size'][0]}, p-value = {correlations['year_vs_sample_size'][1]}")
