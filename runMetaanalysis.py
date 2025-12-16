import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import kstest, spearmanr
import seaborn as sns

file_path_new = r'C:\Users\GM\Downloads\tDCS PEC Python\Classical Meta-Analysis\allData.csv'
data_new = pd.read_csv(file_path_new)
meta_data = data_new[['Name', 'Mean tDCS', 'SD tDCS', 'Mean Sham', 'SD Sham', 'Number tDSC', 'Number Sham', 'Polarity', 'Type', 'Source', 'Year']]

def eggers_regression_test(effect_sizes, variances):
    standard_errors = np.sqrt(variances)
    precision = 1 / standard_errors
    precision_const = sm.add_constant(precision)
    model = sm.OLS(effect_sizes, precision_const).fit()
    intercept, slope = model.params
    
    return model, intercept, model.pvalues[0]  

def plot_funnel_plot(effect_sizes, variances, type_name, source):
    standard_errors = np.sqrt(variances)
    plt.figure(figsize=(10, 6))
    plt.scatter(effect_sizes, standard_errors, alpha=0.75, label='Studies')
    mean_effect_size = np.mean(effect_sizes)
    plt.axvline(x=mean_effect_size, color='red', linestyle='--', label='Mean Effect Size')
    se_range = np.linspace(min(standard_errors), max(standard_errors), 100)
    upper_limit = mean_effect_size + 1.96 * se_range
    lower_limit = mean_effect_size - 1.96 * se_range
    plt.plot(upper_limit, se_range, 'k--', label='95% CI')
    plt.plot(lower_limit, se_range, 'k--')
    plt.gca().invert_yaxis()
    plt.xlabel('Effect Size (Hedges\' g)')
    plt.ylabel('Standard Deviation (SD)')
    plt.title(f'Funnel Plot for {type_name} - {source}')
    plt.legend()
    plt.grid(True)
    filename = f'funnel_plot_{type_name}_{source}.pdf'
    plt.savefig(filename, format='pdf')
    plt.close()

def compute_effect_size(row):
    mean_tDCS = row['Mean tDCS']
    mean_sham = row['Mean Sham']
    sd_tDCS = row['SD tDCS']
    sd_sham = row['SD Sham']
    n_tDCS = row['Number tDSC']
    n_sham = row['Number Sham']
    polarity = row['Polarity']
    pooled_sd = np.sqrt(((n_tDCS - 1) * sd_tDCS ** 2 + (n_sham - 1) * sd_sham ** 2) / (n_tDCS + n_sham - 2))
    d = (mean_tDCS - mean_sham) / pooled_sd
    d *= -polarity
    J = 1 - (3 / (4 * (n_tDCS + n_sham) - 9))
    g = d * J
    variance_d = (n_tDCS + n_sham) / (n_tDCS * n_sham) + (d ** 2) / (2 * (n_tDCS + n_sham))
    variance_g = variance_d * (J ** 2)
    return pd.Series([g, variance_g])
meta_data[['EffectSize', 'Variance']] = meta_data.apply(compute_effect_size, axis=1)
meta_data['Sample Size'] = meta_data['Number tDSC'] + meta_data['Number Sham']
filtered_meta_data = meta_data[meta_data['Type'].isin(['ToM', 'Altruism', 'Empathy'])]

def create_regression_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x, y=y, scatter_kws={'s':50}, line_kws={'color':'red'})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()

def run_meta_analysis_for_type_and_source(data, type_name, source):
    type_source_data = data[(data['Type'] == type_name) & (data['Source'] == source)]
    type_source_data = type_source_data.sort_values(by='EffectSize', ascending=True)
    meta_model = smf.mixedlm("EffectSize ~ 1", type_source_data, groups=type_source_data["Name"], re_formula="~1")
    meta_results = meta_model.fit()
    overall_effect_size = np.average(type_source_data['EffectSize'], weights=1/type_source_data['Variance'])
    qt = np.sum(((type_source_data['EffectSize'] - overall_effect_size) ** 2) / type_source_data['Variance'])
    
    ks_stat, ks_p_value = kstest(type_source_data['EffectSize'], 'norm', args=(type_source_data['EffectSize'].mean(), type_source_data['EffectSize'].std()))
    corr_sample_effect_size, p_value_sample_effect_size = spearmanr(type_source_data['Sample Size'], type_source_data['EffectSize'])
    corr_year_effect_size, p_value_year_effect_size = spearmanr(type_source_data['Year'], type_source_data['EffectSize'])
    corr_year_sample_size, p_value_year_sample_size = spearmanr(type_source_data['Year'], type_source_data['Sample Size'])
    
    return (meta_results, type_source_data, qt, ks_stat, ks_p_value, 
            corr_sample_effect_size, p_value_sample_effect_size, 
            corr_year_effect_size, p_value_year_effect_size, 
            corr_year_sample_size, p_value_year_sample_size)
types = filtered_meta_data['Type'].unique()
sources = filtered_meta_data['Source'].unique()
meta_analysis_results = {}
sorted_data_by_type_and_source = {}
heterogeneity_results = {}
ks_test_results = {}
correlation_results = {}

for type_name in types:
    for source in sources:
        if source == 'Anode':
            if len(filtered_meta_data[(filtered_meta_data['Type'] == type_name) & (filtered_meta_data['Source'] == source)]) > 0:
                results, sorted_data, qt, ks_stat, ks_p_value, corr_sample_effect_size, p_value_sample_effect_size, corr_year_effect_size, p_value_year_effect_size, corr_year_sample_size, p_value_year_sample_size = run_meta_analysis_for_type_and_source(filtered_meta_data, type_name, source)
                meta_analysis_results[(type_name, source)] = results
                sorted_data_by_type_and_source[(type_name, source)] = sorted_data
                heterogeneity_results[(type_name, source)] = qt
                ks_test_results[(type_name, source)] = (ks_stat, ks_p_value)
                correlation_results[(type_name, source)] = {
                    'sample_size_vs_effect_size': (corr_sample_effect_size, p_value_sample_effect_size),
                    'year_vs_effect_size': (corr_year_effect_size, p_value_year_effect_size),
                    'year_vs_sample_size': (corr_year_sample_size, p_value_year_sample_size)
                }
                create_regression_plot(
                    x=sorted_data['Sample Size'],
                    y=sorted_data['EffectSize'],
                    xlabel='Sample Size',
                    ylabel='Effect Size (Hedges\' g)',
                    title=f'Sample Size vs. Effect Size for {type_name} - {source}',
                    filename=f'sample_size_vs_effect_size_{type_name}_{source}.pdf'
                )
                create_regression_plot(
                    x=sorted_data['Year'],
                    y=sorted_data['Sample Size'],
                    xlabel='Year of Publication',
                    ylabel='Sample Size',
                    title=f'Year vs. Sample Size for {type_name} - {source}',
                    filename=f'year_vs_sample_size_{type_name}_{source}.pdf'
                )
                create_regression_plot(
                    x=sorted_data['Year'],
                    y=sorted_data['EffectSize'],
                    xlabel='Year of Publication',
                    ylabel='Effect Size (Hedges\' g)',
                    title=f'Year vs. Effect Size for {type_name} - {source}',
                    filename=f'year_vs_effect_size_{type_name}_{source}.pdf'
                )
                print(f"\nMeta-Analysis Results for Type: {type_name}, Source: {source}")

# Function to plot and save forest plot
def plot_forest_plot(sorted_data, type_name, source, meta_results):
    effect_sizes = sorted_data['EffectSize']
    variances = sorted_data['Variance']
    study_names = sorted_data['Name']
    ci_lower = effect_sizes - 1.96 * np.sqrt(variances)
    ci_upper = effect_sizes + 1.96 * np.sqrt(variances)
    fig, ax = plt.subplots(figsize=(12, len(effect_sizes) * 0.6))  # Reduced the figure height
    if source == 'Anode':
        thisColor = 'red'
    else :
        thisColor = 'blue'
    ax.errorbar(effect_sizes, range(len(effect_sizes)), xerr=[effect_sizes - ci_lower, ci_upper - effect_sizes], fmt='o', color=thisColor, ecolor='gray', elinewidth=3, capsize=0, markersize=12)
    ax.set_yticks(range(len(effect_sizes)))
    ax.set_yticklabels(study_names, fontname='Serif', fontsize=20)
    ax.axvline(x=0, linestyle='--', color='gray')
    ax.set_xlabel('Effect Size (g)', fontname='Serif', fontsize=14)
    ax.set_title(f'Forest Plot for {type_name} - {source}', fontname='Serif', fontsize=16)
    ax.grid(True)
    meta_effect_size = meta_results.fe_params['Intercept']
    meta_ci_lower = meta_effect_size - 1.96 * meta_results.bse['Intercept']
    meta_ci_upper = meta_effect_size + 1.96 * meta_results.bse['Intercept']
    diamond_x = [meta_ci_lower, meta_effect_size, meta_ci_upper, meta_effect_size, meta_ci_lower]
    diamond_y = [-1.5, -1, -1.5, -2, -1.5]
    ax.plot(diamond_x, diamond_y, color=thisColor, linewidth=2, label='Meta-analysis result')
    ax.fill(diamond_x, diamond_y, color=thisColor, alpha=0.1)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'forest_plot_{type_name}_{source}.pdf', format='pdf')
    plt.close()

# for type_name in types:
#     for source in sources:
#         if (type_name, source) in sorted_data_by_type_and_source:
#             plot_forest_plot(sorted_data_by_type_and_source[(type_name, source)], type_name, source, meta_analysis_results[(type_name, source)])
