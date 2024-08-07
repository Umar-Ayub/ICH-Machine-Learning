import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency

def load_data(filepath):
    """Load the dataset from a specified filepath."""
    return pd.read_csv(filepath)

def binarize_mrs90(df):
    """Binarize the MRS90 column in the dataset if not already binarized."""
    df['MRS90_binarized'] = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)
    return df

def calculate_unadjusted_associations(df, target_col):
    """Calculate and report unadjusted associations between predictors and a target variable."""
    results = {}
    for col in df.columns:
        if col != target_col and col not in ['MRS90', 'MORT90']:  # Exclude outcome variables from predictors
            # Check if the predictor is categorical or continuous
            if df[col].dtype == 'object' or len(df[col].unique()) < 10:  # Adjust based on your categorical identification criteria
                # Perform chi-squared test for categorical data
                try:
                    contingency_table = pd.crosstab(df[col], df[target_col])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    results[col] = {'Test': 'Chi-squared', 'Statistic': chi2, 'p-value': p_value}
                except Exception as e:
                    results[col] = {'Test': 'Chi-squared', 'Error': str(e)}
            else:
                # Perform Pearson correlation for continuous data
                try:
                    correlation, p_value = pearsonr(df[col], df[target_col])
                    if np.isnan(correlation):
                        results[col] = {'Test': 'Pearson', 'Error': 'NaN result likely due to low variance'}
                    else:
                        results[col] = {'Test': 'Pearson', 'Statistic': correlation, 'p-value': p_value}
                except Exception as e:
                    results[col] = {'Test': 'Pearson', 'Error': str(e)}
    return results

def display_results(results, target_col):
    """Display the calculated statistics for each predictor."""
    print(f"Results for {target_col}:")
    for predictor, stats in results.items():
        if 'Error' in stats:
            print(f"Predictor: {predictor} - Test: {stats['Test']} - Error: {stats['Error']}")
        else:
            print(f"Predictor: {predictor} - Test: {stats['Test']} - Statistic: {stats['Statistic']:.4f}, p-value: {stats['p-value']:.4f}")
    print("\n")

# Example usage
if __name__ == "__main__":
    df = load_data('../ich_data_w_scores_modified.csv')  # Specify the correct path to your data file
    df = binarize_mrs90(df)  # Binarize the MRS90 outcome if necessary
    outcomes = ['MRS90_binarized', 'MORT90']  # Define your outcome variables
    for outcome in outcomes:
        associations = calculate_unadjusted_associations(df, outcome)
        display_results(associations, outcome)
