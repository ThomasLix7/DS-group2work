import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("QM_pre-process/output.csv")
    df = df.drop(['Customer_ID', 'Source'], axis=1)
    
    return df

def statistical_model(df):
    """
    Train a logistic regression model focused on statistical interpretation
    using odds ratios.
    """
    # Separate features and target
    X = df.drop('Left', axis=1)
    y = df['Left']
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create column names DataFrame for statsmodels
    X_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X_df)
    
    # Train statsmodels logistic regression for detailed statistics
    sm_model = sm.Logit(y, X_with_const)
    sm_result = sm_model.fit(disp=0)  # disp=0 suppresses convergence messages
    
    # Print model summary
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION MODEL STATISTICAL SUMMARY")
    print("="*70)
    print(sm_result.summary())
    
    # Train scikit-learn model for visualization
    log_model = LogisticRegression(random_state=42, penalty=None, solver='newton-cg', max_iter=1000)
    log_model.fit(X_scaled, y)
    
    # Calculate and display odds ratios with confidence intervals
    print("\n" + "="*70)
    print("ODDS RATIOS AND 95% CONFIDENCE INTERVALS")
    print("="*70)
    print(f"{'Feature':<25} {'Odds Ratio':<15} {'Lower 95% CI':<15} {'Upper 95% CI':<15} {'p-value':<10}")
    print("-"*80)
    
    # Extract coefficients and p-values from statsmodels result
    params = sm_result.params
    conf_int = sm_result.conf_int()
    pvalues = sm_result.pvalues
    
    # Create feature importance DataFrame with odds ratios
    feature_importance = pd.DataFrame()
    
    # Skip the constant term (index 0)
    for i, feature in enumerate(X_with_const.columns[1:], 1):
        odds_ratio = np.exp(params.iloc[i])
        ci_lower = np.exp(conf_int.iloc[i, 0])
        ci_upper = np.exp(conf_int.iloc[i, 1])
        p_value = pvalues.iloc[i]
        
        # Add to feature importance DataFrame
        feature_importance = pd.concat([feature_importance, pd.DataFrame({
            'feature': [feature],
            'coefficient': [params.iloc[i]],
            'odds_ratio': [odds_ratio],
            'ci_lower': [ci_lower],
            'ci_upper': [ci_upper],
            'p_value': [p_value],
            'significant': [p_value < 0.05]
        })], ignore_index=True)
        
        # Add stars for significance
        stars = ''
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        
        print(f"{feature:<25} {odds_ratio:<15.3f} {ci_lower:<15.3f} {ci_upper:<15.3f} {p_value:<10.4f} {stars}")
    
    print("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    
    # Sort for plotting - by distance from 1.0 for odds ratio (smallest effect at top)
    feature_importance['distance_from_1'] = abs(feature_importance['odds_ratio'] - 1)
    feature_importance = feature_importance.sort_values('distance_from_1', ascending=True)
    
    # Plot odds ratios with confidence intervals
    plt.figure(figsize=(12, 8))
    
    # Only show top features for clarity
    top_n = 10
    top_features = feature_importance.head(top_n)
    
    # Create data for plotting
    features = top_features['feature']
    odds_ratios = top_features['odds_ratio']
    ci_lower = top_features['ci_lower']
    ci_upper = top_features['ci_upper']
    
    # Calculate error bars (distance from point to bound)
    yerr_lower = odds_ratios - ci_lower
    yerr_upper = ci_upper - odds_ratios
    
    # Create color scheme based on significance
    colors = ['#1f77b4' if p < 0.05 else '#d3d3d3' for p in top_features['p_value']]
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(features))
    
    # Plot bars
    bars = plt.barh(y_pos, odds_ratios, color=colors)
    
    # Add error bars
    plt.errorbar(odds_ratios, y_pos, xerr=[yerr_lower, yerr_upper], 
                 fmt='none', ecolor='black', capsize=5)
    
    # Add reference line at odds ratio = 1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1)
    
    # Customize the plot
    plt.yticks(y_pos, features)
    plt.xlabel('Odds Ratio (log scale)')
    plt.title('Odds Ratios with 95% Confidence Intervals')
    plt.xscale('log')  # Use log scale for better visualization
    
    # Fix x-axis formatting to avoid scientific notation
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    
    # Set x-axis limits more explicitly
    max_upper = max(ci_upper) * 1.1  # Add 10% margin
    min_lower = min(ci_lower) * 0.9  # Add 10% margin
    plt.xlim([min_lower, max_upper])
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Significant (p < 0.05)'),
        Patch(facecolor='#d3d3d3', label='Not significant')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig('logistic_odds_ratios.png')
    plt.close()
    
    # Print interpretation of key significant features
    print("\n" + "="*70)
    print("INTERPRETATION OF MODEL EFFECTS")
    print("="*70)
    
    # Create an explicit copy to avoid SettingWithCopyWarning
    significant_features = feature_importance[feature_importance['significant']].copy()
    
    print("\nKEY SIGNIFICANT EFFECTS:")
    print("-"*50)
    
    # Sort significant features by absolute effect size (largest effect first)
    significant_features['effect_magnitude'] = significant_features.apply(
        lambda row: abs(row['odds_ratio'] - 1), axis=1
    )
    significant_features = significant_features.sort_values('effect_magnitude', ascending=False)
    
    for _, row in significant_features.iterrows():
        feature = row['feature']
        coef = row['coefficient']
        odds = row['odds_ratio']
        
        # Format the effect interpretation
        if coef > 0:
            effect = f"increases"
            magnitude = f"{(odds - 1) * 100:.1f}%"
        else:
            effect = f"decreases"
            magnitude = f"{(1 - odds) * 100:.1f}%"
        
        print(f"• {feature}: A one standard deviation increase {effect} the odds of customer churn by {magnitude}.")
    
    # Return model and important results
    return {
        'statsmodels_result': sm_result,
        'sklearn_model': log_model,
        'feature_importance': feature_importance,
        'scaler': scaler
    }

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train the statistical model and get results
    results = statistical_model(df)
    
    print("\nLogistic Regression analysis complete.")
    print("Model interpretation plots saved as:")
    print("- logistic_odds_ratios.png")

if __name__ == "__main__":
    main() 