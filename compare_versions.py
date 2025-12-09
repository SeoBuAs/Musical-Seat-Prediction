import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_results(results_dir='results'):
    combined_file = os.path.join(results_dir, 'combined_all_versions.csv')
    
    if os.path.exists(combined_file):
        print(f"Loading combined results file: {combined_file}")
        return pd.read_csv(combined_file)
    else:
        print("Combined results file not found. Loading individual files...")
        
        versions = ['normal', 'smote', 'random', 'borderline_smote']
        all_results = []
        
        for version in versions:
            file_path = os.path.join(results_dir, version, f'performance_metrics_{version}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_results.append(df)
                print(f"  Loaded: {version}")
            else:
                print(f"  Not found: {version}")
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            print("No result files found!")
            return None


def create_comparison_plots(df, output_dir='results/comparison_plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Creating Test R² comparison plot by version...")
    test_data = df[df['Split'] == 'Test'].copy()
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=test_data, x='Model', y='R2', hue='Version')
    plt.xticks(rotation=45, ha='right')
    plt.title('Test R² by Model and Sampling Version', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xlabel('Model')
    plt.legend(title='Sampling Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_r2_comparison.png'), dpi=300)
    plt.close()
    
    print("2. Creating average performance comparison plot by version...")
    avg_performance = test_data.groupby('Version')[['R2', 'MSE', 'MAE']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['R2', 'MSE', 'MAE']
    titles = ['Average R² (Higher is Better)', 'Average MSE (Lower is Better)', 'Average MAE (Lower is Better)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        sns.barplot(data=avg_performance, x='Version', y=metric, ax=ax, palette='viridis')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Sampling Version')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_performance_comparison.png'), dpi=300)
    plt.close()
    
    print("3. Creating performance heatmap by model and version...")
    pivot_data = test_data.pivot_table(index='Model', columns='Version', values='R2')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', center=pivot_data.mean().mean())
    plt.title('Test R² Heatmap: Model vs Sampling Version', fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Version')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_heatmap.png'), dpi=300)
    plt.close()
    
    print("4. Creating Validation vs Test performance comparison plot...")
    val_data = df[df['Split'] == 'Validation_K-FOLD'][['Version', 'Model', 'R2']].copy()
    val_data.rename(columns={'R2': 'Validation_R2'}, inplace=True)
    
    test_data_merge = test_data[['Version', 'Model', 'R2']].copy()
    test_data_merge.rename(columns={'R2': 'Test_R2'}, inplace=True)
    
    comparison = pd.merge(val_data, test_data_merge, on=['Version', 'Model'])
    comparison['Overfit_Gap'] = comparison['Validation_R2'] - comparison['Test_R2']
    
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=comparison, x='Validation_R2', y='Test_R2', 
                    hue='Version', style='Model', s=100, alpha=0.7)
    
    max_val = max(comparison['Validation_R2'].max(), comparison['Test_R2'].max())
    min_val = min(comparison['Validation_R2'].min(), comparison['Test_R2'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Match')
    
    plt.title('Validation R² vs Test R² (Overfitting Check)', fontsize=14, fontweight='bold')
    plt.xlabel('Validation R²')
    plt.ylabel('Test R²')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_vs_test.png'), dpi=300)
    plt.close()
    
    print("5. Creating overfitting comparison plot...")
    plt.figure(figsize=(14, 8))
    sns.barplot(data=comparison, x='Model', y='Overfit_Gap', hue='Version')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.title('Overfitting Gap (Validation R² - Test R²)', fontsize=14, fontweight='bold')
    plt.ylabel('Overfit Gap (Higher = More Overfitting)')
    plt.xlabel('Model')
    plt.legend(title='Sampling Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_gap.png'), dpi=300)
    plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")


def create_summary_table(df, output_dir='results'):
    print("\nCreating summary table...")
    
    test_data = df[df['Split'] == 'Test'].copy()
    
    version_summary = test_data.groupby('Version').agg({
        'R2': ['mean', 'std', 'min', 'max'],
        'MSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    version_summary.to_csv(os.path.join(output_dir, 'version_summary.csv'))
    print(f"  Saved: {os.path.join(output_dir, 'version_summary.csv')}")
    
    best_version_per_model = test_data.loc[test_data.groupby('Model')['R2'].idxmax()]
    best_version_per_model = best_version_per_model[['Model', 'Version', 'R2', 'MSE', 'MAE']]
    best_version_per_model.to_csv(os.path.join(output_dir, 'best_version_per_model.csv'), index=False)
    print(f"  Saved: {os.path.join(output_dir, 'best_version_per_model.csv')}")
    
    best_overall = test_data.loc[test_data['R2'].idxmax()]
    print("\n[Best Overall Performance]")
    print(f"  Version: {best_overall['Version']}")
    print(f"  Model: {best_overall['Model']}")
    print(f"  R²: {best_overall['R2']:.4f}")
    print(f"  MSE: {best_overall['MSE']:.4f}")
    print(f"  MAE: {best_overall['MAE']:.4f}")
    
    print("\n[Average Test R² by Version]")
    version_avg = test_data.groupby('Version')['R2'].mean().sort_values(ascending=False)
    for version, r2 in version_avg.items():
        print(f"  {version:20s}: {r2:.4f}")
    
    return version_summary, best_version_per_model


def main():
    print("="*70)
    print("Version Comparison Analysis")
    print("="*70)
    
    df = load_results()
    
    if df is None:
        print("\nCannot load results. Please run experiments first.")
        return
    
    print(f"\nLoaded data: {len(df)} rows")
    print(f"Versions: {df['Version'].unique()}")
    print(f"Models: {df['Model'].unique()}")
    print(f"Splits: {df['Split'].unique()}")
    
    version_summary, best_version_per_model = create_summary_table(df)
    
    create_comparison_plots(df)
    
    print("\n" + "="*70)
    print("Analysis completed!")
    print("="*70)


if __name__ == '__main__':
    main()
