import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Feature_extraction import AudioFeatureExtractor


def analyze_pca_variance(pca, feature_names):

    ######## Analyze and visualize the explained variance from PCA ########

    explained_variance_ratio = pca.explained_variance_ratio_ # this stores the percentange of variance explained by each principal component
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n" + "="*80)
    print("PCA VARIANCE ANALYSIS")
    print("="*80)
    
    # Print detailed variance breakdown
    print("\n--- Individual Component Variance ---")
    for i in range(len(explained_variance_ratio)):
        print(f"PC{i+1:2d}: {explained_variance_ratio[i]*100:6.2f}% | "
              f"Cumulative: {cumulative_variance[i]*100:6.2f}%")
    

    n_for_90 = np.argmax(cumulative_variance >= 0.90) + 1 # this finds the number of components that explains 90% of the variance
    n_for_95 = np.argmax(cumulative_variance >= 0.95) + 1 # this finds the number of components that explains 95% of the variance
    
    print(f"\n--- Variance Thresholds ---")
    print(f"Components for 90% variance: {n_for_90}")
    print(f"Components for 95% variance: {n_for_95}")
    
    # Create scree plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)) # GENERATED FROM CLAUDE FOR REPERESNTATION OF VARIANCE PLOTS OF PCAs
    
    # Plot 1: Scree plot (individual variance)
    axes[0].plot(range(1, len(explained_variance_ratio) + 1), 
                 explained_variance_ratio * 100, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
    axes[0].set_title('Scree Plot - Individual Variance', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(1, len(explained_variance_ratio) + 1))
    
    # Highlight the "elbow" region to suggest how many components to use (basically where does the variance starts to decrease significantly)
    axes[0].axvspan(6, 10, alpha=0.2, color='green', label='Suggested range')
    axes[0].legend()
    
    # Plot 2: Cumulative variance
    axes[1].plot(range(1, len(cumulative_variance) + 1), 
                 cumulative_variance * 100, 'ro-', linewidth=2, markersize=8)
    axes[1].axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% threshold')
    axes[1].axhline(y=95, color='blue', linestyle='--', linewidth=2, label='95% threshold')
    axes[1].axvline(x=n_for_90, color='green', linestyle=':', alpha=0.5)
    axes[1].axvline(x=n_for_95, color='blue', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(1, len(cumulative_variance) + 1))
    
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print component weights for first 3 PCs
    print("\n--- Top 3 Principal Component Weights ---")
    components = pca.components_[:3]  # First 3 PCs
    
    for i, component in enumerate(components):
        print(f"\n** PC{i+1} (explains {explained_variance_ratio[i]*100:.2f}%) **")
        # Get top 5 contributing features
        abs_loadings = np.abs(component)
        top_indices = np.argsort(abs_loadings)[::-1][:5]
        
        for idx in top_indices:
            print(f"  {feature_names[idx]:20s}: {component[idx]:7.4f}")
    
    return n_for_90, n_for_95


def visualize_pca_projection(pca_scores, labels, n_components_used): # in order to see how much data is being segragated in the PCA space ---- if there are any clear clusters, then we can use that to our advantage

    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    if n_components_used >= 2:
        fig = plt.figure(figsize=(15, 5))
        
        # PC1 vs PC2
        ax1 = fig.add_subplot(131)
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax1.scatter(pca_scores[mask, 0], pca_scores[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7, s=100, edgecolors='k')
        ax1.set_xlabel('PC1', fontsize=12)
        ax1.set_ylabel('PC2', fontsize=12)
        ax1.set_title('2D PCA Projection (PC1 vs PC2)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if n_components_used >= 3:
            # PC1 vs PC3
            ax2 = fig.add_subplot(132)
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax2.scatter(pca_scores[mask, 0], pca_scores[mask, 2], 
                           c=[colors[i]], label=label, alpha=0.7, s=100, edgecolors='k')
            ax2.set_xlabel('PC1', fontsize=12)
            ax2.set_ylabel('PC3', fontsize=12)
            ax2.set_title('2D PCA Projection (PC1 vs PC3)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # PC2 vs PC3
            ax3 = fig.add_subplot(133)
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax3.scatter(pca_scores[mask, 1], pca_scores[mask, 2], 
                           c=[colors[i]], label=label, alpha=0.7, s=100, edgecolors='k')
            ax3.set_xlabel('PC2', fontsize=12)
            ax3.set_ylabel('PC3', fontsize=12)
            ax3.set_title('2D PCA Projection (PC2 vs PC3)', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_projections.png', dpi=300, bbox_inches='tight')
        plt.show()


def train_evaluate_pca_knn(train_features, train_labels, test_features, test_labels, 
                           n_components, k=5):

    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features)
    
    # Transform test data using the same PCA
    test_pca = pca.transform(test_features)
    
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_pca, train_labels)
    
    # Predict
    predictions = knn.predict(test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy, predictions, pca, train_pca, test_pca


def compare_n_components(train_features, train_labels, test_features, test_labels, 
                         component_range, k=5):
    """
    Test different numbers of PCA (4, 6, 8, 10, 12, 14) components and compare accuracies
    """
    print("\n" + "="*80)
    print("TESTING DIFFERENT NUMBERS OF COMPONENTS")
    print("="*80)
    
    results = []
    
    for n_comp in component_range:
        accuracy, _, _, _, _ = train_evaluate_pca_knn(
            train_features, train_labels, test_features, test_labels, 
            n_components=n_comp, k=k
        )
        results.append({'n_components': n_comp, 'accuracy': accuracy})
        print(f"n_components={n_comp:2d}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Plot accuracy vs n_components
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_components'], results_df['accuracy'] * 100, 
             'bo-', linewidth=2, markersize=10)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('KNN Accuracy vs Number of PCA Components (k=5)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(component_range)
    
    # Highlight best
    best_idx = results_df['accuracy'].idxmax()
    best_n = results_df.loc[best_idx, 'n_components']
    best_acc = results_df.loc[best_idx, 'accuracy']
    plt.axvline(x=best_n, color='red', linestyle='--', alpha=0.5, 
                label=f'Best: {best_n} components ({best_acc*100:.2f}%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


def plot_confusion_matrix_comparison(test_labels, predictions_baseline, predictions_pca, 
                                    labels, acc_baseline, acc_pca, n_components):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline confusion matrix
    conf_baseline = confusion_matrix(test_labels, predictions_baseline, labels=labels)
    sns.heatmap(conf_baseline, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=axes[0], 
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Baseline (16 features)\nAccuracy: {acc_baseline:.2%}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # PCA confusion matrix
    conf_pca = confusion_matrix(test_labels, predictions_pca, labels=labels)
    sns.heatmap(conf_pca, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels, ax=axes[1], 
                cbar_kws={'label': 'Count'})
    axes[1].set_title(f'PCA ({n_components} components)\nAccuracy: {acc_pca:.2%}', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("="*80)
    print("PCA + KNN AUDIO CLASSIFICATION PIPELINE")
    print("="*80)
    
    extractor = AudioFeatureExtractor(
        target_sr=22050, frame_size=2048, hop_ratio=0.5, ignore_last_seconds=1.0
    )
    
    train_path = "./micro_medlydb/validate/"
    test_path = "./micro_medlydb/test/"
    

    print("\n=== EXTRACTING TRAINING FEATURES ===")
    train_features, train_labels, train_files = extractor.extract_dataset_features(train_path)
    print(f"Training samples: {len(train_labels)}")
    print(f"Feature dimensions: {train_features.shape}")
    
    print("\n=== EXTRACTING TEST FEATURES ===")
    test_features, test_labels, test_files = extractor.extract_dataset_features(test_path)
    print(f"Test samples: {len(test_labels)}")
    print(f"Feature dimensions: {test_features.shape}")
    

    feature_names = [
        "spectral_centroid", "spectral_spread", "spectral_rolloff",
        "spectral_flatness", "zero_crossing_rate", "spectral_flux"
    ]
    for i in range(10):
        feature_names.append(f"mfcc_{i}")
    
    # ==================== STEP 1: FULL PCA ANALYSIS ====================
    print("\n" + "="*80)
    print("STEP 1: PERFORMING FULL PCA ANALYSIS (ALL 16 COMPONENTS)")
    print("="*80)
    
    # Fit PCA with all components to analyze variance
    pca_full = PCA(n_components=None)
    pca_full.fit(train_features)
    
    # Analyze variance and create scree plots
    n_for_90, n_for_95 = analyze_pca_variance(pca_full, feature_names)
    
    # ==================== STEP 2: BASELINE (NO PCA) ====================
    print("\n" + "="*80)
    print("STEP 2: BASELINE KNN (NO PCA - ALL 16 FEATURES)")
    print("="*80)
    
    knn_baseline = KNeighborsClassifier(n_neighbors=5)
    knn_baseline.fit(train_features, train_labels)
    predictions_baseline = knn_baseline.predict(test_features)
    accuracy_baseline = accuracy_score(test_labels, predictions_baseline)
    
    print(f"\nBaseline Accuracy: {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
    
    # ==================== STEP 3: TEST DIFFERENT N_COMPONENTS ====================
    print("\n" + "="*80)
    print("STEP 3: TESTING DIFFERENT NUMBERS OF PRINCIPAL COMPONENTS")
    print("="*80)
    
    # Test range based on scree plot suggestions
    component_range = [4, 6, 8, 10, 12, 14]
    results_df = compare_n_components(
        train_features, train_labels, test_features, test_labels,
        component_range, k=5
    )

    # ==================== STEP 4: BEST PCA MODEL ====================

    # Find all components with maximum accuracy
    max_accuracy = results_df['accuracy'].max()
    tied_components = results_df[results_df['accuracy'] == max_accuracy]['n_components'].tolist()
    
    # If multiple components tie, choose based on variance threshold
    if len(tied_components) > 1:
        print(f"\n Multiple components tied at {max_accuracy:.2%} accuracy: {tied_components}")
        
        # Check which tied components meet 95% variance threshold
        components_above_95 = [n for n in tied_components if n >= n_for_95]
        
        if components_above_95:
            # Choose the smallest that meets 95% threshold
            best_n = min(components_above_95)
            print(f"✓ Choosing {best_n} PCs (smallest that captures ≥95% variance)")
        else:
            # If none meet 95%, choose largest tied component (most info)
            best_n = max(tied_components)
            print(f"✓ Choosing {best_n} PCs (largest tied component, captures most variance)")
    else:
        # No tie, just use the one with best accuracy
        best_n = tied_components[0]
    
    best_accuracy = max_accuracy
    
    print("\n" + "="*80)
    print(f"STEP 4: TRAINING FINAL MODEL WITH BEST n_components={best_n}")
    print("="*80)
    
    accuracy_best, predictions_pca, pca_best, train_pca, test_pca = train_evaluate_pca_knn(
        train_features, train_labels, test_features, test_labels,
        n_components=best_n, k=5
    )
    
    print(f"\nBest PCA Model:")
    print(f"  Number of components: {best_n}")
    print(f"  Accuracy: {accuracy_best:.4f} ({accuracy_best*100:.2f}%)")
    print(f"  Variance explained: {np.sum(pca_best.explained_variance_ratio_)*100:.2f}%")
    
    # Visualize PCA projection
    visualize_pca_projection(train_pca, train_labels, best_n)
    
    # ==================== STEP 5: COMPARISON ====================
    print("\n" + "="*80)
    print("STEP 5: FINAL COMPARISON")
    print("="*80)
    
    print(f"\nBaseline (16 features): {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
    print(f"PCA ({best_n} components):  {accuracy_best:.4f} ({accuracy_best*100:.2f}%)")
    
    improvement = accuracy_best - accuracy_baseline
    print(f"\nImprovement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        print("✓ PCA improved classification accuracy!")
    elif improvement < 0:
        print("✗ PCA reduced classification accuracy")
    else:
        print("= PCA had no effect on accuracy")
    
    # Plot comparison confusion matrices
    labels = sorted(list(set(train_labels)))
    plot_confusion_matrix_comparison(
        test_labels, predictions_baseline, predictions_pca,
        labels, accuracy_baseline, accuracy_best, best_n
    )
    
    # Classification reports
    print("\n--- BASELINE Classification Report ---")
    print(classification_report(test_labels, predictions_baseline, target_names=labels))
    
    print("\n--- PCA Classification Report ---")
    print(classification_report(test_labels, predictions_pca, target_names=labels))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated plots:")
    print("  1. pca_variance_analysis.png - Scree plot and cumulative variance")
    print("  2. accuracy_vs_components.png - Accuracy for different n_components")
    print("  3. pca_projections.png - Data visualization in PC space")
    print("  4. confusion_matrix_comparison.png - Baseline vs PCA comparison")


if __name__ == "__main__":
    main()