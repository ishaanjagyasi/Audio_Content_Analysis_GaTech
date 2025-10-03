import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Feature_extraction import AudioFeatureExtractor


def train_and_evaluate_knn(extractor, train_path, test_path, k=5):

    print("\n=== EXTRACTING TRAINING FEATURES ===")
    train_features, train_labels, train_files = extractor.extract_dataset_features(
        train_path
    )
    print(f"Training samples: {len(train_labels)}")
    print(f"Training label distribution:\n{pd.Series(train_labels).value_counts()}")

    print("\n=== EXTRACTING TEST FEATURES ===")
    test_features, test_labels, test_files = extractor.extract_dataset_features(
        test_path
    )
    print(f"Test samples: {len(test_labels)}")
    print(f"Test label distribution:\n{pd.Series(test_labels).value_counts()}")

    print(f"\n=== TRAINING kNN (k={k}) ===")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)
    print("Training complete!")

    print("\n=== PREDICTING ON TEST SET ===")
    predictions = knn.predict(test_features)

    #  Accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n{'='*50}")
    print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # Confusion matrix
    labels = sorted(list(set(train_labels)))
    conf_matrix = confusion_matrix(test_labels, predictions, labels=labels)

    print("\n=== CONFUSION MATRIX ===")
    df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(df_cm)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(test_labels, predictions, target_names=labels))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix\nk={k}, Accuracy={accuracy:.2%}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return accuracy, conf_matrix, predictions, test_labels


def main():

    extractor = AudioFeatureExtractor(
        target_sr=22050, frame_size=2048, hop_ratio=0.5, ignore_last_seconds=1.0
    )

    train_path = "./micro_medlydb/validate/"
    test_path = "./micro_medlydb/test/"

    accuracy, conf_matrix, predictions, test_labels = train_and_evaluate_knn(
        extractor, train_path, test_path, k=5
    )

    print(f"Final Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
