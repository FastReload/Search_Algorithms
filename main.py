import numpy as np
import csv
import time

# Function to load data from a file
# Assumes the first column is the label (y) and the rest are features (X)
def load_data(filename):
    data = np.loadtxt(filename)  # Load the text file assuming no headers
    y = data[:, 0]               # First column = class labels
    X = data[:, 1:]              # Remaining columns = features
    return X, y

# Normalize features to mean 0 and standard deviation 1
# Removes any feature that has 0 standard deviation (i.e., constant)
def normalize_features(X):
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    valid = stds != 0            # Only keep features with non-zero variance
    X = X[:, valid]
    means = means[valid]
    stds = stds[valid]
    return (X - means) / stds    # Standard score normalization

# Perform Leave-One-Out Cross Validation using 1-Nearest Neighbor
# Takes only the specified feature indices
def leave_one_out_accuracy(X, y, feature_indices):
    if not feature_indices:
        return 0.0
    correct = 0
    for i in range(len(X)):
        test_sample = X[i, feature_indices]                    # Current test sample
        test_label = y[i]
        train_X = np.delete(X, i, axis=0)[:, feature_indices]  # Leave out sample i from training
        train_y = np.delete(y, i)
        dists = np.linalg.norm(train_X - test_sample, axis=1)  # Euclidean distances to all others
        nearest = np.argmin(dists)                             # Index of nearest neighbor
        if train_y[nearest] == test_label:
            correct += 1                                       # Count correct predictions
    return correct / len(X) * 100                              # Accuracy as percentage

# Forward Selection Algorithm for feature subset selection
def forward_selection(X, y):
    all_features_acc = leave_one_out_accuracy(X, y, list(range(X.shape[1])))
    print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evaluation, I get an accuracy of {:.1f}%".format(X.shape[1], all_features_acc))
    print("\nBeginning search.\n")

    n = X.shape[1]
    selected = []                 # Current selected features
    best_overall = []            # Best feature set found so far
    best_overall_acc = 0.0       # Best accuracy found so far

    with open("forward_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Num_Features", "Feature_Indices", "Accuracy"])

        for level in range(n):
            best_feature = None
            best_acc = -1.0

            # Try adding each unused feature and check performance
            for i in range(n):
                if i in selected:
                    continue
                candidate = selected + [i]                       # Add candidate feature
                acc = leave_one_out_accuracy(X, y, candidate)
                feature_indices = [f + 1 for f in candidate]     # 1-based index for printing
                writer.writerow([level + 1, len(candidate), feature_indices, round(acc, 1)])

                print("\tUsing feature(s) {{{}}} accuracy is {:.1f}%".format(",".join(map(str, feature_indices)), acc))
                if acc > best_acc:
                    best_acc = acc
                    best_feature = i

            print()
            if best_feature is not None:
                selected.append(best_feature)                    # Commit best feature of this round
                if best_acc > best_overall_acc:
                    best_overall = selected[:]                   # Save if best so far
                    best_overall_acc = best_acc
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                else:
                    print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                print()

    # Final best result
    print("Finished search!! The best feature subset is {{{}}}, which has an accuracy of {:.1f}%".format(
        ",".join(str(f + 1) for f in best_overall), best_overall_acc
    ))

# Backward Elimination Algorithm for feature subset selection
def backward_elimination(X, y):
    initial_features = list(range(X.shape[1]))
    initial_acc = leave_one_out_accuracy(X, y, initial_features)

    print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evaluation, I get an accuracy of {:.1f}%".format(X.shape[1], initial_acc))
    print("\nBeginning search.\n")

    selected = initial_features[:]     # Start with all features
    best_overall = selected[:]
    best_overall_acc = initial_acc

    with open("backward_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Num_Features", "Feature_Indices", "Accuracy"])

        for level in range(len(selected) - 1):
            worst_feature = None
            best_acc = -1.0

            # Try removing each feature one by one
            for i in selected:
                temp = [f for f in selected if f != i]          # Exclude one feature
                acc = leave_one_out_accuracy(X, y, temp)
                feature_indices = [f + 1 for f in temp]
                writer.writerow([level + 1, len(temp), feature_indices, round(acc, 1)])

                print("\tUsing feature(s) {{{}}} accuracy is {:.1f}%".format(",".join(map(str, feature_indices)), acc))
                if acc > best_acc:
                    best_acc = acc
                    worst_feature = i

            print()
            if worst_feature is not None:
                selected = [f for f in selected if f != worst_feature]  # Remove worst
                if best_acc > best_overall_acc:
                    best_overall = selected[:]
                    best_overall_acc = best_acc
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                else:
                    print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                print()

    # Final best result
    print("Finished search!! The best feature subset is {{{}}}, which has an accuracy of {:.1f}%".format(
        ",".join(str(f + 1) for f in best_overall), best_overall_acc
    ))

# Main entry point
if __name__ == "__main__":
    print("Welcome to Aryan Ramachandra's Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test: ").strip()

    # Load and preprocess dataset
    try:
        X, y = load_data(file_name)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()
    X = normalize_features(X)

    # Ask user which algorithm to run
    print()
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print()
    choice = input().strip()

    # Execute selected algorithm and time the run
    if choice == '1':
        start = time.time()
        forward_selection(X, y)
        end = time.time()
        print(f"\nTotal time taken: {end - start:.2f} seconds")
    elif choice == '2':
        start = time.time()
        backward_elimination(X, y)
        end = time.time()
        print(f"\nTotal time taken: {end - start:.2f} seconds")
    else:
        print("Invalid choice.")
