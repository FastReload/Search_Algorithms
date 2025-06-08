import numpy as np
import csv

def load_data(filename):
    data = np.loadtxt(filename)
    y = data[:, 0]
    X = data[:, 1:]
    return X, y

def normalize_features(X):
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    valid = stds != 0
    X = X[:, valid]
    means = means[valid]
    stds = stds[valid]
    return (X - means) / stds


def leave_one_out_accuracy(X, y, feature_indices):
    if not feature_indices:
        return 0.0
    correct = 0
    for i in range(len(X)):
        test_sample = X[i, feature_indices]
        test_label = y[i]
        train_X = np.delete(X, i, axis=0)[:, feature_indices]
        train_y = np.delete(y, i)
        dists = np.linalg.norm(train_X - test_sample, axis=1)
        nearest = np.argmin(dists)
        if train_y[nearest] == test_label:
            correct += 1
    return correct / len(X) * 100

def forward_selection(X, y):
    all_features_acc = leave_one_out_accuracy(X, y, list(range(X.shape[1])))
    print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evaluation, I get an accuracy of {:.1f}%".format(X.shape[1], all_features_acc))
    print("\nBeginning search.\n")

    n = X.shape[1]
    selected = []
    best_overall = []
    best_overall_acc = 0.0

    with open("forward_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Num_Features", "Feature_Indices", "Accuracy"])

        for level in range(n):
            best_feature = None
            best_acc = -1.0

            for i in range(n):
                if i in selected:
                    continue
                candidate = selected + [i]
                acc = leave_one_out_accuracy(X, y, candidate)
                feature_indices = [f + 1 for f in candidate]
                writer.writerow([level + 1, len(candidate), feature_indices, round(acc, 1)])

                print("\tUsing feature(s) {{{}}} accuracy is {:.1f}%".format(",".join(map(str, feature_indices)), acc))
                if acc > best_acc:
                    best_acc = acc
                    best_feature = i

            print()
            if best_feature is not None:
                selected.append(best_feature)
                if best_acc > best_overall_acc:
                    best_overall = selected[:]
                    best_overall_acc = best_acc
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                else:
                    print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
                    print("Feature set {{{}}} was best, accuracy is {:.1f}%".format(",".join(str(f + 1) for f in selected), best_acc))
                print()

    print("Finished search!! The best feature subset is {{{}}}, which has an accuracy of {:.1f}%".format(
        ",".join(str(f + 1) for f in best_overall), best_overall_acc
    ))