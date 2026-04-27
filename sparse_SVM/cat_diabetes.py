from sparse_SVM_custom import Sparse_SVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.svm import SVC

create_test_data = True
load_data = True
run_standard_svm = True
greedy_backward = True
data_setname = 'cats_day_60_90'
time_window = '(Day 60-90)'

build_cumulative_model = True
remove_glucose = True

C = 1
base_path = "/Users/nathancarlson/Desktop/programs/MATH 532/sparse_SVM/cat_data/day_60_90"


if create_test_data:
    # load data

    with open(f"{base_path}/cat_data_day_60_90.pkl", "rb") as f:
        X = pickle.load(f)
    print(X)
    X_clean = [block[1] for block in X]
    samples = np.vstack(X_clean)
    print('x shape')
    print(samples.shape)
    y = np.concatenate([

        np.full(X[i][1].shape[0], X[i][0])

        for i in range(len(X))

    ])
    print(y.shape)
    with open(f"{base_path}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    feature_names = np.array(feature_names)

    # making all 0 labels 1
    print("raw labels:", np.unique(y))
    y = np.where(y == 0, -1, 1)
    y = y.reshape(-1, 1)
    print("converted labels:", np.unique(y))
    unique, counts = np.unique(y, return_counts=True)

    print("label counts:")

    for u, c in zip(unique, counts):
        print(f"{u}: {c}")

    data = np.concatenate((samples, y), axis=1)



    from sklearn.model_selection import train_test_split
    import numpy as np

    # -----------------------------
    # Data
    # -----------------------------
    X = samples

    # -----------------------------
    # Train / Temp split (80 / 20)
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # -----------------------------
    # Split temp into val / test (50 / 50)
    # -> 10% val, 10% test total
    # -----------------------------
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # Save everything
    # -----------------------------
    np.savez(
        data_setname,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names
    )

    print(f"Saved dataset to {data_setname}.npz")

if load_data:
    data = np.load(f"{data_setname}.npz")

    X_train = data["X_train"]
    y_train = data["y_train"].ravel()

    X_val = data["X_val"]
    y_val = data["y_val"].ravel()

    X_test = data["X_test"]
    y_test = data["y_test"].ravel()

    feature_names = data["feature_names"]

# function takes in data (samples as rows) and labels as the last column
train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)

weights, bias = Sparse_SVM(train_data, C=C, plot_data_option=False)
print('Sparse SVM Trained')
# -----------------------------
# Prediction function
# -----------------------------
def predict(X, w, b):
    scores = X @ w + b
    preds = np.sign(scores)
    preds[preds == 0] = 1  # or -1, but be consistent
    return preds

# -----------------------------
# Compute accuracies
# -----------------------------
y_train_pred = predict(X_train, weights, bias)
y_val_pred   = predict(X_val, weights, bias)

train_acc = np.mean(y_train_pred == y_train)
val_acc   = np.mean(y_val_pred == y_val)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Pair weights with features

weights = np.array(weights).flatten()
abs_weights = np.abs(weights)

## order features from greatest weight to least
# sort indices (largest → smallest)
## THESE ONLY COME FROM FIRST MODEL
idx = np.argsort(abs_weights)[::-1]
ordered_features = feature_names[idx]
ordered_abs_weights = abs_weights[idx]

k = 10

ordered_weights = weights[idx]
print(r'Printing Top k features')

for f, w in zip(ordered_features[:k], ordered_weights[:k]):
    print(f, w)

## Plotting magintude of Features
plt.figure()
plt.xlabel(r"$k$ most important feature (greatest to least)")
plt.ylabel("Absolute Value of Feature Weights (greatest to least)")
plt.plot(np.arange(1,len(ordered_abs_weights)+1, 1),ordered_abs_weights +1e-12)
plt.title("Absolute Value of Feature Weights")
plt.yscale('log')
plt.show()

if greedy_backward:
    results = []

    X_train_curr = X_train.copy()
    X_val_curr = X_val.copy()
    features_curr = feature_names.copy()

    cumulative_removed = []

    step = 0

    while X_train_curr.shape[1] > 1:

        train_data_curr = np.concatenate(
            (X_train_curr, y_train.reshape(-1, 1)), axis=1
        )

        w, b = Sparse_SVM(train_data_curr, C=C, plot_data_option=False)
        w = np.array(w).flatten()

        # validation accuracy
        y_val_pred = predict(X_val_curr, w, b)
        val_acc_curr = np.mean(y_val_pred == y_val)

        # find feature to remove
        idx_remove = np.argmax(np.abs(w))
        feature_removed = features_curr[idx_remove]

        # update cumulative list
        cumulative_removed.append(feature_removed)

        # store snapshot
        results.append({
            "step": step,
            "num_features": X_train_curr.shape[1],
            "val_acc": val_acc_curr,
            "removed_feature": feature_removed,
            "cumulative_removed": cumulative_removed.copy()
        })

        print(f"Step {step} | Acc: {val_acc_curr:.4f} | Removed: {feature_removed}")

        # remove feature
        X_train_curr = np.delete(X_train_curr, idx_remove, axis=1)
        X_val_curr = np.delete(X_val_curr, idx_remove, axis=1)
        features_curr = np.delete(features_curr, idx_remove)

        step += 1

        num_features = [r["num_features"] for r in results]
        val_accs = [r["val_acc"] for r in results]

        plt.figure()
        plt.plot(num_features, val_accs, marker='o')
        plt.gca().invert_xaxis()
        plt.xlabel("Number of Features")
        plt.ylabel("Validation Accuracy")
        plt.title("SVM Backward Feature Elimination")
        plt.grid()
        plt.show()

        print("\nStep | #Features | Val Acc | Removed Feature")
        for r in results:
            print(f"{r['step']:>4} | {r['num_features']:>9} | {r['val_acc']:.4f} | {r['removed_feature']}")

        with open(f"{data_setname}_feature_elimination_log.pkl", "wb") as f:
            pickle.dump(results, f)
        print("Saved full elimination log (with cumulative features).")


if run_standard_svm:
    svm = SVC(
        C=C,
        kernel='linear'   # matches your linear Sparse_SVM
    )

    svm.fit(X_train, y_train)

    y_train_pred = svm.predict(X_train)
    y_val_pred   = svm.predict(X_val)
    y_test_pred  = svm.predict(X_test)

    train_acc = np.mean(y_train_pred == y_train)
    val_acc   = np.mean(y_val_pred == y_val)
    test_acc  = np.mean(y_test_pred == y_test)

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc:   {val_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")


if build_cumulative_model:

    with open(f"{data_setname}_feature_elimination_log.pkl", "rb") as f:
        results = pickle.load(f)

    # IMPORTANT: build ranking (best → worst)
    ranked_features = results[-1]["cumulative_removed"][::-1]

    if remove_glucose:
        # REMOVE glucose BEFORE anything else
        ranked_features = [f for f in ranked_features if f != "SC_GLUCOSE"]

        print("Ranked features (best → worst, glucose removed):")
        print(ranked_features)
        time_window += ' (Glucose Dropped)'

    ranked_features = ranked_features[::-1]

    print("Ranked features (best → worst):")

    print(ranked_features)

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    selected_features = []
    results_forward = []

    # -----------------------------
    # Forward construction
    # -----------------------------
    for k, feat in enumerate(ranked_features):
        selected_features.append(feat)

        idx = [feature_to_idx[f] for f in selected_features]

        X_train_k = X_train[:, idx]
        X_val_k = X_val[:, idx]

        train_data_k = np.concatenate((X_train_k, y_train.reshape(-1, 1)), axis=1)

        w, b = Sparse_SVM(train_data_k, C=C, plot_data_option=False)
        w = np.array(w).flatten()

        y_train_pred = predict(X_train_k, w, b)
        y_val_pred = predict(X_val_k, w, b)

        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)

        results_forward.append({
            "k": k + 1,
            "added_feature": feat,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        print(f"{k + 1:2d} | add {feat:20s} | val={val_acc:.4f}")

    # -----------------------------
    # PLOT ONCE (IMPORTANT FIX)
    # -----------------------------
    ks = [r["k"] for r in results_forward]
    train_accs = [r["train_acc"] for r in results_forward]
    val_accs = [r["val_acc"] for r in results_forward]

    plt.figure()
    #plt.plot(ks, train_accs, label="train")
    plt.plot(ks, val_accs, label="Validation Accuracy")
    plt.xlabel("Number of Features Added")
    plt.ylabel("Accuracy")
    plt.title(f"{time_window} (Greedy Additive SVM)")
    plt.legend()
    plt.grid()
    plt.show()