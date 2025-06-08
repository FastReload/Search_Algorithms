
#  Search Algorithms in AI

Welcome to the **Search Algorithms** repository!  
This project implements core feature selection techniques using a Nearest Neighbor classifier with **Leave-One-Out Cross Validation**. It includes:

- Forward Selection
- Backward Elimination

These are essential wrapper-based feature selection methods used in machine learning and artificial intelligence for dimensionality reduction.

---

##  Folder Structure

```
.
├── main.py             # Main script to run feature selection
├── Plots.py            # Script to generate graphs
├── data_clean.py       # Script to preprocess the Taiwanese Bankruptcy Dataset
├── Logs                # Folder containing all logs for analysis
├── Datasets            # Folder containing all datasets used here
├── Plots               # Folder containing all plots generated    
```

---

##  Getting Started

[Download the Taiwanese Bankruptcy Prediction dataset from UCI](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) – 6,819 samples, 96 financial features. 
In addition to the above, this project also tests two other datasets: `CS205_small_Data__21.txt` and `CS205_large_Data__36.txt`, included in the datasets folder of this repository.



### Prerequisites

- Python 3.x
- NumPy

You can install dependencies via:
```bash
pip install numpy
pip install pandas
pip install csv
pip install matplotlib
```

## Running the Script

1. **Preprocess the dataset**  
   Clean and normalize your dataset using:
   ```bash
   python data_clean.py
   ```

2. **Run the Feature Selection Algorithm**  
   Once the data is prepared, run the main script:
   ```bash
   python main.py
   ```

3. **Generate Visualizations**  
   After running the algorithm, use:
   ```bash
   python plots.py
   ```
   to generate performance graphs and analyze feature selection results.

---

You will be prompted to:
```bash
1. Enter the dataset filename:
2. Choose the search algorithm:
   - `1` → Forward Selection
   - `2` → Backward Elimination
```
The script will output the best feature subset and save logs to CSV.

---

##  Output

Each run generates a log file:
- `forward_log.csv` or `backward_log.csv`
- Columns: `Iteration`, `Num_Features`, `Feature_Indices`, `Accuracy`

This helps visualize the performance of different feature subsets.

---

##  Algorithms Explained

### Forward Selection
Starts with an empty set and adds the feature that improves accuracy the most, one by one.

### Backward Elimination
Starts with all features and removes the one whose removal causes the least drop in accuracy, one by one.

---

##  Example Output

```
Using feature(s) {1} accuracy is 75.0%
Using feature(s) {1, 4} accuracy is 85.0%
Feature set {1, 4, 12} was best, accuracy is 92.5%
```

---

##  Features

- Assumes all features are continuous and binary classification.
- Uses 1-Nearest Neighbor and Leave-One-Out CV for evaluation.
- Ignores features with zero variance.

---

##  License

MIT License

---

## 👨‍💻 Author

Developed by **Aryan Ramachandra**  
If you find this helpful, consider ⭐️ starring the repo!
