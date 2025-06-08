import pandas as pd

# Load the data
df = pd.read_csv("data.csv")

# Separate label and features
if "Bankrupt?" in df.columns:
    y = df["Bankrupt?"]
    X = df.drop(columns=["Bankrupt?"])
else:
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

# Randomly select 40 features
selected_features = X.sample(n=40, axis=1, random_state=42)

# Combine back with label
selected_df = pd.concat([y, selected_features], axis=1)

# Randomly select 1200 instances
selected_df = selected_df.sample(n=1200, random_state=42)

# Save in space-separated format, no headers, no index
selected_df.to_csv("bankruptcy_cleaned.txt", sep=" ", index=False, header=False, float_format="%.8f")
