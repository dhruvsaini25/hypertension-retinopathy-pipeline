import pandas as pd
from sklearn.model_selection import train_test_split

# path to your CSV
csv_path = "dataset/1-Hypertensive Classification/1-Hypertensive Classification/2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv"

df = pd.read_csv(csv_path)

# first split train / temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.iloc[:,1]
)

# split temp into val + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df.iloc[:,1]
)

# save new csvs
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Dataset split complete!")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))