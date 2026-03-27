import pandas as pd

print("=== bilstm_cicids.csv FULL CHECK ===")
df = pd.read_csv(
    'data/raw/new_dataset/team_datasets/person2_layer2/bilstm_cicids.csv'
)
print(f"Full shape : {df.shape}")
print(f"Nulls      : {df.isnull().sum().sum()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nis_attack distribution:")
print(df['is_attack'].value_counts())
print(f"\nFeature stats:")
print(df.describe())