import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().drop_duplicates()

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df.to_csv(output_path, index=False)