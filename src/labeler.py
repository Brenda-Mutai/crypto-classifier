import pandas as pd
data=pd.read_csv(r"C:\Users\BRENDA MUTAI\Documents\crypto-classifier\data\processed\crypto_data_feature_engineered.csv")

# label generation based on future returns
data["future_return"] = data["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1

data["label"] = data.apply(label, axis=1)
data['label_name'] = data['label'].map({0:'SELL', 1:'HOLD', 2:'BUY'})
data.head()
data.to_csv(r"C:\Users\BRENDA MUTAI\Documents\crypto-classifier\data\processed\crypto_data_features_labels.csv", index=False)