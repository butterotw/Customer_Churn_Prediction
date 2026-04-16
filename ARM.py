import argparse
import os
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth

#Designed for user to customize the parameters for ARM
def main():
    ap = argparse.ArgumentParser()
    #add default parameters to the parser
    ap.add_argument("data", default="BankChurners_clean.csv")
    ap.add_argument("output", default=os.path.join("artifacts", "frequent_itemsets.csv"))
    ap.add_argument("min_support", type=float, default=0.02)
    ap.add_argument("max_len", type=int, default=3)
    ap.add_argument("min_customers", type=int, default=100)
    args = ap.parse_args() #get parameters from end user
    df = pd.read_csv(args.data) 

    #Binning features from numerical(ordinal) to categorical for ARM rules
    df["Utilization_Level"] = pd.qcut(df["Avg_Utilization_Ratio"], q=3, labels=["Low", "Medium", "High"])
    df["Transaction_Amount_Level"] = pd.qcut(df["Total_Trans_Amt"], q=3, labels=["Low", "Medium", "High"])
    df["Transaction_Count_Level"] = pd.qcut(df["Total_Trans_Ct"], q=3, labels=["Low", "Medium", "High"])
    df["Inactivity_Level"] = pd.cut(df["Months_Inactive_12_mon"], bins=[-1, 1, 3, df["Months_Inactive_12_mon"].max()], labels=["Few", "Moderate", "Many"],)
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=[0, 35, 55, df["Customer_Age"].max()], labels=["Adult", "Midlife", "Elderly"],)

    cat_col = df.select_dtypes(include='object').drop(columns=['Attrition_Flag']).columns.tolist()
    #Used features for mining rules
    fp_features = cat_col + [
        "Utilization_Level",
        "Transaction_Amount_Level",
        "Transaction_Count_Level",
        "Inactivity_Level",
    ]
    use_df = df[fp_features].copy() #a dataframe with only the used features 

    #outcome vector for churn rate
    y = (df["Attrition_Flag"] == "Attrited Customer").astype(int).values

    # one-hot encoding
    df_encoded = pd.get_dummies(use_df)
    X = df_encoded.astype(bool)

    # mine frequent itemsets using parameters from end user
    freq = fpgrowth(X, min_support=args.min_support, use_colnames=True, max_len=args.max_len)
    if freq.empty:
        out = pd.DataFrame(columns=["items", "item_count", "support", "customers", "churn_rate"])
    else:
        freq = freq.sort_values("support", ascending=False)

        rows = []
        # compute customers + churn rate for each itemset
        for _, r in freq.iterrows():
            items = list(r["itemsets"])
            mask = X[items].all(axis=1).values #mask for customers that achieve the itemset
            customers = int(mask.sum())
            if customers < args.min_customers:
                continue
            churn_rate = float(y[mask].mean()) if customers > 0 else 0.0 #calculate churn rate
            rows.append({
                "items": " + ".join(items),
                "item_count": int(len(items)),
                "support": float(r["support"]),
                "customers": customers,
                "churn_rate": churn_rate,
            })

        out = pd.DataFrame(rows) 

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    #record the results to a meta.json file
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    meta = {
        "data": args.data,
        "min_support": args.min_support,
        "max_len": args.max_len,
        "min_customers": args.min_customers,
        "rows_written": int(len(out)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)

    print(f"Saved: {args.out}")
    print(f"Saved: {meta_path}")
    print(f"Rows: {len(out)}")

if __name__ == "__main__":
    main()
