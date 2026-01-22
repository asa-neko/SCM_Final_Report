import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. データの読み込み
df = pd.read_excel('kadai.xlsx')

# 時系列順にソート
if 'process_end_time' in df.columns:
    df = df.sort_values('process_end_time')
df = df.dropna(subset=['OV'])

# 2. 説明変数(X)と目的変数(y)の定義
X_cols = [f'X{i}' for i in range(1, 84)]
existing_X_cols = [col for col in X_cols if col in df.columns]

X = df[existing_X_cols]
y = df['OV']

# 3. 相関分析で重要変数を特定
correlations = {}
for col in existing_X_cols:
    valid_mask = X[col].notna() & y.notna()
    if valid_mask.sum() > 1:
        corr, p_value = pearsonr(X.loc[valid_mask, col], y[valid_mask])
        correlations[col] = abs(corr)
    else:
        correlations[col] = 0

# 4. 上位の重要な変数を特定（NaNを除外）
top_vars_all = sorted(correlations.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
top_vars = [v for v in top_vars_all[:10] if not np.isnan(v[1])][:5]
print("【OVとの相関係数が高い上位5変数】")
for var, corr in top_vars:
    print(f"{var}: {corr:.4f}")

# 5. 最も相関が高い2つの変数で分析
top_2_var_names = [var for var, _ in top_vars[:2]]
print(f"\n最も重要な2変数: {', '.join(top_2_var_names)}")

# 6. トップ2変数で平均値を基準に4象限分析
if len(top_2_var_names) >= 2:
    threshold_1 = X[top_2_var_names[0]].median()
    threshold_2 = X[top_2_var_names[1]].median()
    
    quad1 = ((X[top_2_var_names[0]] > threshold_1) & (X[top_2_var_names[1]] > threshold_2))
    quad2 = ((X[top_2_var_names[0]] <= threshold_1) & (X[top_2_var_names[1]] > threshold_2))
    quad3 = ((X[top_2_var_names[0]] <= threshold_1) & (X[top_2_var_names[1]] <= threshold_2))
    quad4 = ((X[top_2_var_names[0]] > threshold_1) & (X[top_2_var_names[1]] <= threshold_2))
    
    print(f"\n【{top_2_var_names[0]} と {top_2_var_names[1]} による4象限分析】")
    print(f"第1象限 ({top_2_var_names[0]}>{threshold_1:.2f}, {top_2_var_names[1]}>{threshold_2:.2f}): OV平均={y[quad1].mean():.4f}, 件数={quad1.sum()}")
    print(f"第2象限 ({top_2_var_names[0]}<={threshold_1:.2f}, {top_2_var_names[1]}>{threshold_2:.2f}): OV平均={y[quad2].mean():.4f}, 件数={quad2.sum()}")
    print(f"第3象限 ({top_2_var_names[0]}<={threshold_1:.2f}, {top_2_var_names[1]}<={threshold_2:.2f}): OV平均={y[quad3].mean():.4f}, 件数={quad3.sum()}")
    print(f"第4象限 ({top_2_var_names[0]}>{threshold_1:.2f}, {top_2_var_names[1]}<={threshold_2:.2f}): OV平均={y[quad4].mean():.4f}, 件数={quad4.sum()}")

print("\nAnalysis complete!")
