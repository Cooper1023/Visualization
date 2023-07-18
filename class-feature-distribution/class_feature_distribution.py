import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mushroom-data.csv")


def hide_spines(ax, spines=["top", "right", "left", "bottom"]):
    for spine in spines:
        ax.spines[spine].set_visible(False)


# 提取特征变量名
features = df.columns.drop(['class']).tolist()

fig = plt.figure(figsize=(16, 9))
for idx, feature in enumerate(features):
    ax = fig.add_subplot(4, 3, idx + 1)
    ax.grid(axis="y", linewidth=1, linestyle="--", zorder=0)
    sns.countplot(x=feature, palette="Blues", hue="class", data=df, alpha=1, linewidth=1.5, zorder=2)
    # 根据分类计数
    feature_data_p = df[df["class"] == "p"][feature].value_counts()
    feature_data_e = df[df["class"] == "e"][feature].value_counts()

    for idx_p in feature_data_p.index:
        if idx_p not in feature_data_e.index:
            feature_data_e[idx_p] = 0

    for idx_e in feature_data_e.index:
        if idx_e not in feature_data_p.index:
            feature_data_p[idx_e] = 0

    feature_data_p = feature_data_p.sort_index()
    feature_data_e = feature_data_e.sort_index()

    if idx % 3 == 0:
        ax.set_ylabel("Count", fontsize=14, fontfamily="serif", labelpad=7)
    else:
        ax.set_ylabel("")

    ax.set_xlabel(feature, fontsize=14, fontfamily="serif", labelpad=7)
    hide_spines(ax, spines=["top", "right", "left"])
    ax.spines["bottom"].set(linewidth=2)
    ax.set_ylim(1)
    ax.legend()

fig.text(x=0.05, y=1.01, s="Class Features Distributions", fontsize=22, fontweight="bold", fontfamily="serif")
fig.tight_layout(w_pad=2, h_pad=1.5)
plt.savefig('Class_Features_Distributions.png',
            dpi=300,
            bbox_inches='tight')
plt.show()
