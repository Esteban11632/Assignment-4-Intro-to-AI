# Esteban Murillo & Ezequiel Buck
# Homework 4
# Intro to AI

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Read the CSV files
df = pd.read_csv('credit_delinquency_v2.csv')

fractions = ['C_fraction', '0_fraction', '1_fraction']

# Plot bar chart of the C, 0, 1 ratios of the STATUS column
for col in fractions:
    df.hist(column=[col], bins=50)
    plt.title(col)
    # Save graph only if graphs folder exists
    if os.path.exists('graphs'):
        plt.savefig(f'graphs/{col}_histogram.png',
                    dpi=300, bbox_inches='tight')
    plt.show()

# Are customers with more on-time payments less likely to become delinquent?
# Are customers with more late payments more likely to become delinquent?
# Is there a relationship between having closed accounts and delinquency?
# Plot a bar chart of the Delinquent vs. the fractions to find out.
for col in fractions:
    plt.figure(figsize=(10, 6))
    df[col] = pd.qcut(df[col], 20, duplicates="drop")
    lm = sns.barplot(data=df, x=col, y="Delinquent")
    plt.xticks(rotation=30, ha='right')
    plt.title(f"Delinquent vs. {col}")
    plt.xlabel(f"{col} Intervals")
    plt.ylabel("Delinquency Rate")
    plt.tight_layout()
    # Save graph only if graphs folder exists
    if os.path.exists('graphs'):
        plt.savefig(f'graphs/delinquent_vs_{col}.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
