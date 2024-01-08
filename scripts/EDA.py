import os
import sys
import click
import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
import altair_ally as aly
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



@click.command()
@click.option('--input_filepath', type=str, help="Path to raw data")
@click.option('--output_filepath', type=str, help="Path to directory where processed data csv will be written to")
def main(input_filepath, output_filepath):
    # reading and
    df = pd.read_csv(input_filepath, encoding="utf-8")
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=123)
    data_description = train_df.describe().T
    data_description = data_description.reset_index().rename(columns={'index': 'variable'})
    pd.DataFrame(data_description).to_csv(os.path.join(output_filepath, "tables", "description.csv"), index=False)
    print(data_description)

    # finding missing value
    zero_val = (train_df == 0.00).astype(int).sum(axis=0)
    mis_val = train_df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(train_df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={0: 'Zero Values', 1: 'Missing Values', 2: '% of Total Values'})
    mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
    mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(train_df)
    mz_table['Data Type'] = train_df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(train_df.shape[1]) + " columns and " + str(train_df.shape[0]) + " Rows.\n"
                                                                                                       "There are " + str(
        mz_table.shape[0]) +
          " columns that have missing values.")
    mz_table = mz_table.reset_index().rename(columns={'index': 'variable'})
    pd.DataFrame(mz_table).to_csv(os.path.join(output_filepath, "tables", "missing_values.csv"), index=False)

    # correlation
    #correlation_plot = aly.corr(train_df)
    #correlation_plot.save(os.path.join(output_filepath, "figures", "correlation.png"), index=False)

    # numeric distribution
    alt.data_transformers.enable("vegafusion")

    non_empty_numeric_cols = [col for col in train_df.select_dtypes(
    "number").columns if train_df[col].notna().any()]
    numeric_cols_dist = alt.Chart(train_df).mark_bar().encode(
        alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=20)),
        y="count()",
    ).properties(
        width=250,
        height=150
    ).repeat(
        non_empty_numeric_cols,
        columns=3
    )
    numeric_cols_dist.save(os.path.join(output_filepath, "figures", "numeric_dist.png"))

    # categrical distribution
    cat_cols = list(train_df.select_dtypes("object").columns.drop("time"))

    categorical_cols_dist = (
        alt.Chart(train_df)
        .mark_bar()
        .encode(
            x=alt.X(alt.repeat("repeat"), type="nominal"),
            y="count()",
        )   
        .properties(width=550, height=150)
        .repeat(repeat=cat_cols, columns=3)
    )
    categorical_cols_dist.save(os.path.join(output_filepath, "figures", "categ_dist.png"))

    #service name vs clicks plot
    plt.figure(figsize=(8, 6))
    service_name = sns.scatterplot(x="ext_service_name", y="clicks", data=train_df)

    plt.xlabel("Service Name", size=13)
    plt.ylabel("Clicks", size=13)
    plt.title("Service name vs clicks", size=15, weight="bold")
    plt.savefig(os.path.join(output_filepath, "figures", "service_name.png"))


    # channel name vs clicks plot
    plt.figure(figsize=(8, 6))
    channel_name = sns.scatterplot(x="channel_name", y="clicks", data=train_df)

    plt.xlabel("Channel name", size=13)
    plt.ylabel("Clicks", size=13)
    plt.title("Channel name vs clicks", size=15, weight="bold")
    plt.savefig(os.path.join(output_filepath, "figures", "channel_name.png"))


if __name__ == '__main__':
    main()
