import pandas as pd


def concatenate_column_values(dframe, cols):
    """

    @param dframe: Pandas DataFrame
    @param cols: List of columns
    @return: Pandas DataFrame
    """
    return pd.Series(
        map(
            ''.join,
            dframe[cols].values.astype(str).tolist()
        ),
        index=dframe.index
    )
