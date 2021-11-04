import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def drop_least_frequent_entities(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    df["head_count"] = df["head"].map(df["head"].value_counts())
    df["tail_count"] = df["tail"].map(df["tail"].value_counts())
    # add log so distribution can be more even, otherwise most will be dominated by popular entities
    df["count"] = np.log(df["head_count"] + df["tail_count"])
    df = df.drop(columns=["head_count", "tail_count"])
    df = df.sample(weights=df["count"], frac=frac)
    return df


def import_data(
    language: str,
    keep_literals: bool = False,
    subsample: float = 1.0,
    subsample_method: str = "freq",
):
    """
    Read in raw .ttl data and filter through data
    :param language:
    :param keep_literals: if True, tail concepts like "6"^^<http://www.w3.org/2001/XMLSchema#integer> will be kept
    :return:
    """
    data_path = f"data/dbpedia/infobox-properties_lang={language}.ttl"
    df = pd.read_table(
        data_path, sep=" ", names=["head", "relation", "tail", "parse_error"]
    )

    # remove some hard to parse ones (usually will be filtered out later too)
    orig_len = len(df)
    df = df[df["parse_error"] == "."]
    df = df.drop(columns=["parse_error"])
    drop_len = len(df)
    diff_len = orig_len - drop_len
    print(
        f"Dropped {diff_len} ({(diff_len/orig_len):.4f}) of the incorrectly parsed rows. Now data length = {drop_len}"
    )

    # filter triples so only ones left are the ones with tails also being a resource (not some surface form/facts)
    dbpedia_str = (
        "<http://dbpedia.org/resource/"
        if language == "en"
        else f"<http://{language}.dbpedia.org/resource/"
    )
    orig_len = len(df)
    if keep_literals:
        # avoid filtering out date, year, literals that NELL dataset also include
        mask = df["tail"].str.contains(dbpedia_str) | ~df["tail"].str.contains(
            f"@{language}"
        )
    else:
        mask = df["tail"].str.contains(dbpedia_str)
    df = df[mask]
    drop_len = len(df)
    diff_len = orig_len - drop_len
    print(
        f"Dropped {diff_len} ({(diff_len / orig_len):.4f}) of data with tail not being a resource. Now data length = {drop_len}"
    )

    if subsample_method == "freq":
        df = drop_least_frequent_entities(df, frac=subsample)
    else:
        df = df.sample(frac=subsample)
    train, test_valid = train_test_split(df, test_size=0.1)
    test, valid = train_test_split(test_valid, test_size=0.5)

    os.makedirs(f"data/dbpedia/{language}", exist_ok=True)
    train.to_csv(
        f"data/dbpedia/{language}/train.txt", sep="\t", header=False, index=False
    )
    valid.to_csv(
        f"data/dbpedia/{language}/valid.txt", sep="\t", header=False, index=False
    )
    test.to_csv(
        f"data/dbpedia/{language}/test.txt", sep="\t", header=False, index=False
    )

    print(
        f"{language} dbpedia data created: \n"
        f"Train: {len(train)}, valid: {len(valid)}, test: {len(test)}"
    )


if __name__ == "__main__":
    np.random.seed(42)
    import_data("en", subsample=0.1, subsample_method="freq")
    # import_data("zh")
    # import_data("ky", subsample=0.1, subsample_method="freq")
