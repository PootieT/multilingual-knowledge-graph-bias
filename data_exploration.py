import pandas as pd


def import_data(language: str):
    data_path = f"data/dbpedia/infobox-properties_lang={language}.ttl"
    df = pd.read_table(data_path, sep=" ", names=["head", "relation", "tail", "parse_error"])

    # remove some hard to parse ones (usually will be filtered out later too)
    orig_len = len(df)
    df = df[df["parse_error"]=="."]
    df = df.drop(columns=["parse_error"])
    drop_len = len(df)
    diff_len = orig_len - drop_len
    print(f"Dropped {diff_len} ({(diff_len/orig_len):.2f}) of the incorrectly parsed rows. Now data length = {drop_len}")

    # filter triples so only ones left are the ones with tails also being a resource (not some surface form/facts)
    dbpedia_str = "<http://dbpedia.org/resource/" if language=="en" else f"<http://{language}.dbpedia.org/resource/"
    orig_len = len(df)
    df = df[df["tail"].str.contains(dbpedia_str)]
    drop_len = len(df)
    diff_len = orig_len - drop_len
    print(f"Dropped {diff_len} ({(diff_len / orig_len):.2f}) of data with tail not being a resource. Now data length = {drop_len}")
    # TODO could probably avoid filtering out date, year, literals that NELL dataset also include

    df.to_csv(f"data/dbpedia/{language}_filtered.txt", sep="\t")
    return df


if __name__ == "__main__":
    # import_data("en")
    import_data("zh")
    # import_data("ky")