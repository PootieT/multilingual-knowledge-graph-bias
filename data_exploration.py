import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

GENDER_SF = {
    "en": [
        [
            "male@en",
            "m@en",
            "boy@en",
            "boys@en",
            "males@en",
        ],
        [
            "f@en",
            "female@en",
            "w@en",
            "women@en",
            "girls@en",
            "girl@en",
        ],
        "<http://dbpedia.org/resource/Male>",
        "<http://dbpedia.org/resource/Female>",
    ],
    "id": [
        [
            "male@id",
            "laki-laki@id",
            "lelaki@id",
            "jantan@id",
            "laki- laki@id",
            "pria@id",
            "<http://id.dbpedia.org/resource/jantan>",
            "<http://id.dbpedia.org/resource/laki-laki>",
            "<http://id.dbpedia.org/resource/pria>",
            "<http://id.dbpedia.org/resource/lelaki>",
        ],
        [
            "female@id",
            "betina@id",
            "perempuan@id",
            "wanita@id",
            "<http://id.dbpedia.org/resource/betina>",
            "<http://id.dbpedia.org/resource/wanita>",
            "<http://id.dbpedia.org/resource/perempuan>",
        ],
        "<http://id.dbpedia.org/resource/Male>",
        "<http://id.dbpedia.org/resource/Female>",
    ],
    "sv": [
        [
            "herr@sv",
            "herrar@sv",
            "man@sv",
            "m@sv",
            "hingst@sv",
            "manligt@sv",
            "pojke@sv",
            "hane@sv",
            "<http://sv.dbpedia.org/resource/valack>",
            "<http://sv.dbpedia.org/resource/hingst>",
            "<http://sv.dbpedia.org/resource/klapphingst>",
            "<http://sv.dbpedia.org/resource/pojke>",
        ],
        [
            "kvinna@sv",
            "dam@sv",
            "f@sv",
            "kvinnligt@sv",
            "hona@sv",
            "female@sv",
            "sto@sv",
            "flicka@sv",
            "hona, trots mansröst.@sv",
            "<http://sv.dbpedia.org/resource/sto>",
            "<http://sv.dbpedia.org/resource/kvinna>",
            "<http://sv.dbpedia.org/resource/flicka>",
            "<http://sv.dbpedia.org/resource/kvinnor>",
        ],
        "<http://sv.dbpedia.org/resource/Man>",
        "<http://sv.dbpedia.org/resource/Kvinna>",
    ],
}

GENDER_RELATION = {
    "en": "<http://dbpedia.org/property/gender>",
    "id": "<http://id.dbpedia.org/property/gender>",
    "sv": "<http://sv.dbpedia.org/property/kön>",
}

PROFESSION_RELATION = {
    "en": [
        "<http://dbpedia.org/property/profession>",
        "<http://dbpedia.org/ontology/occupation>",
        "<http://dbpedia.org/property/occupation>",
        "<http://purl.org/linguistics/gold/hypernym>",  # TODO this pertains to other things as well but preserves to hierarchical structure
    ],
    "id": [
        "<http://id.dbpedia.org/property/profession>",
        "<http://id.dbpedia.org/property/occupation>",
    ],
    "sv": [
        "<http://sv.dbpedia.org/property/yrke>",
        "<http://sv.dbpedia.org/property/occupation>",
    ],
}


def drop_least_frequent_entities(
    df: pd.DataFrame,
    frac: float,
    language: str,
    keep_gender=True,
    keep_profession=True,
    keep_person=True,
) -> pd.DataFrame:
    if keep_gender:
        gender_idx = df["relation"] == GENDER_RELATION.get(
            language, f"<http://{language}.dbpedia.org/property/gender>"
        )
        df_gender = df[gender_idx]
        df = df[~gender_idx]
    if keep_profession:
        prof_idx = (
            df["relation"].str.lower().isin(PROFESSION_RELATION.get(language, []))
        )
        df_prof = df[prof_idx]
        df = df[~prof_idx]
    if keep_person:
        person_w_gender_path = f"data/dbpedia/{language}/person_with_gender.ent"
        if os.path.exists(person_w_gender_path):
            with open(person_w_gender_path) as f:
                person_list = [l.strip() for l in f.readlines()]
            ppl_idx = df["head"].isin(person_list)
            df_ppl = df[ppl_idx]
            df = df[~ppl_idx]
        else:
            df_ppl = pd.DataFrame()

    df["head_count"] = df["head"].map(df["head"].value_counts())
    df["tail_count"] = df["tail"].map(df["tail"].value_counts())
    # df["rel_count"] = df["relation"].map(df["relation"].value_counts())
    # add log so distribution can be more even, otherwise most will be dominated by popular entities
    df["count"] = df["head_count"] + df["tail_count"]

    df = df.drop(columns=["head_count", "tail_count"])
    df = df.sample(weights=df["count"], frac=frac)

    # rel_to_keep = df["relation"].value_counts().sort_values(
    #     ascending=False
    # ).cumsum() / len(df) < (frac)
    # relations = rel_to_keep.index[: sum(rel_to_keep)]
    # df = df[df["relation"].isin(relations)]
    df = df.drop(columns=["count"])

    if keep_gender:
        df = df.append(df_gender)
    if keep_profession:
        df = df.append(df_prof)
    if keep_person:
        df = df.append(df_ppl)
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
    df = remove_parse_errors(df)

    # filter triples so only ones left are the ones with tails also being a resource (not some surface form/facts)
    dbpedia_str = (
        "<http://dbpedia.org/resource/"
        if language == "en"
        else f"<http://{language}.dbpedia.org/resource/"
    )
    orig_len = len(df)

    df.loc[df["tail"].isin(GENDER_SF.get(language, [])[0]), "tail"] = GENDER_SF[
        language
    ][2]
    df.loc[df["tail"].isin(GENDER_SF.get(language, [])[1]), "tail"] = GENDER_SF[
        language
    ][3]

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
        df = drop_least_frequent_entities(df, frac=subsample, language=language)
    else:
        df = df.sample(frac=subsample)
    print(f"Sampled {subsample} of the dataset.")

    print(
        f"# unique relations: {len(df['relation'].unique())}, "
        f"# unique heads: {len(df['head'].unique())}, "
        f"# unique tails: {len(df['tail'].unique())}"
    )

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


def remove_parse_errors(df):
    orig_len = len(df)
    df = df[df["parse_error"] == "."]
    df = df.drop(columns=["parse_error"])
    drop_len = len(df)
    diff_len = orig_len - drop_len
    print(
        f"Dropped {diff_len} ({(diff_len / orig_len):.4f}) of the incorrectly parsed rows. Now data length = {drop_len}"
    )
    return df


if __name__ == "__main__":
    np.random.seed(42)
    # import_data("en", subsample=0.1, subsample_method="freq")
    # import_data("zh")
    # import_data("ky", subsample=0.1, subsample_method="freq")
    import_data("id")
    import_data("sv", subsample=0.2, subsample_method="freq")
