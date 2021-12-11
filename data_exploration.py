import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
            "male@in",
            "laki-laki@in",
            "lelaki@in",
            "jantan@in",
            "laki- laki@in",
            "pria@in",
            "men@in",
            "<http://id.dbpedia.org/resource/jantan>",
            "<http://id.dbpedia.org/resource/laki-laki>",
            "<http://id.dbpedia.org/resource/pria>",
            "<http://id.dbpedia.org/resource/lelaki>",
            "<http://id.dbpedia.org/resource/male>",
        ],
        [
            "female@in",
            "betina@in",
            "perempuan@in",
            "wanita@in",
            "women@in",
            "f@in",
            "<http://id.dbpedia.org/resource/betina>",
            "<http://id.dbpedia.org/resource/wanita>",
            "<http://id.dbpedia.org/resource/perempuan>",
        ],
        "<http://id.dbpedia.org/resource/Laki-laki>",
        "<http://id.dbpedia.org/resource/Perempuan>",
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
        "<http://dbpedia.org/property/occupation>",
        "<http://dbpedia.org/property/profession>",
        "<http://dbpedia.org/ontology/occupation>",
        "<http://purl.org/linguistics/gold/hypernym>",  # TODO this pertains to other things as well but preserves to hierarchical structure
    ],
    "id": [
        "<http://id.dbpedia.org/property/occupation>",
        "<http://id.dbpedia.org/property/profession>",
    ],
    "sv": [
        "<http://sv.dbpedia.org/property/yrke>",
        "<http://sv.dbpedia.org/property/occupation>",
    ],
}


def subsample_dbpedia(
    df: pd.DataFrame,
    frac: float,
    language: str,
    subsample_method: str = "freq",
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

    if subsample_method == "freq":
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
    else:
        df = df.sample(frac=frac)

    if keep_gender:
        df = df.append(df_gender)
    if keep_profession:
        df = df.append(df_prof)
    if keep_person:
        df = df.append(df_ppl)
    return df


def import_dbpedia_data(
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

    if language != "en":
        df.loc[
            df["tail"].str.lower().isin(GENDER_SF.get(language, [])[0]), "tail"
        ] = GENDER_SF[language][2]
        df.loc[
            df["tail"].str.lower().isin(GENDER_SF.get(language, [])[1]), "tail"
        ] = GENDER_SF[language][3]

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

    df = subsample_dbpedia(
        df, frac=subsample, language=language, subsample_method=subsample_method
    )
    print(f"Sampled {subsample} of the dataset.")

    print(
        f"# unique relations: {len(df['relation'].unique())}, "
        f"# unique heads: {len(df['head'].unique())}, "
        f"# unique tails: {len(df['tail'].unique())},"
        f"# Unique entities: {len(set(df['head'].unique()).union(set(df['tail'].unique())))}"
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


def extract_dbpedia_entities(
    data_path: str,
    language: str,
    prof_threshold: Optional[int] = None,
    human_threshold: Optional[int] = None,
):
    df = pd.read_csv(data_path, names=["head", "rel", "tail"], sep="\t")
    df_gender = df[df["rel"] == GENDER_RELATION[language]]
    df_occ = df[df["rel"] == PROFESSION_RELATION[language][0]]
    all_people = df_occ["head"].unique().tolist()
    all_people.extend(df_gender["head"].tolist())

    out_dir = os.path.dirname(data_path)
    with open(f"{out_dir}/gender.ent", "w") as f:
        f.writelines([f"{l}\n" for l in df_gender["tail"].unique().tolist()])

    with open(f"{out_dir}/professions.ent", "w") as f:
        # get professions with at least 10 people
        if prof_threshold is None:
            f.writelines([f"{l}\n" for l in df_occ["tail"].unique().tolist()])
        else:
            occs = (
                df_occ["tail"]
                .value_counts()[df_occ["tail"].value_counts() > 10]
                .index.tolist()
            )
            f.writelines([f"{l}\n" for l in occs])

    with open(f"{out_dir}/humans.ent", "w") as f:
        if human_threshold is None:
            f.writelines([f"{l}\n" for l in all_people])
        else:
            facts_per_human = df[df["head"].isin(all_people)]["head"].value_counts()
            humans = facts_per_human[facts_per_human > 5].index.tolist()
            f.writelines([f"{l}\n" for l in humans])


WIKI_GENDER_REL = "P21"  # https://www.wikidata.org/wiki/Property:P21
WIKI_OCCUPATION_REL = "P106"  # https://www.wikidata.org/wiki/Property:P106


def extract_wikidata_entities(data_path: str):
    df = pd.read_csv(data_path, names=["head", "rel", "tail"], sep="\t")
    df_gender = df[df["rel"] == WIKI_GENDER_REL]
    df_occ = df[df["rel"] == WIKI_OCCUPATION_REL]
    df_person = df[(df["rel"] == "P31") & (df["tail"] == "Q5")]

    out_dir = os.path.dirname(data_path)
    with open(f"{out_dir}/gender.ent", "w") as f:
        f.writelines([f"{l}\n" for l in df_gender["tail"].unique().tolist()])

    with open(f"{out_dir}/professions.ent", "w") as f:
        # get professions with at least 10 people
        # f.writelines([f"{l}\n" for l in df_occ["tail"].unique().tolist()])
        occs = (
            df_occ["tail"]
            .value_counts()[df_occ["tail"].value_counts() > 10]
            .index.tolist()
        )
        f.writelines([f"{l}\n" for l in occs])

    with open(f"{out_dir}/humans.ent", "w") as f:
        facts_per_human = df[df["head"].isin(df_person["head"])]["head"].value_counts()
        humans = facts_per_human[facts_per_human > 5].index.tolist()
        f.writelines([f"{l}\n" for l in humans])

    with open(f"{out_dir}/relations.rel", "w") as f:
        f.writelines([WIKI_GENDER_REL + "\n", WIKI_OCCUPATION_REL + "\n"])

    # with open(f"{out_dir}/professions.ent", "w") as f:
    # df_union = pd.merge(left=df_gender, right=df_occ, how="inner", on="head")
    #     f.writelines(df_union["head"].unique().tolist())
    # df_union.to_csv(f"{out_dir}/gender_and_profession.csv")


def import_wikidata(
    subsample: float = 1.0,
    subsample_human: float = 1.0,
    path: str = "data/wikidata5m_inductive",
):
    df = pd.read_csv(
        f"{path}/train_full.txt",
        names=["head", "relation", "tail"],
        sep="\t",
    )
    person_idx = (df["relation"] == "P31") & (df["tail"] == "Q5")
    df_person = df[person_idx]
    facts_per_human = df[df["head"].isin(df_person["head"])]["head"].value_counts()
    humans = facts_per_human[facts_per_human > 15].index.tolist()
    less_important_humans = facts_per_human[facts_per_human <= 15].index.tolist()
    df_important_person_idx = df["head"].isin(humans)
    df_person = df[df_important_person_idx]
    df = df[~df_important_person_idx]
    df_less_important_humans_idx = df["head"].isin(less_important_humans)
    df = df[~df_less_important_humans_idx]

    gender_idx = df["relation"] == WIKI_GENDER_REL
    df_gender = df[gender_idx]
    off_idx = df["relation"] == WIKI_OCCUPATION_REL
    df_occ = df[off_idx]

    df = df[(~gender_idx) & (~off_idx)]
    # df["head_count"] = df["head"].map(df["head"].value_counts())
    # df["tail_count"] = df["tail"].map(df["tail"].value_counts())
    # df["count"] = df["head_count"] + df["tail_count"]

    # df = df.drop(columns=["head_count", "tail_count"])
    # df = df.sample(weights=df["count"], frac=subsample)
    # df = df.drop(columns=["count"])
    df = df.sample(frac=subsample)

    if subsample_human < 1.0:
        df_person = df_person.sample(frac=subsample_human)
        with open(f"{path}/humans.ent", "w") as f:
            f.writelines([f"{l}\n" for l in df_person["head"].unique().tolist()])

    df = df.append(df_gender)
    df = df.append(df_occ)
    df = df.append(df_person)

    print(
        f"# unique relations: {len(df['relation'].unique())}, "
        f"# unique heads: {len(df['head'].unique())}, "
        f"# unique tails: {len(df['tail'].unique())}"
    )
    df.to_csv(f"{path}/train.txt", sep="\t", header=False, index=False)


def pd_prof_and_gender_pivot(df: pd.DataFrame) -> pd.DataFrame:
    gb_df = df.pivot_table(
        index="occupation",
        columns="gender",
        values="human",
        aggfunc=lambda x: len(x.unique()),
    ).fillna(0)


def append_dbpedia_train_count(df, language: str):
    df_train = pd.read_csv(
        f"data/dbpedia/{language}/train.txt",
        sep="\t",
        names=["head", "rel", "tail"],
    )
    df_gender = df_train[df_train["rel"] == GENDER_RELATION[language]]
    df = df.reset_index()
    df = df.drop(columns=["index"])
    for i, p in tqdm(enumerate(df["profession"])):
        df.loc[i, "count"] = sum(df_train["tail"] == p)
        gender_counts = (
            df_train[df_train["tail"] == p]
            .merge(df_gender, on="head", how="inner")["tail_y"]
            .value_counts()
        )
        for gender, cnt in zip(gender_counts.index, gender_counts):
            if gender.lower() in GENDER_SF[language][0]:
                df.loc[i, "male_count"] = cnt
            elif gender.lower() in GENDER_SF[language][1]:
                df.loc[i, "female_count"] = cnt

        if i > 30:  # only need top 20 counts
            break
    return df


def append_wikidata_alias(df: pd.DataFrame) -> pd.DataFrame:
    id2name = {}
    with open("data/wikidata5m_inductive/wikidata5m_alias/wikidata5m_entity.txt") as f:
        for line in tqdm(f):
            line = line.strip().split("\t")
            id2name[line[0]] = line[1]
    df["profession_name"] = df["profession"].apply(
        lambda x: id2name.get(x, "Not Found")
    )
    return df


def append_wikidata_train_count():
    df = pd.read_csv("data/FB15k-237/poincare_profession_name_bias_1.0_gender.csv")
    df_train = pd.read_csv(
        "data/wikidata5m_transductive/train.txt",
        sep="\t",
        names=["head", "rel", "tail"],
    )
    for i, p in tqdm(enumerate(df["profession"])):
        df.loc[i, "count"] = sum(df_train["tail"] == p)
    # pass
    df.to_csv("data/FB15k-237/poincare_profession_name_bias_1.0_gender.csv")


def append_wikidata_deviation(data_dir: str, file: str):
    df = pd.read_csv(f"{data_dir}/{file}")


def append_whole_wikidata_male_female_table():
    df = pd.read_csv("data/wikidata5m_inductive/male_female_occupation_count.csv")
    df["profession"] = df["occupation"].apply(lambda x: x.split("/")[-1])
    df = append_wikidata_alias(df)
    df["m2f_ratio"] = df["male"] / df["female"]
    df = df.sort_values(["m2f_ratio"], ascending=False)
    df = df[df["male"] + df["female"] > 10]
    df = df.drop(columns=["occupation"])
    df.to_csv(
        "data/wikidata5m_inductive/male_female_occupation_count_filtered.csv",
        index=False,
    )


FB_GENDER_REL = "/people/person/gender"
FB_OCCUPATION_REL = "/people/person/profession"


def extract_FB15K_entities(data_path: str):
    df = pd.read_csv(data_path, names=["head", "rel", "tail"], sep="\t")
    df_gender = df[df["rel"] == FB_GENDER_REL]
    df_occ = df[df["rel"] == FB_OCCUPATION_REL]
    df_person = df[df["rel"].str.startswith("/people/person")]

    out_dir = os.path.dirname(data_path)
    with open(f"{out_dir}/gender.ent", "w") as f:
        f.writelines([f"{l}\n" for l in df_gender["tail"].unique().tolist()])

    with open(f"{out_dir}/professions.ent", "w") as f:
        # get professions with at least 10 people
        f.writelines([f"{l}\n" for l in df_occ["tail"].unique().tolist()])
        # occs = (
        #     df_occ["tail"]
        #     .value_counts()[df_occ["tail"].value_counts() > 10]
        #     .index.tolist()
        # )
        # f.writelines([f"{l}\n" for l in occs])

    with open(f"{out_dir}/humans.ent", "w") as f:
        # facts_per_human = df[df["head"].isin(df_person["head"])]["head"].value_counts()
        # humans = facts_per_human[facts_per_human > 5].index.tolist()
        # f.writelines([f"{l}\n" for l in humans])
        f.writelines([f"{l}\n" for l in df_person["head"].unique().tolist()])


def append_wikidata_equivalent(df):
    id2name = {}
    with open("data/FB15k-237/fb2w.nt") as f:
        for line in tqdm(f):
            if not line.startswith("#") and len(line) > 1:
                line = line.strip().split("\t")
                id2name[line[0]] = line[2]
    df["fb_profession"] = df["profession"]
    df["profession"] = df["fb_profession"].apply(
        lambda x: id2name.get(
            f"<http://rdf.freebase.com/ns/{x[1:].replace('/','.')}>", "Not Found"
        )
    )
    df["profession"] = df["profession"].apply(
        lambda x: x.split("/")[-1].replace("> .", "")
    )
    return df


def append_FB15k_train_count(df):
    # df = pd.read_csv("data/FB15k-237/poincare_profession_name_bias_1.0_gender_male.csv")
    df_train = pd.read_csv(
        "data/FB15k-237/train.txt",
        sep="\t",
        names=["head", "rel", "tail"],
    )
    df_gender = df_train[df_train["rel"] == "/people/person/gender"]
    df = df.reset_index()
    df = df.drop(columns=["index"])
    for i, p in tqdm(enumerate(df["fb_profession"])):
        df.loc[i, "count"] = sum(df_train["tail"] == p)
        gender_counts = (
            df_train[df_train["tail"] == p]
            .merge(df_gender, on="head", how="inner")["tail_y"]
            .value_counts()
        )
        for gender, cnt in zip(gender_counts.index, gender_counts):
            df.loc[i, gender] = cnt
    df = df.rename(columns={"/m/05zppz": "male_count", "/m/02zsn": "female_count"})
    # pass
    # df.to_csv("data/FB15k-237/poincare_profession_name_bias_1.0_gender_male.csv")
    return df


def print_dataset_statistics(path: str):
    df = pd.read_csv(f"{path}/train.txt", sep="\t", names=["head", "relation", "tail"])
    print("train length:", len(df))
    cur_len = len(df)
    df = df.append(
        pd.read_csv(f"{path}/test.txt", sep="\t", names=["head", "relation", "tail"])
    )
    print("test length:", len(df) - cur_len)
    cur_len = len(df)
    df = df.append(
        pd.read_csv(f"{path}/valid.txt", sep="\t", names=["head", "relation", "tail"])
    )
    print("valid length:", len(df) - cur_len)

    print(
        f"# unique relations: {len(df['relation'].unique())}, "
        f"# unique heads: {len(df['head'].unique())}, "
        f"# unique tails: {len(df['tail'].unique())},"
        f"# Unique entities: {len(set(df['head'].unique()).union(set(df['tail'].unique())))}"
    )

    if "dbpedia" in path:
        lg = path.split("/")[-1]
        gender_triples = sum(df["relation"] == GENDER_RELATION[lg])
        print(f"gender triples:{gender_triples}")
    else:
        gender_triples = sum(df["relation"] == FB_GENDER_REL)
        print(f"gender triples:{gender_triples}")


if __name__ == "__main__":
    np.random.seed(42)
    # import_data("zh")
    # import_data("ky", subsample=0.1, subsample_method="freq")

    # import_dbpedia_data("en", subsample=0.1, subsample_method="random")
    extract_dbpedia_entities(
        "data/dbpedia/en/train.txt",
        language="en",
        human_threshold=True,
        prof_threshold=True,
    )
    # print_dataset_statistics("data/dbpedia/en")

    # import_dbpedia_data("id")
    # extract_dbpedia_entities("data/dbpedia/id/train.txt", language="id")
    # print_dataset_statistics("data/dbpedia/id")

    # import_dbpedia_data("sv", subsample=0.2, subsample_method="random")
    # extract_dbpedia_entities("data/dbpedia/sv/train.txt", language="sv")
    print_dataset_statistics("data/dbpedia/sv")

    # extract_wikidata_entities("data/wikidata5m_transductive/train.txt")
    # import_wikidata(0.05, 1.0, "data/wikidata5m_transductive")

    # append_whole_wikidata_male_female_table()
    # extract_FB15K_entities("data/FB15k-237/train.txt")
    # print_dataset_statistics("data/FB15k-237")
