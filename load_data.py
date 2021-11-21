import pandas as pd
from itertools import cycle, islice
from torch.utils.data import Dataset, IterableDataset


def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def add_reverse_triple_df(df: pd.DataFrame) -> pd.DataFrame:
    df_rev = df.copy()
    df_rev["head"] = df["tail"]
    df_rev["tail"] = df["head"]
    df_rev["relation"] = df_rev["relation"].astype(str) + "_reverse"
    df = df.append(df_rev, ignore_index=True)
    return df


class Data:
    def __init__(self, data_dir="data/WN18RR/"):
        with open(f"{data_dir}train.txt", "rb") as fp:
            c_generator = _count_generator(fp.raw.read)
            num_lines = sum(
                buffer.count(b"\n") for buffer in c_generator
            )  # count each \n

        if num_lines < 4 * 10e5:
            self.train_data = self.load_data(data_dir, "train")
            self.valid_data = self.load_data(data_dir, "valid")
            self.test_data = self.load_data(data_dir, "test")
            self.data = self.train_data + self.valid_data + self.test_data  # never used
            self.entities = self.get_entities(self.data)
            self.train_relations = self.get_relations(self.train_data)
            self.valid_relations = self.get_relations(self.valid_data)
            self.test_relations = self.get_relations(self.test_data)
            self.relations = (
                self.train_relations
                + [i for i in self.valid_relations if i not in self.train_relations]
                + [i for i in self.test_relations if i not in self.train_relations]
            )
            self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
            self.relation_idxs = {
                self.relations[i]: i for i in range(len(self.relations))
            }
            print(
                f"# entities: {len(self.entity_idxs)}, # relations: {len(self.relation_idxs)}, "
            )
        else:  # if a large file, load as pandas dataframe
            print("dataset too large, using pandas")
            self.train_data = self.load_data_pd(data_dir, "train")
            self.valid_data = self.load_data_pd(data_dir, "valid")
            self.test_data = self.load_data_pd(data_dir, "test")

            add_reverse_triple_df(self.train_data)
            add_reverse_triple_df(self.valid_data)
            add_reverse_triple_df(self.test_data)

            self.entities = self.get_entities_pd()
            self.train_relations = self.get_relations_pd(self.train_data)
            self.valid_relations = self.get_relations_pd(self.valid_data)
            self.test_relations = self.get_relations_pd(self.test_data)
            self.relations = (
                self.train_relations
                + [i for i in self.valid_relations if i not in self.train_relations]
                + [i for i in self.test_relations if i not in self.train_relations]
            )

            self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
            self.relation_idxs = {
                self.relations[i]: i for i in range(len(self.relations))
            }
            self.on_hot_encode_df(self.train_data)
            self.on_hot_encode_df(self.valid_data)
            self.on_hot_encode_df(self.test_data)

    @staticmethod
    def load_data(data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            # data += [[i[2], i[1]+"_reverse", i[0]] for i in data]  # moved to data preprocessing
        return data

    @staticmethod
    def load_data_pd(data_dir, data_type="train"):
        return pd.read_csv(
            f"{data_dir}/{data_type}.txt", names=["head", "relation", "tail"], sep="\t"
        )

    @staticmethod
    def get_relations(data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    @staticmethod
    def get_relations_pd(df):
        relations_set = set(df["relation"].unique())
        relations = sorted(list(relations_set))
        return relations

    @staticmethod
    def get_entities(data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def get_entities_pd(self):
        relations_set = set([])
        for df in [self.train_data, self.valid_data, self.test_data]:
            relations_set = relations_set.union(set(df["head"].unique()))
            relations_set = relations_set.union(set(df["tail"].unique()))

        relations = sorted(list(relations_set))
        return relations

    def on_hot_encode_df(self, df: pd.DataFrame):
        df["head"] = df["head"].map(self.entity_idxs)
        df["relation"] = df["relation"].map(self.relation_idxs)
        df["tail"] = df["tail"].map(self.entity_idxs)


class DBpediaIterableDataset(IterableDataset):
    """
    if pandas is too large, gonna need this dataset class to stream in data from file.
    This guy is really good
    https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse_file(self, file_path: str):
        with open(file_path, "r") as f:
            for line in f:
                triple = line.strip().split()
                yield from triple

    def get_stream(self, file_path: str):
        return cycle(self.parse_file(file_path))

    def __iter__(self):
        return self.get_stream(self.file_path)
