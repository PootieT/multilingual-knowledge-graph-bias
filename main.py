from typing import Optional, List, Union
import os
import numpy as np
import pandas as pd
import torch
import wandb
import time
from collections import defaultdict

from tqdm import tqdm

from load_data import Data
from model import *
from rsgd import *
import argparse


class Experiment:
    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.learning_rate = args.lr
        self.dim = args.dim
        self.nneg = args.nneg
        self.num_iterations = args.num_iterations
        self.batch_size = args.batch_size
        self.cuda = args.cuda
        self.model_save_path = f"{args.model_save_path}/{'/'.join(args.data_dir.split('/')[1:])}{self.model}_model.pt"
        self.model_reload_path = args.model_reload_path

    def get_data_indices(
        self, data: Union[List, pd.DataFrame], entity_idxs, relation_idxs
    ):
        if type(data) == pd.DataFrame:
            data_indices = data[["head", "relation", "tail"]]
        else:
            data_indices = [
                (
                    entity_idxs[data[i][0]],
                    relation_idxs[data[i][1]],
                    entity_idxs[data[i][2]],
                )
                for i in range(len(data))
            ]
        return data_indices

    def get_er_vocab(self, data: Union[List, pd.DataFrame], idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        if isinstance(data, list):
            for triple in data:
                er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        else:
            print("Getting entity-relation dictionary")
            er_vocab = {
                k: g.to_list() for k, g in data.groupby(["head", "relation"])["tail"]
            }
        return er_vocab

    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)

    def load_model(self, model):
        if self.model_reload_path:
            model.load_state_dict(torch.load(self.model_reload_path))

    def log_eval(self, hits: List[List[float]], ranks: np.array):
        hits_10 = np.mean(hits[9])
        hits_3 = np.mean(hits[2])
        hits_1 = np.mean(hits[0])
        mean_rank = np.mean(ranks)
        mrr = np.mean(1.0 / np.array(ranks))

        print(f"Hits @10: {hits_10:.5f}")
        print(f"Hits @3: {hits_3:.5f}")
        print(f"Hits @1: {hits_1:.5f}")
        print(f"Mean rank: {mean_rank:.5f}")
        print(f"Mean reciprocal rank: {mrr:.5f}")

        wandb.log(
            {
                "hits_10": hits_10,
                "hits_3": hits_3,
                "hits_1": hits_1,
                "mean_rank": mean_rank,
                "mean reciprocal rank": mrr,
            }
        )

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_indices(
            data.test_data, data.entity_idxs, data.relation_idxs
        )
        sr_vocab = self.get_er_vocab(test_data_idxs)

        print("Number of data points: %d" % len(test_data_idxs))

        batch_pbar = tqdm(
            range(0, len(test_data_idxs), self.batch_size), position=0, leave=True
        )
        for i in batch_pbar:
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_s = model.forward(
                e1_idx.repeat(len(data.entities)),
                r_idx.repeat(len(data.entities)),
                range(len(data.entities)),
            )

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        self.log_eval(hits, ranks)

    def train_and_eval(self, data: Data):
        print("Training the %s model..." % self.model)
        # self.entity_idxs = {data.entities[i]: i for i in range(len(data.entities))}
        # self.relation_idxs = {data.relations[i]: i for i in range(len(data.relations))}

        train_data_idxs = self.get_data_indices(
            data.train_data, data.entity_idxs, data.relation_idxs
        )
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model == "poincare":
            model = MuRP(data, self.dim)
        else:
            model = MuRE(data, self.dim)

        self.load_model(model)
        wandb.watch(model, log_freq=100)

        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(
            model.parameters(), lr=self.learning_rate, param_names=param_names
        )

        if self.cuda:
            model.cuda()

        # er_vocab = self.get_er_vocab(train_data_idxs)

        print("Starting training...")
        for epoch in range(1, self.num_iterations + 1):
            model.train()

            losses = []
            if isinstance(train_data_idxs, pd.DataFrame):
                train_data_idxs = train_data_idxs.sample(frac=1).reset_index(drop=True)
            else:
                np.random.shuffle(train_data_idxs)
            batch_pbar = tqdm(
                range(0, len(train_data_idxs), self.batch_size), position=0, leave=True
            )
            num_batches = len(batch_pbar)
            for batch_idx, j in enumerate(batch_pbar):
                data_batch = np.array(train_data_idxs[j : j + self.batch_size])
                negsamples = np.random.choice(
                    list(data.entity_idxs.values()),
                    size=(data_batch.shape[0], self.nneg),
                )

                e1_idx = torch.tensor(
                    np.tile(
                        np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1] + 1)
                    )
                )
                r_idx = torch.tensor(
                    np.tile(
                        np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1] + 1)
                    )
                )
                e2_idx = torch.tensor(
                    np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1)
                )

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                batch_pbar.set_description(
                    f"Epoch {epoch}, loss: {np.mean(losses[-50:]):.5f}"
                )

                if batch_idx % self.args.log_interval == 0:
                    wandb.log({"train loss": loss})

                if self.args.eval_per_epoch > 1 and (
                    batch_idx % int(num_batches / self.args.eval_per_epoch) == 0
                ):
                    model.eval()
                    with torch.no_grad():
                        self.evaluate(model, data)

            if self.args.eval_per_epoch <= 1 and (
                epoch % int(1 / self.args.eval_per_epoch) == 0
            ):
                model.eval()
                with torch.no_grad():
                    self.evaluate(model, data)

        self.save_model(model)

    def embed(self, data: Data, embed_file_paths: str):
        if self.model == "poincare":
            model = MuRP(data, self.dim)
        else:
            model = MuRE(data, self.dim)
        self.load_model(model)
        model.eval()

        for embed_file_path in embed_file_paths:
            with open(embed_file_path, "r") as f:
                items = [l.strip() for l in f.readlines()]

            data_indices = []
            not_found_cnt = 0
            for item in items:
                if embed_file_path.endswith(".rel"):
                    data_indices.append(data.relation_idxs.get(item, -1))
                else:
                    data_indices.append(data.entity_idxs.get(item, -1))
                if data_indices[-1] == -1:
                    print(f"Item: {item} not found in vocab")
                    not_found_cnt += 1
            print(
                f"In total {not_found_cnt} ({not_found_cnt/len(items)}) items are not found in vocab."
            )

            # relation embedding include both Wu and Rv/Rvh
            emb = (
                torch.zeros((len(items), self.dim, 2))
                if embed_file_path.endswith(".rel")
                else torch.zeros((len(items), self.dim))
            )
            with torch.no_grad():
                print("embedding data ...")
                batch_pbar = tqdm(
                    range(0, len(data_indices), self.batch_size), position=0, leave=True
                )
                for batch_idx, j in enumerate(batch_pbar):
                    data_batch = torch.tensor(data_indices[j : j + self.batch_size])
                    if self.cuda:
                        data_batch = data_batch.cuda()
                    if embed_file_path.endswith(".rel"):
                        embeddings = model.embed(r_idx=data_batch)
                        emb[j : j + self.batch_size] = embeddings * (
                            (data_batch != -1)
                            .unsqueeze(1)
                            .unsqueeze(2)
                            .repeat(1, embeddings.shape[1], embeddings.shape[2])
                        )
                    else:
                        embeddings = model.embed(u_idx=data_batch)
                        emb[j : j + self.batch_size] = embeddings * (
                            (data_batch != -1)
                            .unsqueeze(1)
                            .repeat(1, embeddings.shape[1])
                        )

            torch.save(
                emb,
                self.model_save_path.replace(
                    "model.pt", embed_file_path.split("/")[-1] + ".pt"
                ),
            )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/WN18RR/",
        nargs="?",
        help="Which dataset to use: data/FB15k-237/ or data/WN18RR/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="poincare",
        nargs="?",
        help="Which model to use: poincare or euclidean.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./dumps",
        nargs="?",
        help="path to save the trained model.",
    )
    parser.add_argument(
        "--model_reload_path",
        type=Optional[str],
        default=None,
        nargs="?",
        help="If not None, load the model",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500,
        nargs="?",
        help="Number of iterations.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, nargs="?", help="Batch size."
    )
    parser.add_argument(
        "--nneg", type=int, default=50, nargs="?", help="Number of negative samples."
    )
    parser.add_argument(
        "--lr", type=float, default=50, nargs="?", help="Learning rate."
    )
    parser.add_argument(
        "--dim", type=int, default=40, nargs="?", help="Embedding dimensionality."
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True,
        nargs="?",
        help="Whether to use cuda (GPU) or not (CPU).",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        nargs="?",
        help="wandb log interval (in batches)",
    )
    parser.add_argument(
        "--eval_per_epoch",
        type=float,
        default=100,
        nargs="?",
        help="how many times the model is evaluated during epoch. If less than 1, evaluates once every few epochs",
    )

    parser.add_argument(
        "--embed_only",
        type=bool,
        default=False,
        nargs="?",
        help="Whether to only provide embedding prediction.",
    )
    parser.add_argument(
        "--embed_files",
        type=Optional[str],
        default=None,
        nargs="?",
        help="Whether to predict embedding for the given list of entities / relations. If entity file, the file should"
        "end with .ent, if relation, the file should end of .rel",
    )

    args = parser.parse_args()
    if args.embed_files is not None:
        args.embed_files = args.embed_files.split(",")
    return args


def main(args):
    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=args.data_dir)
    experiment = Experiment(args)
    if not args.embed_only:
        wandb.init(config=args)
        wandb.run.name = f"{args.model}_{'_'.join(args.data_dir.split('/')[1:])}"
        experiment.train_and_eval(d)
    if args.embed_files:
        experiment.embed(d, embed_file_paths=args.embed_files)


if __name__ == "__main__":
    args = get_parser()
    main(args)
