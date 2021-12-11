import random
from typing import Optional, List, Union
import os
import json
import numpy as np
import pandas as pd
import torch
import wandb
import time
from collections import defaultdict

from tqdm import tqdm

from data_exploration import (
    append_wikidata_alias,
    append_wikidata_equivalent,
    append_FB15k_train_count,
    GENDER_RELATION,
    PROFESSION_RELATION,
    append_dbpedia_train_count,
)
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
        self.eval_batch_size = args.eval_batch_size
        self.cuda = args.cuda
        self.model_save_path = f"{args.model_save_path}/{'/'.join(args.data_dir.split('/')[1:])}/{self.model}_model.pt"
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
        metrics = {
            "hits_10": hits_10,
            "hits_3": hits_3,
            "hits_1": hits_1,
            "mean_rank": mean_rank,
            "mean reciprocal rank": mrr,
        }
        if not self.args.eval_only:
            wandb.log(metrics)

        with open(self.model_save_path.replace("_model.pt", "_eval.json"), "w") as f:
            json.dump(metrics, f)

    def evaluate(self, model, data, partial_eval=1.0):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_indices(
            data.test_data, data.entity_idxs, data.relation_idxs
        )
        if partial_eval != 1.0:
            test_data_idxs = random.sample(
                test_data_idxs, k=int(len(test_data_idxs) * partial_eval)
            )
        sr_vocab = self.get_er_vocab(test_data_idxs)

        print("Number of data points: %d" % len(test_data_idxs))

        batch_pbar = tqdm(range(0, len(test_data_idxs)), position=0, leave=True)
        for i in batch_pbar:
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            if len(data.entities) < self.eval_batch_size:
                predictions_s = model.forward(
                    e1_idx.repeat(len(data.entities)),
                    r_idx.repeat(len(data.entities)),
                    range(len(data.entities)),
                )
            else:
                predictions_s = torch.zeros(len(data.entities))
                for b in range(0, len(data.entities), self.eval_batch_size):
                    b_size = min(self.eval_batch_size, len(data.entities) - b)
                    predictions_s[b : b + b_size] = model.forward(
                        e1_idx.repeat(b_size),
                        r_idx.repeat(b_size),
                        range(b, b + b_size),
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

        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(
            model.parameters(), lr=self.learning_rate, param_names=param_names
        )

        if self.cuda:
            model.cuda()

        if self.args.eval_only:
            model.eval()
            with torch.no_grad():
                self.evaluate(model, data, partial_eval=self.args.partial_eval)
            exit()

        # er_vocab = self.get_er_vocab(train_data_idxs)
        wandb.watch(model, log_freq=100)
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
                    wandb.log({"epoch": epoch, "train loss": loss})

                if self.args.eval_per_epoch > 1 and (
                    batch_idx % int(num_batches / self.args.eval_per_epoch) == 0
                ):
                    model.eval()
                    with torch.no_grad():
                        self.evaluate(model, data, partial_eval=self.args.partial_eval)

            if self.args.eval_per_epoch <= 1 and (
                epoch % int(1 / self.args.eval_per_epoch) == 0
            ):
                model.eval()
                with torch.no_grad():
                    self.evaluate(model, data, partial_eval=self.args.partial_eval)

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

    def bias_prediction(
        self,
        data: Data,
        person_file_paths: str,
        person_subsample_frac: float,
        sensitive_relation: str,
        sensitive_file_path: str,
        sensitive_in_order: bool,
        profession_relation: str,
        profession_file_path: str,
    ) -> np.array:
        """
        Alexa Paper bias update. Update person entity with a triple to either side of sensitive attribute,
        then calculate
        :param data:
        :param person_file_paths:
        :return:
        """
        if self.model == "poincare":
            model = MuRP(data, self.dim)
        else:
            model = MuRE(data, self.dim)
        self.load_model(model)
        model.train()

        s_rel_idx = data.relation_idxs[sensitive_relation]
        p_rel_idx = data.relation_idxs[profession_relation]

        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(
            model.parameters(), lr=self.learning_rate, param_names=param_names
        )
        with open(person_file_paths, "r") as f:
            items = [l.strip() for l in f.readlines()]
        with open(profession_file_path, "r") as f:
            p_entities = [
                l.strip() for l in f.readlines() if l.strip() in data.entity_idxs
            ]
            p_ent_indices = [data.entity_idxs[e] for e in p_entities]
        with open(sensitive_file_path, "r") as f:
            s_ent_indices = [
                data.entity_idxs[l.strip()]
                for l in f.readlines()
                if l.strip() in data.entity_idxs
            ]

        training_triples = []
        not_found_cnt = 0
        exist_entities = []
        for item in items:
            if item not in data.entity_idxs:
                # print(f"Item: {item} not found in vocab")
                not_found_cnt += 1
            else:
                exist_entities.append(item)

        exist_entities = random.choices(
            exist_entities, k=int(len(exist_entities) * person_subsample_frac)
        )
        print(
            f"Subsampled {person_subsample_frac} from existing entities: {len(exist_entities)} people entity left"
        )
        for person in exist_entities:
            for s_ent in s_ent_indices:
                training_triples.append([data.entity_idxs[person], s_rel_idx, s_ent])

        with open(person_file_paths.replace(".ent", ".exist.ent"), "w") as f:
            f.writelines([f"{e}\n" for e in exist_entities])
        print(
            f"In total {not_found_cnt} ({not_found_cnt/len(items)}) items are not found in vocab."
        )

        profession_scores_person = np.zeros([len(exist_entities), len(p_ent_indices)])
        batch_pbar = tqdm(
            range(0, len(training_triples), self.batch_size), position=0, leave=True
        )
        for batch_idx, j in enumerate(batch_pbar):
            # calculate profession scores
            data_batch = np.array(training_triples[j : j + self.batch_size])
            pre_scores = self.get_profession_scores(
                model, data_batch, p_rel_idx, p_ent_indices, len(s_ent_indices)
            )

            e1_idx = torch.tensor(data_batch[:, 0])
            r_idx = torch.tensor(data_batch[:, 1])
            e2_idx = torch.tensor(data_batch[:, 2])

            targets = np.ones(e1_idx.shape)
            ## Uncomment below line for second method of gradient calculation
            # targets[list(range(1, len(e1_idx), 2))] = 0.0
            targets = torch.DoubleTensor(targets)

            opt.zero_grad()
            if self.cuda:
                e1_idx, r_idx, e2_idx, targets = to_cuda(e1_idx, r_idx, e2_idx, targets)

            #### Method 1 of calculating bias, subtract female gradient from male
            #### and update one person at a time, for this to run, use batch_size=2
            #### and gender.ent file contain male and female entity in that order
            for i in range(e1_idx.shape[0]):
                predictions = model.forward(e1_idx[i], r_idx[i], e2_idx[i])
                loss = model.loss(predictions, targets[i])
                loss.backward()
                if i == 0:
                    if self.model == "euclidean":
                        male_grad = model.E.weight.grad[e1_idx[i]].clone()
                    else:
                        male_grad = model.Eh.weight.grad[e1_idx[i]].clone()
                    opt.zero_grad()
                else:
                    if self.model == "euclidean":
                        female_grad = model.E.weight.grad[e1_idx[i]]
                        grad = (
                            male_grad - female_grad
                            if sensitive_in_order
                            else female_grad - male_grad
                        )
                        model.E.weight.requires_grad = False
                        model.E.weight[e1_idx[i]] = euclidean_update(
                            model.E.weight[e1_idx[i]], grad.data, 0.01
                        )
                        model.E.weight.requires_grad = True
                    else:
                        # Reimannian SGD
                        female_grad = model.Eh.weight.grad[e1_idx[i]]
                        grad = (
                            male_grad - female_grad
                            if sensitive_in_order
                            else female_grad - male_grad
                        )
                        model.Eh.weight.requires_grad = False
                        d_p = poincare_grad(model.Eh.weight[e1_idx[i]], grad.data)
                        model.Eh.weight[e1_idx[i]] = poincare_update(
                            model.Eh.weight[e1_idx[i]], d_p, 0.01
                        )
                        model.Eh.weight.requires_grad = True

            #### Second method of calculation, setting male target to 1 and female target to 0. This is much faster
            #### because batch size does not need to be one, and can be generalized to multiple sensitvie attributes.
            # predictions = model.forward(e1_idx, r_idx, e2_idx)
            # loss = model.loss(predictions, targets)
            # loss.backward()
            # opt.step()
            post_scores = self.get_profession_scores(
                model, data_batch, p_rel_idx, p_ent_indices, len(s_ent_indices)
            )
            profession_scores_person[
                int(j / len(s_ent_indices)) : int(
                    (j + self.batch_size) / len(s_ent_indices)
                )
            ] = (post_scores - pre_scores)

            # reload original model because we don't want to change the weights between batches
            self.load_model(model)

        profession_scores_person = profession_scores_person.mean(axis=0)
        df = pd.DataFrame(
            {"profession": p_entities, "scores": profession_scores_person}
        )
        df = df.sort_values(by=["scores"], ascending=False)
        if "FB15k" in self.args.data_dir:
            df = append_wikidata_equivalent(df)
        if "FB15K" in self.args.data_dir or "wikidata" in self.args.data_dir:
            df = append_wikidata_alias(df)
        if "FB15k" in self.args.data_dir:
            df = append_FB15k_train_count(df)
            df = df.drop(columns=["profession"])

        if "dbpedia" in self.args.data_dir:
            df = append_dbpedia_train_count(df, self.args.data_dir.split("/")[-1])

        df["scores"] = df["scores"].round(8)
        df = df.fillna(0)
        df["count"] = df["count"].astype(int)
        if "male_count" in df.columns:
            df["male_count"] = df["male_count"].astype(int)
        if "female_count" in df.columns:
            df["female_count"] = df["female_count"].astype(int)
        df = df.rename(
            columns={
                "scores": "Scores",
                "profession_name": "Profession",
                "count": "Count",
                "male_count": "Male Count",
                "female_count": "Female Count",
            }
        )
        df.to_csv(
            f"{self.args.data_dir}/{self.model}_profession_name_bias_{person_subsample_frac}_{sensitive_file_path.split('/')[-1].replace('.ent', '')}{'_reverse' if not sensitive_in_order else ''}.csv",
            index=True,
        )

    def get_profession_scores(
        self,
        model,
        batch_idx,
        p_rel_idx,
        p_ent_indices,
        num_sensitive_attributes: int = 2,
    ) -> np.array:
        with torch.no_grad():
            bs = batch_idx.shape[0]
            total_scores = np.zeros(
                [int(bs / num_sensitive_attributes), len(p_ent_indices)]
            )
            for i in range(0, bs - 1, num_sensitive_attributes):
                p_triples = torch.tensor(
                    [[batch_idx[i][0], p_rel_idx, p] for p in p_ent_indices]
                )
                e1_idx = p_triples[:, 0].clone()
                r_idx = p_triples[:, 1].clone()
                e2_idx = p_triples[:, 2].clone()
                if self.cuda:
                    e1_idx, r_idx, e2_idx = to_cuda(e1_idx, r_idx, e2_idx)
                predictions = (
                    model.forward(e1_idx, r_idx, e2_idx).detach().cpu().numpy()
                )  # scores = predictions[np.arange(len(p_triples)), p_ent_indices]
                total_scores[int(i / 2)] = predictions
        return total_scores


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
        "--eval_batch_size",
        type=int,
        default=128,
        nargs="?",
        help="Batch size during eval.",
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
        "--partial_eval",
        type=float,
        default=1.0,
        nargs="?",
        help="what fraction of evaluation data is evaluated",
    )
    parser.add_argument(
        "--eval_only",
        type=bool,
        default=False,
        nargs="?",
        help="Whether to only evaluate.",
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

    parser.add_argument(
        "--bias_check_only",
        type=bool,
        default=False,
        nargs="?",
        help="Whether to only provide embedding prediction.",
    )
    parser.add_argument(
        "--person_file",
        type=Optional[str],
        default=None,
        nargs="?",
        help="person entity file",
    )
    parser.add_argument(
        "--person_subsample_frac",
        type=float,
        default=1.0,
        nargs="?",
        help="fraction of people to select",
    )
    parser.add_argument(
        "--sensitive_file",
        type=Optional[str],
        default=None,
        nargs="?",
        help="sensitive entity file",
    )
    parser.add_argument(
        "--sensitive_in_order",
        type=bool,
        default=True,
        nargs="?",
        help="Whether to only provide embedding prediction.",
    )
    parser.add_argument(
        "--profession_file",
        type=Optional[str],
        default=None,
        nargs="?",
        help="profession entity file",
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
    if args.bias_check_only:
        if "wikidata" in args.data_dir:
            sensitive_rel, occupation_rel = "P21", "P106"
        elif "FB15K" in args.data_dir:
            sensitive_rel, occupation_rel = (
                "/people/person/gender",
                "/people/person/profession",
            )
        else:  # dbpedia
            language = args.data_dir.split("/")[-1]
            sensitive_rel, occupation_rel = (
                GENDER_RELATION[language],
                PROFESSION_RELATION[language][0],
            )
        experiment.bias_prediction(
            d,
            args.person_file,
            args.person_subsample_frac,
            sensitive_rel,
            args.sensitive_file,
            args.sensitive_in_order,
            occupation_rel,
            args.profession_file,
        )
        exit()
    if not args.embed_only:
        if not args.eval_only:
            wandb.init(config=args)
            wandb.run.name = f"{args.model}_{'_'.join(args.data_dir.split('/')[1:])}"
        experiment.train_and_eval(d)
    if args.embed_files:
        experiment.embed(d, embed_file_paths=args.embed_files)


if __name__ == "__main__":
    args = get_parser()
    main(args)
