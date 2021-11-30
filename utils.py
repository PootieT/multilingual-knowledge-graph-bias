import torch


def artanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv) * v / normv


def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1 - 1e-5)
    return artanh(normv) * v / normv


def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5)
    y = torch.tanh(normv / (1 - sqxnorm)) * v / normv
    return p_sum(x, y)


def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1 - 1e-5)
    dotxy = torch.sum(x * y, dim=-1, keepdim=True)
    numerator = (1 + 2 * dotxy + sqynorm) * x + (1 - sqxnorm) * y
    denominator = 1 + 2 * dotxy + sqxnorm * sqynorm
    return numerator / denominator


def convert_tsv_to_ttl(file_path: str):
    with open(file_path, "r") as f:
        lines = [l.replace('"\n', ">\n") for l in f.readlines()]
        lines = [l.replace('"', "<") for l in lines]
        with open(file_path.replace("tsv", "txt"), "w") as f:
            f.write("".join(lines))


# convert_tsv_to_ttl("data/dbpedia/en/person_with_gender.tsv")
def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]
