from types import SimpleNamespace
from main import main, get_parser


### KY KG, 30K train
# args = {
#     "data_dir": "data/dbpedia/ky/",
#     "model": "poincare",
#     "num_iterations": 30,
#     "nneg": 50,
#     "batch_size": 128,
#     "lr": 50,
#     "dim": 40,
#     "cuda": True,
#     "model_save_path": "./poincare_model.pt",
#     "model_reload_path": None,
#     "log_interval": 1000,
#     "eval_per_epoch": 0.2
# }

### ZH KG, 2.2M train
# args = {
#     "data_dir": "data/dbpedia/zh/",
#     "model": "poincare",
#     "num_iterations": 170,
#     "nneg": 50,
#     "batch_size": 8192,
#     "lr": 50,
#     "dim": 40,
#     "cuda": True,
#     "model_save_path": "./dumps",
#     "model_reload_path": None,
#     "log_interval": 100,  # eval per # epoch
#     "eval_per_epoch": 1,
# }

### subsampled 0.1 en , 2.2M train
args = get_parser()
args.data_dir = f"data/dbpedia/en/"
args.model = "poincare"
args.num_iterations = 20
args.nneg = 50
args.batch_size = 2048
args.lr = 50
args.dim = 40
args.cuda = True
args.log_interval = 400
args.eval_per_epoch = 1

main(args)
