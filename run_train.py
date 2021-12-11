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

## Indonesian, 2.2M
# args = get_parser()
# args.data_dir = f"data/dbpedia/id/"
# args.model = "poincare"
# # args.model = "euclidean"
# args.num_iterations = 500
# args.nneg = 200
# args.batch_size = 4096
# args.eval_batch_size = 524288
# args.lr = 500
# args.dim = 40
# args.cuda = True
# args.log_interval = 25
# args.eval_per_epoch = 0.1
# # args.eval_only = True
# args.partial_eval = 0.01
#
# main(args)

## swedish, 1.7M
# args = get_parser()
# args.data_dir = f"data/dbpedia/sv/"
# # args.model = "poincare"
# args.model = "euclidean"
# args.num_iterations = 500
# args.nneg = 200
# # args.batch_size = 4096
# args.batch_size = 8192
# args.eval_batch_size = 524288
# args.lr = 200
# args.dim = 40
# args.cuda = True
# args.log_interval = 25
# args.eval_per_epoch = 0.1
# # args.eval_only = True
# args.partial_eval = 0.005
#
# main(args)


## english dbpedia
# args = get_parser()
# args.data_dir = f"data/dbpedia/en/"
# # args.model = "euclidean"
# # args.lr = 200
# # args.batch_size = 8192
# args.model = "poincare"
# args.lr = 100
# args.batch_size = 2048
# args.nneg = 200
# args.num_iterations = 500
# args.eval_batch_size = 524288
# args.dim = 40
# args.cuda = True
# args.log_interval = 25
# args.eval_per_epoch = 0.1
# # args.eval_only = True
# args.partial_eval = 0.005
#
# main(args)

### subsampled 0.1 en , 2.2M train
# args = get_parser()
# # args.data_dir = f"data/dbpedia/en/"
# args.data_dir = f"data/wikidata5m_transductive"
# # args.model = "poincare"
# args.model = "euclidean"
# args.num_iterations = 20
# args.nneg = 50
# args.batch_size = 8192
# args.eval_batch_size = 524288
# args.lr = 50
# args.dim = 40
# args.cuda = True
# args.log_interval = 25
# args.eval_per_epoch = 1
# # args.eval_only = True
# args.partial_eval = 0.05
#
# main(args)

# FB15K
args = get_parser()
# args.data_dir = f"data/dbpedia/en/"
args.data_dir = f"data/FB15k-237"
# args.model = "poincare"
args.model = "euclidean"
args.num_iterations = 500
args.nneg = 200
args.batch_size = 8192
args.eval_batch_size = 524288
args.lr = 200
args.dim = 40
args.cuda = True
args.log_interval = 25
args.eval_per_epoch = 0.1
# args.eval_only = True
# args.partial_eval = 0.05

main(args)
