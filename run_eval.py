from types import SimpleNamespace
from main import main, get_parser

for lg in ["en", "id", "sv"]:
    for model in ["poincare", "euclidean"]:
        args = get_parser()
        args.data_dir = f"data/dbpedia/{lg}/"
        # args.model_reload_path = "dumps/dbpedia/en/euclidean_model.pt"
        args.model_reload_path = f"dumps/dbpedia/{lg}/{model}_model.pt"
        args.model = model
        args.batch_size = 2048
        args.dim = 40
        args.cuda = True
        args.eval_only = True

        main(args)
