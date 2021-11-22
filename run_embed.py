from types import SimpleNamespace
from main import main, get_parser

# subsampled 0.1 en
args = get_parser()
args.data_dir = "data/dbpedia/en/"
args.model_reload_path = "dumps/dbpedia/en/poincare_model.pt"
args.model = "poincare"
args.batch_size = 48
args.dim = 40
args.cuda = False
args.embed_only = True
args.embed_files = [
    "data/dbpedia/en/person_with_gender.ent",
    "data/dbpedia/en/professions.ent",
]

main(args)
