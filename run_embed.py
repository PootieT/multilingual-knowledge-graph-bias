from types import SimpleNamespace
from main import main, get_parser

# subsampled 0.1 en
args = get_parser()
args.data_dir = "data/dbpedia/en/"
# args.model_reload_path = "dumps/dbpedia/en/euclidean_model.pt"
args.model_reload_path = "dumps/dbpedia/en/poincare_model.pt"
args.model = "poincare"
args.batch_size = 48
args.dim = 40
args.cuda = True
args.embed_only = True
args.embed_files = [
    "data/dbpedia/en/person_with_gender_.ent",
    "data/dbpedia/en/professions.ent",
    "data/dbpedia/en/relations.rel",
    "data/dbpedia/en/gender.ent",
]

main(args)
