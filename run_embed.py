from types import SimpleNamespace
from main import main, get_parser


dataset_dir = "data/wikidata5m_inductive/"

# subsampled 0.1 en
args = get_parser()
# args.data_dir = "data/dbpedia/en/"
args.data_dir = dataset_dir
# args.model_reload_path = "dumps/dbpedia/en/euclidean_model.pt"
# args.model_reload_path = "dumps/dbpedia/en/poincare_model.pt"
args.model_reload_path = "dumps/wikidata5m_inductive/poincare_model.pt"
args.model = "poincare"
# args.model = "euclidean"
args.batch_size = 48
args.dim = 40
args.cuda = True
args.embed_only = True
args.embed_files = [
    # "data/dbpedia/en/person_with_gender_.ent",
    f"{dataset_dir}/professions.ent",
    f"{dataset_dir}/relations.rel",
    f"{dataset_dir}/gender.ent",
    f"{dataset_dir}/humans.ent",
]

main(args)
