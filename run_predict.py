from types import SimpleNamespace
from main import main, get_parser


dataset_dir = "data/wikidata5m_transductive"

# subsampled 0.1 en
args = get_parser()
# args.data_dir = "data/dbpedia/en/"
args.data_dir = dataset_dir
# args.model_reload_path = "dumps/wikidata5m_transductive/poincare_model.pt"
args.model_reload_path = "dumps/wikidata5m_transductive/euclidean_model.pt"

# args.model = "poincare"
args.model = "euclidean"
args.batch_size = 48
args.lr = 10000
args.dim = 40
args.cuda = True
args.bias_check_only = True
args.person_file = f"{dataset_dir}/humans.ent"
args.person_subsample_frac = 0.01
args.sensitive_file = f"{dataset_dir}/gender_small.ent"
args.profession_file = f"{dataset_dir}/professions.ent"

main(args)
