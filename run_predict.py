from types import SimpleNamespace
from main import main, get_parser


# subsampled 0.1 en
args = get_parser()

# dataset_dir = "data/wikidata5m_transductive"
# args.model_reload_path = "dumps/wikidata5m_transductive/poincare_model.pt"
# args.model_reload_path = "dumps/wikidata5m_transductive/euclidean_model.pt"

# dataset_dir = "data/FB15k-237"
# args.model_reload_path = "dumps/FB15k-237/euclidean_model.pt"
# args.model_reload_path = "dumps/FB15k-237/poincare_model.pt"
# args.person_subsample_frac = 1.0

dataset_dir = "data/dbpedia/id"
# args.model_reload_path = "dumps/dbpedia/id/poincare_model.pt"
args.model_reload_path = "dumps/dbpedia/id/euclidean_model.pt"
args.person_subsample_frac = 0.1

# dataset_dir = "data/dbpedia/sv"
# args.model_reload_path = "dumps/dbpedia/sv/poincare_model.pt"
# args.model_reload_path = "dumps/dbpedia/sv/euclidean_model.pt"
# args.person_subsample_frac = 1.0

# dataset_dir = "data/dbpedia/en"
# args.model_reload_path = "dumps/dbpedia/en/poincare_model.pt"
# args.model_reload_path = "dumps/dbpedia/en/euclidean_model.pt"
# args.person_subsample_frac = 1.0


args.data_dir = dataset_dir
# args.model = "poincare"
args.model = "euclidean"
args.batch_size = 2
args.lr = 0.01
args.dim = 40
args.cuda = True
args.bias_check_only = True
args.person_file = f"{dataset_dir}/humans.ent"
args.sensitive_file = f"{dataset_dir}/gender.ent"
args.sensitive_in_order = False
args.profession_file = f"{dataset_dir}/professions.ent"
main(args)
