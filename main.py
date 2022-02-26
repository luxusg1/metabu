import pandas as pd
import metabu
import argparse
import os
'''
metafeatures and target are dataframe object and must have task_id columns

'''


def train_metabu(metafeatures, target_representation, learning_rate, top_k, alpha):
    normalized_mf = metabu.utils.normalize_metafeatures(metafeatures)
    top_k_target = metabu.utils.get_top_k_target(target_representation, target_representation.task_id.unique(), top_k)
    preprocessor = metabu.get_target_preprocessor(target_representation)
    task_id_for_train = set(top_k_target.task_id) - set(metafeatures.index)

    # get only available mf and target representation
    for i in list(task_id_for_train):
        top_k_target = top_k_target[top_k_target.task_id != i]

    model, metabu_mf = metabu.get_learned_metafeatures(datasets_has_priors_use_for_train=top_k_target.task_id.unique(),
                                                       train_metafeatures=normalized_mf, top_k_dataset=top_k_target,
                                                       prior_preprocessor=preprocessor, learning_rate=learning_rate,
                                                       alpha=alpha, seed=1)
    return metabu_mf


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--metafeatures_file',
                        default='examples/data/mfe.csv',
                        help='input metafeatures')

    parser.add_argument('--target_representation_file',
                        required=False,
                        default='examples/data/adaboost.csv',
                        help='target representation file')

    parser.add_argument('--store',
                        default="examples/output",
                        help='Path to store')

    parser.add_argument('--learning_rate',
                        default=0.001,
                        help='learning rate')

    parser.add_argument('--alpha',
                        default=0.5,
                        help='lambda')

    parser.add_argument('--top_k',
                        default=20,
                        help='top datasets used to compute matrix distance')

    args = parser.parse_args()

    # get metafeatures
    metafeatures = pd.read_csv(args.metafeatures_file).set_index('task_id')

    # get target representation
    target_representation = pd.read_csv(args.target_representation_file)

    train_mf = train_metabu(metafeatures=metafeatures,
                            target_representation=target_representation,
                            learning_rate=args.learning_rate,
                            top_k=args.top_k,
                            alpha=args.alpha)

    pd.DataFrame(train_mf).to_csv(os.path.join(args.store, 'metabu_mf.csv'), index=False)
