import pandas as pd
import metabu
import argparse
import os

'''
metafeatures and target are dataframe object and must have task_id columns

'''


def train_metabu(metafeatures, target_representation, learning_rate, top_k, alpha, ranking_column_name):
    top_k_target = metabu.get_top_k_target(target_representation, target_representation.task_id.unique(), top_k,
                                           ranking_column_name)

    to_remove_datasets = set(list(top_k_target.task_id.unique())) - set(list(metafeatures.task_id.unique()))
    for id in list(to_remove_datasets):
        top_k_target = top_k_target[top_k_target.task_id != id]

    mfe = metafeatures.set_index('task_id')
    datasets_has_priors_use_for_train = top_k_target.task_id.unique()
    model, metabu_mf = metabu.get_learned_metafeatures(datasets_has_priors_use_for_train=datasets_has_priors_use_for_train,
                                                       train_metafeatures=mfe, top_k_dataset=top_k_target,
                                                       learning_rate=0.001,
                                                       alpha=0.5,
                                                       ranking_column_name=ranking_column_name)

    metabu_mf = pd.DataFrame(metabu_mf)
    metabu_mf['task_id'] = datasets_has_priors_use_for_train

    return metabu_mf


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--metafeatures_file',
                        default='examples/data/mfe.csv',
                        help='input metafeatures')

    parser.add_argument('--target_representation_file',
                        required=False,
                        default='examples/data/adaboost_preprocessed.csv',
                        help='target representation file')

    parser.add_argument('--store',
                        default="examples/output",
                        help='Path to store')

    parser.add_argument('--learning_rate',
                        default=0.001,
                        help='learning rate')

    parser.add_argument('--alpha',
                        default=0.5,
                        help='alpha')

    parser.add_argument('--top_k',
                        default=20,
                        help='top datasets used to compute matrix distance')

    parser.add_argument('--ranking_column_name',
                        required=True,
                        help='the name of columns to rank the target representation in the target representation file')

    args = parser.parse_args()

    # get metafeatures
    metafeatures = pd.read_csv(args.metafeatures_file)

    # get target representation
    target_representation = pd.read_csv(args.target_representation_file)

    train_mf = train_metabu(metafeatures=metafeatures,
                            target_representation=target_representation,
                            learning_rate=args.learning_rate,
                            top_k=args.top_k,
                            alpha=args.alpha,
                            ranking_column_name=args.ranking_column_name)

    # store metabu mf
    train_mf.to_csv(os.path.join(args.store, 'metabu_mf.csv'), index=False)
