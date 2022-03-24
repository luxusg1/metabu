from metabu.metabu import Metabu

# from metabu import utils
# from metabu.fused_gromov_wasserstein import *

# from metabu.prae import fgw

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--basic_representation_file', '-b',
#                         default='examples/data/basic_representations.csv',
#                         help='basic representation file')
#
#     parser.add_argument('--target_representation_file', '-t',
#                         required=False,
#                         default='examples/data/target_representation_adaboost.csv',
#                         help='target representation file')
#
#     parser.add_argument('--ranking_column_name',
#                         required=True,
#                         help='the name of columns to rank the target representation in the target representation file')
#
#     parser.add_argument('--store',
#                         default="examples/output",
#                         help='Path to store')
#
#     parser.add_argument('--test_ids', nargs='+',
#                         help='list of task id to be removed when train metabu_representation',
#                         required=True)
#
#     args = parser.parse_args()
#
#     # get basic representation
#     basic_representation = pd.read_csv(args.basic_representation_file)
#     basic_representation = basic_representation.fillna(0)
#
#     # get target representation
#     target_representation = pd.read_csv(args.target_representation_file)
#
#     # train metabu representation
#     test_ids = [int(task_id) for task_id in args.test_ids]
#     train_ids = [task_id for task_id in list(target_representation.task_id.unique()) if task_id not in test_ids]
#     print(train_ids)
#     _, metabu_representation_train, metabu_representation_test = train(basic_representation=basic_representation,
#                                                                        target_representation=target_representation,
#                                                                        test_ids=test_ids,
#                                                                        train_ids=train_ids,
#                                                                        ranking_column_name=args.ranking_column_name)
#
#     # store metabu mf
#     metabu_representation_test = pd.DataFrame(metabu_representation_test)
#     metabu_representation_test['task_id'] = args.test_ids
#     metabu_representation_test.to_csv(os.path.join(args.store, 'metabu_representation.csv'), index=False)
