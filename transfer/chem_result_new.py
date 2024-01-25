import pandas as pd
import time

result_path = './results.txt'
with open(result_path, 'a+') as f:
        f.write('{}\n'.format(time.strftime("%Y-%m-%d %H:%M:%S")))

result_log_path_list = []
result_log_path_list.append('./transfer_exp/chem_new/chembl_filtered10/fine_tune_result.txt')
result_log_path_list.append('./transfer_exp/chem/search/fine_tune_result.txt')
for i in [30,40,50,60,70,80,90,100]:
    result_log_path_list.append('./transfer_exp/chem_new/chembl_filtered{}/fine_tune_result.txt'.format(i))

result_list = []
for result_log_path in result_log_path_list:
    result_list.append(pd.read_csv(result_log_path, sep=' '))

datasets = ["bbbp", "tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bace"]

for result in result_list:
    for dataset in datasets:
        now_result = result.loc[result['dataset'] == dataset]
        mean = now_result['test_acc'].mean()
        std = now_result['test_acc'].std()
        with open(result_path, 'a+') as f:
            f.write('{} {:.2f} ± {:.2f}\t'.format(dataset, mean*100, std*100))
    with open(result_path, 'a+') as f:
        f.write('\n')

with open(result_path, 'a+') as f:
        f.write('\n')
