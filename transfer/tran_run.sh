mode=$1
gpu=$2
if [ "$mode" = "pretrain" ]; then
    echo =============
    echo ">>>>  pretrain in ZINC" 
    python chem_pretrain.py --dataset_root=dataset --dataset=zinc_standard_agent --seed=123 --save=zinc --epoch=20 --device=$gpu
elif [ "$mode" = "finetune" ]; then
    echo =============
    echo ">>>>  finetune in MoleculeNet"
    datasets=("bbbp" "tox21" "toxcast" "sider" "clintox" "muv" "hiv" "bace")
    for dataset in ${datasets[*]}
    do
    for seed in {0..9}
    do
    python chem_finetune.py --dataset=${dataset} --seed=${seed} --cl_exp_dir=transfer_exp/chembl_filtered --cl_model_name=cl_model.pth --save=semi_finetune --epoch=100 --device=$gpu
    done
    done
fi