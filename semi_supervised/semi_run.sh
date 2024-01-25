mode=$1
gpu=$2
dataset_list=(PROTEINS DD NCI1 COLLAB github_stargazers REDDIT-BINARY REDDIT-MULTI-5K)
save=semi_result
lr=0.1
hidden=64
batch_size=512
epochs=100
if [ "$mode" = "pretrain" ]; then
    echo =============
    echo ">>>>  pretrain in unlabeled data" 
    for dname in ${dataset_list[*]} 
    do
        if [ "$dname" = "github_stargazers" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 256 \
                --batch_size 256 \
                --lr 0.001 \
                --epochs 50 \
                --save $save \
                --gpu $gpu
        echo "Finished training on ${dname}"
        elif [ "$dname" = "PROTEINS" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 128 \
                --batch_size 256 \
                --lr 0.01 \
                --epochs 100 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "DD" ]; then
            echo =============
            echo ">>>> Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 256 \
                --batch_size 128 \
                --lr 0.0001 \
                --epochs 100 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "NCI1" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 128 \
                --batch_size 256 \
                --lr 0.0001 \
                --epochs 30 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "REDDIT-BINARY" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 256 \
                --batch_size 256 \
                --lr 0.001 \
                --epochs 100 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "REDDIT-MULTI-5K" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 256 \
                --batch_size 128 \
                --lr 0.001 \
                --epochs 80 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "COLLAB" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_pretrain \
                --dataset $dname \
                --hidden 256 \
                --batch_size 128 \
                --lr 0.0001 \
                --epochs 80 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}" 
        fi
    done
    echo "Finished all training!"
elif [ "$mode" = "finetune" ]; then
    echo =============
    echo ">>>>  finetune in 10% labeled data"
    for dname in ${dataset_list[*]} 
    do
        if [ "$dname" = "github_stargazers" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 256 \
                --batch_size 128 \
                --lr 0.0001 \
                --save $save \
                --gpu $gpu
        echo "Finished training on ${dname}"
        elif [ "$dname" = "PROTEINS" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 128 \
                --batch_size 16 \
                --lr 0.0001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "DD" ]; then
            echo =============
            echo ">>>> Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 256 \
                --batch_size 64 \
                --lr 0.001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "NCI1" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 128 \
                --batch_size 64 \
                --lr 0.0001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "REDDIT-BINARY" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 256 \
                --batch_size 128 \
                --lr 0.0001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "REDDIT-MULTI-5K" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 256 \
                --batch_size 64 \
                --lr 0.0001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}"
        elif [ "$dname" = "COLLAB" ]; then
            echo =============
            echo ">>>>  Dataset: ${dname}"  
            python main.py \
                --exp=cl_finetune \
                --dataset $dname \
                --hidden 256 \
                --batch_size 64 \
                --lr 0.001 \
                --save $save \
                --gpu $gpu
            echo "Finished training on ${dname}" 
        fi
    done
    echo "Finished all training!"
fi
