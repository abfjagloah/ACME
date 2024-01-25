gpu=$1
dataset_list=(MUTAG PROTEINS DD NCI1 IMDB-BINARY REDDIT-BINARY REDDIT-MULTI-5K COLLAB)
save=us_result
lr=0.1
num_gc_layers=3
tau=0.5
hidden_dim=64
batch_size=512
pool=sum
for dname in ${dataset_list[*]} 
do
    if [ "$dname" = "MUTAG" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 3 \
            --lr 0.1 \
            --tau 0.5 \
            --hidden_dim 64 \
            --batch_size 512 \
            --pool mean \
            --save $save \
            --gpu $gpu
    echo "Finished training on ${dname}"
    elif [ "$dname" = "PROTEINS" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 2 \
            --lr 0.001 \
            --tau 0.5 \
            --hidden_dim 256 \
            --batch_size 512 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "DD" ]; then
        echo =============
        echo ">>>> Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 3 \
            --lr 0.0001 \
            --tau 0.2 \
            --hidden_dim 512 \
            --batch_size 256 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NCI1" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 5 \
            --lr 0.0001 \
            --tau 0.1 \
            --hidden_dim 512 \
            --batch_size 512 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "IMDB-BINARY" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 4 \
            --lr 0.0001 \
            --tau 0.9 \
            --hidden_dim 256 \
            --batch_size 256 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "REDDIT-BINARY" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 2 \
            --lr 0.001 \
            --tau 0.5 \
            --hidden_dim 128 \
            --batch_size 256 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "REDDIT-MULTI-5K" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 4 \
            --lr 0.001 \
            --tau 0.5 \
            --hidden_dim 128 \
            --batch_size 512 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}"
    elif [ "$dname" = "COLLAB" ]; then
        echo =============
        echo ">>>>  Dataset: ${dname}"  
        python us_main.py \
            --dataset $dname \
            --num_gc_layers 3 \
            --lr 0.0001 \
            --tau 0.1 \
            --hidden_dim 128 \
            --batch_size 512 \
            --pool $pool \
            --save $save \
            --gpu $gpu
        echo "Finished training on ${dname}" 
    fi
done
echo "Finished all training!"