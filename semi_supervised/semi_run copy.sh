
# Pre-train
# python main.py --exp=cl_pretrain  --dataset=PROTEINS --save=cl_exp --epochs=50 --batch_size=64 --lr=0.01
# python main.py --exp=cl_pretrain  --dataset=DD --save=cl_exp  --epochs=50 --batch_size=64 --lr=0.001
# python main.py --exp=cl_pretrain  --dataset=NCI1 --save=cl_exp --epochs=50 --batch_size=128 --lr=0.0001
# python main.py --exp=cl_pretrain  --dataset=COLLAB --save=cl_exp --epochs=100 --batch_size=64 --lr=0.001
# python main.py --exp=cl_pretrain  --dataset=github_stargazers --save=cl_exp --epochs=50 --batch_size=128 --lr=0.01
# python main.py --exp=cl_pretrain  --dataset=REDDIT-BINARY --save=cl_exp --epochs=100 --batch_size=256 --lr=0.001
# python main.py --exp=cl_pretrain  --dataset=REDDIT-MULTI-5K --save=cl_exp --epochs=50 --batch_size=128 --lr=0.001

# Fine-tune
python main.py --exp=cl_finetune  --dataset=PROTEINS --save=cl_exp --epochs=100 --batch_size=128 --lr=0.01
python main.py --exp=cl_finetune  --dataset=DD --save=cl_exp  --epochs=100 --batch_size=64 --lr=0.0001
python main.py --exp=cl_finetune  --dataset=NCI1 --save=cl_exp --epochs=100 --batch_size=64 --lr=0.0001
python main.py --exp=cl_finetune  --dataset=COLLAB --save=cl_exp --epochs=100 --batch_size=128 --lr=0.001
python main.py --exp=cl_finetune  --dataset=github_stargazers --save=cl_exp --epochs=100 --batch_size=128 --lr=0.001
python main.py --exp=cl_finetune  --dataset=REDDIT-BINARY --save=cl_exp --epochs=100 --batch_size=32 --lr=0.0001
python main.py --exp=cl_finetune  --dataset=REDDIT-MULTI-5K --save=cl_exp --epochs=100 --batch_size=128 --lr=0.0001
