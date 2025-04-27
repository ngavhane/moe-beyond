# deepseek_prompts
==================================================================================================

Instructions to run training:

# 1) Create & enter project dir
mkdir deepseek-logs
cd deepseek-logs

# 2) (Optional) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install torch bitsandbytes transformers accelerate pandas numpy scikit-learn tensorboardX tensorboard

# 4) Download Code
git clone <this_repo>

# 5) Run training commandline (Sample below)
python transformer_train.py --data-dir /home/ubuntu/ngavhane-fs/dataset_csvs/ --batch-size 4 --epochs 50 --lr 1e-4 --log-dir ./logs --checkpoint-path ./logs/best_model.pth --num-layers 4 --nhead 8 --dim-feedforward 2048 --dropout-rate 0.1 --max-files 400 --max-length 512

==================================================================================================
