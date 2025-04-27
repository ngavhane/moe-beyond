# DeepSeek Prompts

A repository for running and training models with DeepSeek. Follow the instructions below to get started with training the model.

---

## Instructions to Run Training

### 1. Create and Enter Project Directory
First, create a directory for storing logs and navigate into it:

```bash
mkdir deepseek-logs
cd deepseek-logs
```

### 2. (Optional) Create and Activate a Virtual Environment
It's recommended to use a virtual environment for managing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required dependencies:

```bash
pip install torch bitsandbytes transformers accelerate pandas numpy scikit-learn tensorboardX tensorboard
```

### 4. Download the Code
Clone this repository to your local machine:

```bash
git clone <this_repo>
```

### 5. Run the Training Command
Now you're ready to run the training! Use the following command (sample below) to start training:

```bash
python transformer_train.py \
  --data-dir /home/ubuntu/ngavhane-fs/dataset_csvs/ \
  --batch-size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --log-dir ./logs \
  --checkpoint-path ./logs/best_model.pth \
  --num-layers 4 \
  --nhead 8 \
  --dim-feedforward 2048 \
  --dropout-rate 0.1 \
  --max-files 400 \
  --max-length 512
```

---

## License
Add later
```
