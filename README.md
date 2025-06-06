# MoE-Beyond

Transformer-Based Expert Activation Prediction for MoE models.

---

## Setup

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
---
## Instructions to Run Training

### 1. Run the Training Command
Now you're ready to run the training! Use the following command (sample below) to start training:

```bash
python transformer_train.py \
  --data-dir /path/to/dataset_csvs/ \
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
## Instructions to Download Model (.pth file)

### 1. Run the Git LFS Commands after cloning the repo
```bash
For Linux/Ubuntu
sudo apt-get install git-lfs
git clone <this_repo>
cd /dir/with/model.pth/file/and/.gitattributes
git lfs install
git lfs pull
```
---
## Instructions to Run Inference

### 1. Run the Inference Command
Now you're ready to run the infernce! Use the following command (sample below) to start running inference:

```bash
python transformer_train.py \
  --data-dir /path/to/dataset_csvs/ \
  --checkpoint-path ./logs/best_model.pth \
  --max-files 400 \
  --eval-only
```
---



## License
TODO: Add license here
```
