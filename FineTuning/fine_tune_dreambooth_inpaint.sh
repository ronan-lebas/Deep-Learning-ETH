export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
now="$(date +%Y%m%d%H%M%S)"
export OUTPUT_DIR="results/results$now"

accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=dataset_finetuning3/ \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a piece of trash underwater" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800