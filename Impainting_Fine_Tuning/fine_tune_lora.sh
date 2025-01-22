export MODEL_NAME="stabilityai/stable-diffusion-2"
now="$(date +%Y%m%d%H%M%S)"
export OUTPUT_DIR="results/results$now"

accelerate launch --mixed_precision="no"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=dataset_finetuning/ \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="A piece of trash underwater" \
  --seed=1337
