DATASET_PATH = "TrashCan/instance_version/"
AUGMENTED_DATASET_PATH = "augmented_dataset/"
LORA_PATH = "weights/"
LORA_WEIGHTS = "LoRA_2_pytorch_lora_weights.safetensors"
FOCUS_NET_PATH = "weights/FocusNetv4_Mbasic_padding_LL1+IOU_OSGD_LR0.065_E150.pth"

category_labels = {
    1: "rov",
    2: "plant",
    3: "animal_fish",
    4: "animal_starfish",
    5: "animal_shells",
    6: "animal_crab",
    7: "animal_eel",
    8: "animal_etc",
    9: "trash_clothing",
    10: "trash_pipe",
    11: "trash_bottle",
    12: "trash_bag",
    13: "trash_snack_wrapper",
    14: "trash_can",
    15: "trash_cup",
    16: "trash_container",
    17: "trash_unknown_instance",
    18: "trash_branch",
    19: "trash_wreckage",
    20: "trash_tarp",
    21: "trash_rope",
    22: "trash_net",
}

prompts = {
    1: "an underwater remotely operated vehicle (ROV), realistic, blending with the environment",
    2: "an underwater plant, swaying gently, realistic, blending with the marine environment",
    3: "a fish swimming underwater, realistic, blending with the marine environment",
    4: "a starfish resting on the seabed, realistic, blending with the marine environment",
    5: "shells on the seabed, realistic, blending with the environment",
    6: "a crab walking on the ocean floor, realistic, clearly visible, blending with the marine environment",
    7: "an eel swimming underwater, realistic, blending with the marine environment",
    8: "a small underwater animal, realistic, blending with the marine environment",
    9: "a piece of clothing as trash underwater, realistic, blending with the environment",
    10: "a metal pipe as trash underwater, realistic, blending with the environment",
    11: "a trash bottle underwater, realistic, blending with the marine environment",
    12: "a plastic bag as trash underwater, realistic, blending with the marine environment",
    13: "a snack wrapper as trash underwater, realistic, blending with the marine environment",
    14: "a discarded can as trash underwater, realistic, blending with the marine environment",
    15: "a plastic cup as trash underwater, realistic, blending with the marine environment",
    16: "a container as trash underwater, realistic, blending with the marine environment",
    17: "an plastic trash object underwater, realistic, blending with the marine environment",
    18: "a wooden branch underwater as trash, realistic, blending with the marine environment",
    19: "a piece of wreckage underwater, realistic, blending with the marine environment",
    20: "a tarp as trash underwater, realistic, blending with the marine environment",
    21: "a realistic rope underwater as trash, realistic, blending with the marine environment",
    22: "a fishing net tangled underwater as trash, realistic, blending with the marine environment",
}