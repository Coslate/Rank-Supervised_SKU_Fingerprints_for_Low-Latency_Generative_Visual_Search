import numpy as np
import torch
from Image_processing.image_process import ACTION_FUNCS, VLAAction
from Image_processing.image_feature import compute_quality_features


def generate_labels(base_dataset, model_P, clip_model, preprocess, device="cuda"):
    samples = []
    
    for img, meta in base_dataset:
        # extract CLIP features (raw input)
        print('-'*20+"\n"+"Image No.: " + str(len(samples)))
        with torch.no_grad():
            c = preprocess(img).unsqueeze(0)
            v = clip_model.encode_image(c)
            v = v / v.norm(dim=-1, keepdim=True)
            visual_np = v.cpu().numpy()[0]

        # image quality features
        q_np = compute_quality_features(img)

        # test all actions using model_P
        rewards = []
        for action in VLAAction:
            processed = ACTION_FUNCS[action](img)
            score = model_P(processed)  
            rewards.append(score)

        best_action = int(np.argmax(rewards))

        samples.append({
            "visual": visual_np.astype(np.float32),
            "quality": q_np.astype(np.float32),
            "label": best_action,
        })
        

    return samples
