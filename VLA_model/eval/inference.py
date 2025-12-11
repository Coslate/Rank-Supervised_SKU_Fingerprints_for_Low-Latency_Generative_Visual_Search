import torch
from PIL import Image
from model.policy import VLAPolicy
from Image_processing.image_feature import compute_quality_features
from Image_processing.image_process import VLAAction
import argparse
import open_clip

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image-label pairs for VLA"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="open_clip model name (e.g., ViT-B-16).",
    )

    """
    !!! Notice: please keep the model same as what you used for 'main_gen_label' 
    """

    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag.",
    )

@torch.no_grad()
def predict_best_action(img, vla_model, clip_model, preprocess, device="cuda"):
    clip_in = preprocess(img).unsqueeze(0).to(device)
    vfeat = clip_model.encode_image(clip_in)
    vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)

    qfeat = compute_quality_features(img)
    qfeat = torch.tensor(qfeat).unsqueeze(0).to(device)

    logits = vla_model(vfeat, qfeat)
    action_id = int(logits.argmax(dim=-1))

    return VLAAction(action_id)

def main():
    args = parse_args()
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    visual_dim = clip_model.visual.output_dim
    policy = VLAPolicy(visual_dim=visual_dim, quality_dim=10, num_actions=len(VLAAction))
    policy.load_state_dict(torch.load("vla_policy.pt"))
    policy.eval()

    img = Image.open("test.jpg")

    action = predict_best_action(img, policy, clip_model, preprocess)

    print("Best action:", action.name)

if __name__ == "__main__":
    main()
