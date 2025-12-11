import pickle
from train.policy_train import train_policy
from Image_processing.image_process import VLAAction
import argparse
import open_clip
import torch

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
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/soinew/genAIdata/VLA/labels.pkl",
        help="Path to input labels pickle file.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3000,
        help="Number of training epochs (default: 3000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on (default: cuda if available, else cpu).",
    )

    return parser.parse_args()

def main():    
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    visual_dim = clip_model.visual.output_dim
    quality_dim = 10
    num_actions = len(VLAAction)
    load_dir = args.input_path
    
    print(f"Loading training data from {load_dir}...")
    samples = pickle.load(open(load_dir, "rb"))
    print(f"Loaded {len(samples)} training samples")
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    model = train_policy(
        samples, 
        visual_dim, 
        quality_dim, 
        num_actions, 
        device=str(device),
        num_epochs=args.num_epochs
    )

    output_path = "vla_policy.pt"
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved model to {output_path}")

if __name__ == "__main__":
    main()

