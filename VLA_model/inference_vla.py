"""
VLA Model Inference Script

This script loads a trained VLA policy model and predicts the best action
for a given input image.
"""

import torch
from PIL import Image
from pathlib import Path
import argparse
import open_clip
import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try importing with VLA_model prefix first, fallback to direct import
try:
    from VLA_model.model.policy import VLAPolicy
    from VLA_model.Image_processing.image_feature import compute_quality_features
    from VLA_model.Image_processing.image_process import VLAAction, ACTION_FUNCS
    from models.clip_sku_baseline import ClipSkuBaseline
except ImportError:
    # If running from within VLA_model directory
    from model.policy import VLAPolicy
    from Image_processing.image_feature import compute_quality_features
    from Image_processing.image_process import VLAAction, ACTION_FUNCS
    # Add project root for models import
    project_root_for_models = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root_for_models not in sys.path:
        sys.path.insert(0, project_root_for_models)
    from models.clip_sku_baseline import ClipSkuBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA Model Inference - Predict action label for an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/soinew/Rank-Supervised_SKU_Fingerprints_for_Low-Latency_Generative_Visual_Search/VLA_model/checkpoint_vla/vla_policy.pt",
        help="Path to VLA policy checkpoint file (default: vla_policy.pt).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="open_clip model name (e.g., ViT-B-16). Must match training configuration.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag. Must match training configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--sku_checkpoint",
        type=str,
        default="/home/soinew/Rank-Supervised_SKU_Fingerprints_for_Low-Latency_Generative_Visual_Search/VLA_model/data/modelforVLA.pt",
        help="Path to ClipSkuBaseline checkpoint for SKU scoring (default: data/modelforVLA.pt).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of top SKU scores to show for each action (default: 1).",
    )
    return parser.parse_args()


def load_vla_model(checkpoint_path: str, clip_model_name: str, clip_pretrained: str, device: str):
    """
    Load the VLA policy model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        clip_model_name: CLIP model name (must match training)
        clip_pretrained: CLIP pretrained tag (must match training)
        device: Device to load model on
    
    Returns:
        policy_model: Loaded VLAPolicy model
        clip_model: CLIP model for feature extraction
        preprocess: Image preprocessing function
    """
    print(f"Loading VLA model from {checkpoint_path}...")
    
    # Create CLIP model (must match training configuration)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Get dimensions from CLIP model
    visual_dim = clip_model.visual.output_dim
    quality_dim = 10  # Fixed quality feature dimension
    num_actions = len(VLAAction)
    
    # Create VLA policy model
    policy_model = VLAPolicy(
        visual_dim=visual_dim,
        quality_dim=quality_dim,
        num_actions=num_actions
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint)
    policy_model.to(device)
    policy_model.eval()
    
    print("Model loaded successfully!")
    print(f"  - Visual dimension: {visual_dim}")
    print(f"  - Quality dimension: {quality_dim}")
    print(f"  - Number of actions: {num_actions}")
    
    return policy_model, clip_model, preprocess


def load_sku_model(checkpoint_path: str, clip_model_name: str, clip_pretrained: str, device: str):
    """
    Load the ClipSkuBaseline model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        clip_model_name: CLIP model name (must match training)
        clip_pretrained: CLIP pretrained tag (must match training)
        device: Device to load model on
    
    Returns:
        sku_model: Loaded ClipSkuBaseline model
        idx2sku: Dictionary mapping index to SKU ID
        sku_preprocess: Image preprocessing function for SKU model
    """
    print(f"\nLoading SKU model from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args
    args = ckpt.get("args", {})
    num_skus = len(ckpt["sku2idx"])
    sku2idx = ckpt["sku2idx"]  # sku_id (string) -> index (int)
    
    # Create inverse mapping: index -> sku_id
    idx2sku = {idx: sku_id for sku_id, idx in sku2idx.items()}
    
    # Create CLIP model (must match training)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.get("clip_model", clip_model_name),
        pretrained=args.get("clip_pretrained", clip_pretrained)
    )
    
    # Create ClipSkuBaseline model
    sku_model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=args.get("freeze_towers", True),
    )
    
    # Load trained weights
    sku_model.load_state_dict(ckpt["model_state"])
    sku_model.to(device)
    sku_model.eval()
    
    print(f"SKU model loaded successfully! ({num_skus} SKUs)")
    return sku_model, idx2sku, preprocess


@torch.no_grad()
def get_sku_score(img: Image.Image, sku_model: ClipSkuBaseline, sku_preprocess, device: str, top_k: int = 1):
    """
    Get the top SKU score for a processed image.
    
    Args:
        img: PIL Image (already processed by an action)
        sku_model: Trained ClipSkuBaseline model
        sku_preprocess: Image preprocessing function
        device: Device to run on
        top_k: Number of top scores to return
    
    Returns:
        top_score: Top SKU score (float)
        top_sku_id: Top SKU ID (string)
    """
    img_tensor = sku_preprocess(img).unsqueeze(0).to(device)
    
    # Get image embedding
    img_emb = sku_model.encode_image(img_tensor)  # (1, D)
    
    # Get all SKU embeddings
    sku_embs = sku_model.sku_embeddings()  # (num_skus, D)
    
    # Get logit scale
    logit_scale = sku_model.logit_scale.exp()
    
    # Compute similarity scores: image embedding @ SKU embeddings^T
    scores = logit_scale * (img_emb @ sku_embs.t())  # (1, num_skus)
    scores = scores.squeeze(0)  # (num_skus,)
    
    # Get top-k predictions
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)), dim=0)
    
    return float(top_scores[0].item()), top_indices[0].item()


@torch.no_grad()
def predict_action(image_path: str, policy_model: VLAPolicy, clip_model, preprocess, device: str):
    """
    Predict the best action for a given image.
    
    Args:
        image_path: Path to input image
        policy_model: Trained VLA policy model
        clip_model: CLIP model for visual features
        preprocess: Image preprocessing function
        device: Device to run inference on
    
    Returns:
        action: VLAAction enum value (the predicted action)
        action_id: Integer ID of the action
        logits: Raw logits from the model (for debugging)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    
    # Extract visual features using CLIP
    clip_input = preprocess(img).unsqueeze(0).to(device)
    vfeat = clip_model.encode_image(clip_input)
    vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)  # Normalize
    
    # Extract quality features
    qfeat = compute_quality_features(img)
    qfeat = torch.tensor(qfeat, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get action prediction
    logits = policy_model(vfeat, qfeat)
    action_id = int(logits.argmax(dim=-1).item())
    action = VLAAction(action_id)
    
    return action, action_id, logits, img


def main():
    args = parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate checkpoint path (try relative to project root if not absolute)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = Path(project_root) / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Validate SKU checkpoint path
    sku_checkpoint_path = Path(args.sku_checkpoint)
    if not sku_checkpoint_path.is_absolute():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sku_checkpoint_path = Path(project_root) / sku_checkpoint_path
    if not sku_checkpoint_path.exists():
        raise FileNotFoundError(f"SKU checkpoint not found: {sku_checkpoint_path}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load VLA model
    policy_model, clip_model, preprocess = load_vla_model(
        str(checkpoint_path),
        args.clip_model,
        args.clip_pretrained,
        str(device)
    )
    
    # Load SKU model
    sku_model, idx2sku, sku_preprocess = load_sku_model(
        str(sku_checkpoint_path),
        args.clip_model,
        args.clip_pretrained,
        str(device)
    )
    
    # Predict action
    print(f"\nPredicting action for image: {image_path}")
    action, action_id, logits, original_img = predict_action(
        str(image_path),
        policy_model,
        clip_model,
        preprocess,
        str(device)
    )
    
    # Print VLA model results
    print("\n" + "="*80)
    print("VLA MODEL SCORES")
    print("="*80)
    print(f"Predicted Action: {action.name} (ID: {action_id})")
    print("\nVLA Action Scores:")
    vla_scores = {}
    for i, action_enum in enumerate(VLAAction):
        score = logits[0][i].item()
        vla_scores[action_enum] = score
        marker = " <-- SELECTED" if i == action_id else ""
        print(f"  {action_enum.name:20s}: {score:8.4f}{marker}")
    
    # Get SKU scores for each action
    print("\n" + "="*80)
    print("SKU MODEL SCORES (after applying each action)")
    print("="*80)
    print(f"Top-{args.top_k} SKU score for each processed image:\n")
    
    sku_scores = {}
    for action_enum in VLAAction:
        # Apply action to image
        processed_img = ACTION_FUNCS[action_enum](original_img.copy())
        
        # Get SKU score
        top_score, top_sku_idx = get_sku_score(
            processed_img, sku_model, sku_preprocess, str(device), args.top_k
        )
        top_sku_id = idx2sku[top_sku_idx]
        sku_scores[action_enum] = (top_score, top_sku_id)
        
        marker = " <-- VLA SELECTED" if action_enum == action else ""
        print(f"  {action_enum.name:20s}: Score = {top_score:8.4f}, Top SKU = {top_sku_id}{marker}")
    
    # Combined comparison
    print("\n" + "="*80)
    print("COMBINED COMPARISON")
    print("="*80)
    print(f"{'Action':<20} {'VLA Score':<12} {'SKU Score':<12} {'Top SKU ID':<15}")
    print("-" * 80)
    for action_enum in VLAAction:
        vla_score = vla_scores[action_enum]
        sku_score, sku_id = sku_scores[action_enum]
        marker = " <-- VLA SELECTED" if action_enum == action else ""
        print(f"{action_enum.name:<20} {vla_score:<12.4f} {sku_score:<12.4f} {sku_id:<15}{marker}")
    print("="*80)
    
    # Apply the predicted action to the original image
    processed_img = ACTION_FUNCS[action](original_img.copy())
    
    print(f"\nProcessed image size: {processed_img.size}")
    print(f"Processed image mode: {processed_img.mode}")
    
    return action, action_id, processed_img


if __name__ == "__main__":
    main()

