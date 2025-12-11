from train.generate_labels import generate_labels
import argparse
import pickle
import open_clip
from pathlib import Path
from PIL import Image
import torch
import sys
import os
from functools import partial
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from models.clip_sku_baseline import ClipSkuBaseline
from dataset.df2_clip_sku_dataset import DeepFashion2ImageSkuEvalDataset
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import DatasetFolder

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
    
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag.",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/soinew/genAIdata/VLA/labels.pkl",
        help="open_clip pretrained tag.",
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default= "/home/soinew/Rank-Supervised_SKU_Fingerprints_for_Low-Latency_Generative_Visual_Search/VLA_model/data/modelforVLA.pt",
        help="Path to trained model checkpoint (.pt file).",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top SKU predictions to return.",
    )

    parser.add_argument(
        "--image_load",
        type=str,
        default="/home/soinew/genAIdata/dataforVLA",
        help="Image load path.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()

def load_model(checkpoint_path: Path, clip_model_name: str, clip_pretrained: str, device: str):
    """
    Load trained ClipSkuBaseline model from checkpoint.
    
    Returns:
        model: Loaded ClipSkuBaseline model
        sku2idx: Dictionary mapping SKU index to SKU ID (inverse of training's sku2idx)
        args: Training arguments from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args
    args = ckpt.get("args", {})
    num_skus = len(ckpt["sku2idx"])
    sku2idx = ckpt["sku2idx"]  # sku_id (string) -> index (int)
    
    # Create inverse mapping: index -> sku_id
    idx2sku = {idx: sku_id for sku_id, idx in sku2idx.items()}
    
    # print(f"Model trained with {num_skus} SKUs")
    # print(f"CLIP model: {args.get('clip_model', clip_model_name)}")
    # print(f"CLIP pretrained: {args.get('clip_pretrained', clip_pretrained)}")
    
    # Create CLIP model (must match training)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.get("clip_model", clip_model_name),
        pretrained=args.get("clip_pretrained", clip_pretrained)
    )
    
    # Create ClipSkuBaseline model
    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=args.get("freeze_towers", True),
    )
    
    # Load trained weights
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, idx2sku, preprocess
    



def predict_sku(image, model: ClipSkuBaseline, preprocess, device: str, top_k: int = 5):
    """
    Predict SKU for a single image.
    
    Args:
        image_path: Path to input image
        model: Trained ClipSkuBaseline model
        preprocess: Image preprocessing function
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        predictions: List of (sku_idx, sku_id, score) tuples, sorted by score
    """
    # Load and preprocess image
    img = image
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension: (1, 3, H, W)
    
    # Get image embedding
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)  # (1, D)
        
        # Get all SKU embeddings
        sku_embs = model.sku_embeddings()  # (num_skus, D)
        
        # Get logit scale
        logit_scale = model.logit_scale.exp()
        
        # Compute similarity scores: image embedding @ SKU embeddings^T
        scores = logit_scale * (img_emb @ sku_embs.t())  # (1, num_skus)
        scores = scores.squeeze(0)  # (num_skus,)
    
    # Get top-k predictions
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)), dim=0)
    
    predictions = [
        (int(idx.item()), float(score.item()))
        for idx, score in zip(top_indices, top_scores)
    ]
    
    return predictions

def score(image,model,idx2sku,preprocess):
    args = parse_args()
    device = torch.device(args.device)
    
    # Load model
    # model, idx2sku, preprocess = load_model(
    #     args.checkpoint,
    #     args.clip_model,
    #     args.clip_pretrained,
    #     device
    # )
    
    # Predict SKU
    print(f"\nPredicting SKU for image: {image}")
    predictions = predict_sku(image, model, preprocess, device, args.top_k)
    
    # Print results
    print(f"\nTop {args.top_k} SKU predictions:")
    print("-" * 60)
    for rank, (sku_idx, score) in enumerate(predictions, 1):
        sku_id = idx2sku[sku_idx]
        print(f"Rank {rank}: SKU ID = {sku_id}, Score = {score:.4f}")
    print("-" * 60)
    
    # Return top prediction
    top_sku_idx, top_score = predictions[0]
    top_sku_id = idx2sku[top_sku_idx]
    print(f"\nPredicted SKU: {top_sku_id} (score: {top_score:.4f})")

    return top_score

class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, img_name   # return both image and filename


def main():
    args = parse_args()
    device = torch.device(args.device)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model, idx2sku, preprocess = load_model(
        args.checkpoint,
        args.clip_model,
        args.clip_pretrained,
        device
    )
    
    dataset = ImageFolderDataset(args.image_load)
    def single_sample_collate(batch):
    # batch is a list with one element when batch_size=1
        return batch[0]
    loader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=single_sample_collate,)
    # Create a partial function that binds model, idx2sku, and preprocess
    score_fn = partial(score, model=model, idx2sku=idx2sku, preprocess=preprocess)
    samples = generate_labels(loader, score_fn, clip_model, preprocess)
    print(len(samples))
    save_dir = args.output_path
    with open(save_dir, "wb") as f:
        pickle.dump(samples, f)

    print(f"Saved label samples to {save_dir}")

if __name__ == "__main__":
    main()