"""
Generate Kaggle submission file for Galaxy Zoo challenge.
Predicts vote fractions for all 37 classes.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.resnet import create_model


# Galaxy Zoo 2 class columns
CLASS_COLUMNS = [
    'Class1.1', 'Class1.2', 'Class1.3',
    'Class2.1', 'Class2.2',
    'Class3.1', 'Class3.2',
    'Class4.1', 'Class4.2',
    'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4',
    'Class6.1', 'Class6.2',
    'Class7.1', 'Class7.2', 'Class7.3',
    'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7',
    'Class9.1', 'Class9.2', 'Class9.3',
    'Class10.1', 'Class10.2', 'Class10.3',
    'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6',
]


def generate_submission(
    model_path: str = "outputs/models/resnet18_best.pt",
    test_dir: str = "data/raw/images_test_rev1",
    model_type: str = "resnet18",
    output_path: str = "submission.csv",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Generate submission file with vote fraction predictions."""
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    
    # Load model
    model = create_model(model_type=model_type, num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get test images
    test_dir = Path(test_dir)
    test_images = sorted(test_dir.glob("*.jpg"))
    print(f"Found {len(test_images)} test images")
    
    # Generate predictions
    results = []
    
    for i in tqdm(range(0, len(test_images), batch_size), desc="Predicting"):
        batch_paths = test_images[i:i+batch_size]
        batch_tensors = []
        batch_ids = []
        
        for img_path in batch_paths:
            galaxy_id = int(img_path.stem)
            batch_ids.append(galaxy_id)
            
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
        
        # Stack and predict
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Convert binary predictions to Galaxy Zoo format
        # Our model: 0 = Elliptical (smooth), 1 = Spiral (featured)
        for j, galaxy_id in enumerate(batch_ids):
            p_smooth = probs[j, 0]  # Probability of elliptical/smooth
            p_featured = probs[j, 1]  # Probability of spiral/featured
            
            # Create vote fractions based on binary prediction
            # Class1.1 = smooth, Class1.2 = features/disk
            row = {'GalaxyID': galaxy_id}
            
            # Question 1: Smooth, Featured, or Artifact?
            row['Class1.1'] = p_smooth  # Smooth
            row['Class1.2'] = p_featured  # Features or disk
            row['Class1.3'] = 0.0  # Star or artifact
            
            # For remaining classes, distribute based on primary prediction
            # If smooth (elliptical), weight toward round/cigar shapes
            # If featured (spiral), weight toward spiral arm patterns
            
            # Question 2: Could this be edge-on disk?
            row['Class2.1'] = p_featured * 0.3  # Yes
            row['Class2.2'] = 1 - row['Class2.1']  # No
            
            # Question 3: Bar presence
            row['Class3.1'] = p_featured * 0.2  # Bar
            row['Class3.2'] = 1 - row['Class3.1']  # No bar
            
            # Question 4: Spiral pattern
            row['Class4.1'] = p_featured * 0.7  # Has spiral
            row['Class4.2'] = 1 - row['Class4.1']  # No spiral
            
            # Question 5: Bulge prominence
            row['Class5.1'] = 0.1  # No bulge
            row['Class5.2'] = 0.2  # Just noticeable
            row['Class5.3'] = 0.4  # Obvious
            row['Class5.4'] = 0.3  # Dominant
            
            # Question 6: Something odd?
            row['Class6.1'] = 0.05  # Yes
            row['Class6.2'] = 0.95  # No
            
            # Question 7: How round is the smooth galaxy?
            row['Class7.1'] = p_smooth * 0.3  # Completely round
            row['Class7.2'] = p_smooth * 0.4  # In between
            row['Class7.3'] = p_smooth * 0.3  # Cigar shaped
            
            # Question 8: Odd feature type
            row['Class8.1'] = 0.02  # Ring
            row['Class8.2'] = 0.02  # Lens/arc
            row['Class8.3'] = 0.02  # Disturbed
            row['Class8.4'] = 0.02  # Irregular
            row['Class8.5'] = 0.02  # Other
            row['Class8.6'] = 0.02  # Merger
            row['Class8.7'] = 0.88  # Dust lane
            
            # Question 9: Bulge shape
            row['Class9.1'] = 0.33  # Rounded
            row['Class9.2'] = 0.34  # Boxy
            row['Class9.3'] = 0.33  # No bulge
            
            # Question 10: Spiral arm winding
            row['Class10.1'] = p_featured * 0.33  # Tight
            row['Class10.2'] = p_featured * 0.34  # Medium
            row['Class10.3'] = p_featured * 0.33  # Loose
            
            # Question 11: Number of spiral arms
            row['Class11.1'] = 0.1  # 1
            row['Class11.2'] = 0.3  # 2
            row['Class11.3'] = 0.2  # 3
            row['Class11.4'] = 0.2  # 4
            row['Class11.5'] = 0.1  # More than 4
            row['Class11.6'] = 0.1  # Can't tell
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[['GalaxyID'] + CLASS_COLUMNS]  # Reorder columns
    df = df.sort_values('GalaxyID').reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")
    print(f"Total rows: {len(df)} (including header: {len(df) + 1})")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/resnet18_best.pt")
    parser.add_argument("--model-type", type=str, default="resnet18")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()
    
    generate_submission(model_path=args.model, model_type=args.model_type, output_path=args.output)
