# ... Keep necessary imports ...
import os
from PIL import Image
from torchvision import transforms
import shutil
import torch
import numpy as np
import argparse
import vits_fine

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT prediction script', add_help=False)
    # Only keep necessary arguments
    parser.add_argument('--model', default='vit_custom', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # New arguments
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save renamed images')
    return parser

def predict_image(model, image_path, device, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
    return probabilities.cpu().numpy()

def main(args):
    # Set device
    device = torch.device(args.device)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = vits_fine.__dict__[args.model](num_classes=5)
    
    # Load trained weights
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
    model.to(device)
    model.eval()
    
    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Class name mapping (change according to actual situation)
    class_names = ['crack', 'lack_of_fusion', 'lack_of_penetration',
                   'porosity', 'slag']  # Replace with actual class names
    
    # Process all images in the test directory
    for filename in os.listdir(args.test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.test_dir, filename)
            
            # Predict
            probs = predict_image(model, image_path, device, transform)
            
            # Get the highest probability and its corresponding class
            max_prob = np.max(probs)
            pred_class = class_names[np.argmax(probs)]
            
            # Construct new file name: probability_class_original_file_name
            new_filename = f"{max_prob:.2f}_{pred_class}_{filename}"
            new_path = os.path.join(args.output_dir, new_filename)
            
            # Copy and rename file
            shutil.copy2(image_path, new_path)
            print(f"Processed {filename} -> {new_filename}")
            print(f"Class probabilities: {list(zip(class_names, probs))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)