# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import os.path as osp
import numpy as np
import argparse
from PIL import Image

from network.atloc import AtLoc, AtLocPlus
from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from torch.autograd import Variable

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AtLoc Single Image Inference')
    
    # Model related parameters
    parser.add_argument('--model', type=str, default='AtLoc', choices=['AtLoc', 'AtLocPlus'],
                        help='Model type')
    parser.add_argument('--weights', type=str, required=True,
                        help='Model weights file path')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Input image path')
    
    # Data preprocessing parameters
    parser.add_argument('--dataset', type=str, default='7Scenes', choices=['7Scenes', 'RobotCar'],
                        help='Dataset type')
    parser.add_argument('--scene', type=str, default='chess',
                        help='Scene name')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory path')
    parser.add_argument('--cropsize', type=int, default=256,
                        help='Image crop size')
    
    # Model parameters
    parser.add_argument('--lstm', action='store_true', default=False,
                        help='Whether to use LSTM')
    parser.add_argument('--test_dropout', type=float, default=0.0,
                        help='Dropout rate during testing')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()

def load_model(args):
    """Load model"""
    print("Loading model: {}".format(args.model))
    
    # Set device
    cuda = torch.cuda.is_available()
    device = "cuda:{}".format(args.gpu) if cuda else "cpu"
    print("Using device: {}".format(device))
    
    # Create model
    feature_extractor = models.resnet34(pretrained=False)
    atloc = AtLoc(feature_extractor, droprate=args.test_dropout, pretrained=False, lstm=args.lstm)
    
    if args.model == 'AtLoc':
        model = atloc
    elif args.model == 'AtLocPlus':
        model = AtLocPlus(atlocplus=atloc)
    else:
        raise NotImplementedError("Unimplemented model type: {}".format(args.model))
    
    model.eval()
    model.to(device)
    
    # Load weights
    weights_filename = osp.expanduser(args.weights)
    if osp.isfile(weights_filename):
        print("Loading weights from {}".format(weights_filename))
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print("Weights loaded successfully")
    else:
        raise IOError("Weights file does not exist: {}".format(weights_filename))
    
    return model, device

def load_stats(args):
    """Load dataset statistics"""
    stats_file = osp.join(args.data_dir, args.dataset, args.scene, 'stats.txt')
    pose_stats_file = osp.join(args.data_dir, args.dataset, args.scene, 'pose_stats.txt')
    
    if not osp.exists(stats_file):
        print("Warning: Stats file does not exist {}".format(stats_file))
        print("Using default ImageNet statistics")
        # Default ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        pose_m = np.array([0.0, 0.0, 0.0])
        pose_s = np.array([1.0, 1.0, 1.0])
    else:
        print("Loading stats_file statistics from {}".format(stats_file))
        print("Loading pose_stats_file statistics from {}".format(pose_stats_file))
        stats = np.loadtxt(stats_file)
        mean = stats[0]
        std = np.sqrt(stats[1])
        
        if osp.exists(pose_stats_file):
            pose_m, pose_s = np.loadtxt(pose_stats_file)
        else:
            print("Warning: Pose stats file does not exist {}".format(pose_stats_file))
            pose_m = np.array([0.0, 0.0, 0.0])
            pose_s = np.array([1.0, 1.0, 1.0])
    
    return mean, std, pose_m, pose_s

def preprocess_image(image_path, mean, std, cropsize):
    """Preprocess image"""
    print("Loading and preprocessing image: {}".format(image_path))
    
    if not osp.exists(image_path):
        raise IOError("Image file does not exist: {}".format(image_path))
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print("Original image size: {}".format(image.size))
    
    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize(cropsize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    print("Preprocessed tensor shape: {}".format(image_tensor.shape))
    
    return image_tensor

def inference(model, image_tensor, device):
    """Perform inference"""
    print("Performing inference...")
    
    # Move data to device
    image_tensor = image_tensor.to(device)
    
    # Disable gradient computation
    with torch.no_grad():
        # Forward pass
        output = model(image_tensor)
        
        # Move output to CPU and convert to numpy
        output = output.cpu().numpy()
    
    print("Model output shape: {}".format(output.shape))
    return output

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix"""
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R

def create_transformation_matrix(translation, rotation_quat):
    """Create 4x4 transformation matrix from translation and quaternion"""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(rotation_quat)
    T[:3, 3] = translation
    return T

def postprocess_output(output, pose_m, pose_s):
    """Postprocess output"""
    print("Postprocessing output...")
    
    # Get position and rotation
    translation = output[0, :3]  # First 3 values are position
    rotation_log = output[0, 3:]  # Last 3 values are log quaternion
    
    # Convert log quaternion to quaternion
    rotation_quat = qexp(rotation_log)
    
    # Denormalize position
    translation_denorm = (translation * pose_s) + pose_m
    
    # Create transformation matrix
    transformation_matrix = create_transformation_matrix(translation_denorm, rotation_quat)
    
    return translation_denorm, rotation_quat, translation, rotation_log, transformation_matrix

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    print("=" * 60)
    print("AtLoc Single Image Inference")
    print("=" * 60)
    
    try:
        # Load model
        model, device = load_model(args)
        
        # Load statistics
        mean, std, pose_m, pose_s = load_stats(args)
        print("Data statistics:")
        print("  Mean: {}".format(mean))
        print("  Std: {}".format(std))
        print("  Pose mean: {}".format(pose_m))
        print("  Pose std: {}".format(pose_s))
        
        # Preprocess image
        image_tensor = preprocess_image(args.image_path, mean, std, args.cropsize)
        
        # Inference
        output = inference(model, image_tensor, device)
        
        # Postprocess
        translation_denorm, rotation_quat, translation_norm, rotation_log, transformation_matrix = postprocess_output(output, pose_m, pose_s)
        
        # Output results
        print("=" * 60)
        print("Inference Results:")
        print("=" * 60)
        
        # Print 4x4 Transformation Matrix (same format as ground truth)
        print("4x4 Transformation Matrix:")
        for i in range(4):
            row_str = " ".join(["{:10.7f}".format(transformation_matrix[i, j]) for j in range(4)])
            print(row_str)
        print("")
        
        print("Detailed Results:")
        print("Normalized position (x, y, z): [{:.6f}, {:.6f}, {:.6f}]".format(
            translation_norm[0], translation_norm[1], translation_norm[2]))
        print("Log quaternion (log_qx, log_qy, log_qz): [{:.6f}, {:.6f}, {:.6f}]".format(
            rotation_log[0], rotation_log[1], rotation_log[2]))
        print("-" * 60)
        print("Denormalized position (x, y, z): [{:.6f}, {:.6f}, {:.6f}] meters".format(
            translation_denorm[0], translation_denorm[1], translation_denorm[2]))
        print("Quaternion (w, x, y, z): [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
            rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3]))
        
        # Calculate rotation angles (for intuitive understanding)
        # Simple quaternion to Euler angles conversion (roll, pitch, yaw)
        w, x, y, z = rotation_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)) * 180 / np.pi
        pitch = np.arcsin(2*(w*y - z*x)) * 180 / np.pi
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) * 180 / np.pi
        
        print("Euler angles (roll, pitch, yaw): [{:.2f}°, {:.2f}°, {:.2f}°]".format(
            roll, pitch, yaw))
        print("=" * 60)
        
        # Save results to file
        output_file = args.image_path.replace(osp.splitext(args.image_path)[1], '_pose.txt')
        with open(output_file, 'w') as f:
            f.write("AtLoc Pose Estimation Results\n")
            f.write("=" * 40 + "\n")
            f.write("Input image: {}\n".format(args.image_path))
            f.write("Model used: {}\n".format(args.model))
            f.write("Weights file: {}\n".format(args.weights))
            f.write("\n4x4 Transformation Matrix:\n")
            for i in range(4):
                row_str = " ".join(["{:10.7f}".format(transformation_matrix[i, j]) for j in range(4)])
                f.write(row_str + "\n")
            f.write("\nDetailed Results:\n")
            f.write("Position (meters): {:.6f}, {:.6f}, {:.6f}\n".format(
                translation_denorm[0], translation_denorm[1], translation_denorm[2]))
            f.write("Quaternion: {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(
                rotation_quat[0], rotation_quat[1], rotation_quat[2], rotation_quat[3]))
            f.write("Euler angles (degrees): {:.2f}, {:.2f}, {:.2f}\n".format(
                roll, pitch, yaw))
        
        print("Results saved to: {}".format(output_file))
        
    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()