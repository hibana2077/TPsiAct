import os
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from scipy.io import loadmat
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io

class DatasetConverter:
    def __init__(self, output_dir="./parquet_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def image_to_bytes(self, image):
        """Convert image to bytes"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            # Ensure ndarray is uint8 and create PIL image
            if image.dtype != np.uint8:
                # Scale floats in [0,1] to 0-255, otherwise cast
                if np.issubdtype(image.dtype, np.floating):
                    image = np.clip(image, 0.0, 1.0)
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure image is RGB for consistent downstream handling
        if hasattr(image, 'mode') and image.mode != 'RGB':
            image = image.convert('RGB')

        buf = io.BytesIO()
        image.save(buf, format='PNG')
        data = buf.getvalue()
        buf.close()
        return data
    
    def convert_cifar10(self):
        """Convert CIFAR-10 dataset"""
        print("Converting CIFAR-10...")
        
        # Load train and test sets
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
        
        # Class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Process train set
        train_data = []
        for i, (image, label) in enumerate(train_dataset):
            train_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'class_name': class_names[label],
                'split': 'train'
            })
            if i % 5000 == 0:
                print(f"CIFAR-10 train progress: {i}/{len(train_dataset)}")
        
        # Process test set
        test_data = []
        for i, (image, label) in enumerate(test_dataset):
            test_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'class_name': class_names[label],
                'split': 'test'
            })
            if i % 1000 == 0:
                print(f"CIFAR-10 test progress: {i}/{len(test_dataset)}")
        
        # Merge and save
        all_data = train_data + test_data
        df = pd.DataFrame(all_data)
        
        # Save as Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(self.output_dir, 'cifar10.parquet'))
        print(f"CIFAR-10 conversion complete, total {len(all_data)} records")
    
    def convert_cifar100(self):
        """Convert CIFAR-100 dataset"""
        print("Converting CIFAR-100...")
        
        # Load train and test sets
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
        
        # Process train set
        train_data = []
        for i, (image, label) in enumerate(train_dataset):
            train_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'split': 'train'
            })
            if i % 5000 == 0:
                print(f"CIFAR-100 train progress: {i}/{len(train_dataset)}")
        
        # Process test set
        test_data = []
        for i, (image, label) in enumerate(test_dataset):
            test_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'split': 'test'
            })
            if i % 1000 == 0:
                print(f"CIFAR-100 test progress: {i}/{len(test_dataset)}")
        
        # Merge and save
        all_data = train_data + test_data
        df = pd.DataFrame(all_data)
        
        # Save as Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(self.output_dir, 'cifar100.parquet'))
        print(f"CIFAR-100 conversion complete, total {len(all_data)} records")
    
    def convert_fashion_mnist(self):
        """Convert Fashion-MNIST dataset"""
        print("Converting Fashion-MNIST...")
        
        # Load train and test sets
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)
        
        # Class names
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # Process train set
        train_data = []
        for i, (image, label) in enumerate(train_dataset):
            # Convert grayscale image to RGB
            image = image.convert('RGB')
            train_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'class_name': class_names[label],
                'split': 'train'
            })
            if i % 5000 == 0:
                print(f"Fashion-MNIST train progress: {i}/{len(train_dataset)}")
        
        # Process test set
        test_data = []
        for i, (image, label) in enumerate(test_dataset):
            # Convert grayscale image to RGB
            image = image.convert('RGB')
            test_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'class_name': class_names[label],
                'split': 'test'
            })
            if i % 1000 == 0:
                print(f"Fashion-MNIST test progress: {i}/{len(test_dataset)}")
        
        # Merge and save
        all_data = train_data + test_data
        df = pd.DataFrame(all_data)
        
        # Save as Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(self.output_dir, 'fashion_mnist.parquet'))
        print(f"Fashion-MNIST conversion complete, total {len(all_data)} records")
    
    def convert_svhn(self):
        """Convert SVHN dataset"""
        print("Converting SVHN...")
        
        # Load train and test sets
        train_dataset = datasets.SVHN(root='./data', split='train', download=True)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True)
        
        # Process train set
        train_data = []
        for i, (image, label) in enumerate(train_dataset):
            train_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'split': 'train'
            })
            if i % 5000 == 0:
                print(f"SVHN train progress: {i}/{len(train_dataset)}")
        
        # Process test set
        test_data = []
        for i, (image, label) in enumerate(test_dataset):
            test_data.append({
                'image': self.image_to_bytes(image),
                'label': label,
                'split': 'test'
            })
            if i % 1000 == 0:
                print(f"SVHN test progress: {i}/{len(test_dataset)}")
        
        # Merge and save
        all_data = train_data + test_data
        df = pd.DataFrame(all_data)
        
        # Save as Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(self.output_dir, 'svhn.parquet'))
        print(f"SVHN conversion complete, total {len(all_data)} records")
    
    def convert_all(self):
        """Convert all datasets"""
        print("Starting conversion of all datasets...")
        
        try:
            self.convert_cifar10()
        except Exception as e:
            print(f"CIFAR-10 conversion failed: {e}")
        
        try:
            self.convert_cifar100()
        except Exception as e:
            print(f"CIFAR-100 conversion failed: {e}")
        
        try:
            self.convert_fashion_mnist()
        except Exception as e:
            print(f"Fashion-MNIST conversion failed: {e}")
        
        try:
            self.convert_svhn()
        except Exception as e:
            print(f"SVHN conversion failed: {e}")
        
        print("All conversions complete!")

def load_parquet_dataset(parquet_path):
    """Load dataset in Parquet format"""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    print(f"Dataset size: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    # Print train/test split counts if available
    if 'split' in df.columns:
        counts = df['split'].value_counts().to_dict()
        train_count = counts.get('train', 0)
        test_count = counts.get('test', 0)
        print(f"Split counts: train={train_count}, test={test_count}")
    # Show info for first few records
    for i in range(min(3, len(df))):
        print(f"Record {i+1}:")
        print(f"  Label: {df.iloc[i]['label']}")
        if 'class_name' in df.columns:
            print(f"  Class name: {df.iloc[i]['class_name']}")
        print(f"  Split: {df.iloc[i]['split']}")
        print(f"  Image size: {len(df.iloc[i]['image'])} bytes")
    
    return df

def image_from_bytes(image_bytes):
    """Convert bytes to PIL Image"""
    buf = io.BytesIO(image_bytes)
    img = Image.open(buf)
    # Ensure image data is fully loaded into memory and detached from the buffer
    img = img.convert('RGB').copy()
    buf.close()
    return img

# Usage example
if __name__ == "__main__":
    # Create converter
    converter = DatasetConverter()
    
    # Convert all datasets
    converter.convert_all()
    
    # Example of loading converted data
    print("\n=== Example of loading converted data ===")
    
    # Load CIFAR-10
    if os.path.exists("./parquet_datasets/cifar10.parquet"):
        print("\nLoading CIFAR-10:")
        cifar10_df = load_parquet_dataset("./parquet_datasets/cifar10.parquet")
        
        # Show first image
        first_image = image_from_bytes(cifar10_df.iloc[0]['image'])
        print(f"First image size: {first_image.size}")
    
    print("\nAll Parquet files have been saved to ./parquet_datasets/ directory")