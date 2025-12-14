"""
Script kiểm tra setup và debug các vấn đề có thể ảnh hưởng đến chất lượng kết quả
"""
import os
import torch
import numpy as np
from PIL import Image
import imageio


def check_model_files(checkpoints_dir):
    """Kiểm tra các file model có tồn tại không"""
    print("=" * 60)
    print("KIỂM TRA MODEL FILES")
    print("=" * 60)
    
    required_files = {
        'edge': ['EdgeModel_gen.pth'],
        'inpaint': ['InpaintingModel_gen.pth'],
        'both': ['EdgeModel_gen.pth', 'InpaintingModel_gen.pth']
    }
    
    files_found = []
    files_missing = []
    
    # Kiểm tra edge model
    edge_file = os.path.join(checkpoints_dir, 'EdgeModel_gen.pth')
    if os.path.exists(edge_file):
        files_found.append(('EdgeModel_gen.pth', edge_file))
        size = os.path.getsize(edge_file) / (1024 * 1024)  # MB
        print(f"✓ EdgeModel_gen.pth: {size:.2f} MB")
    else:
        files_missing.append(('EdgeModel_gen.pth', edge_file))
        print(f"✗ EdgeModel_gen.pth: KHÔNG TỒN TẠI")
    
    # Kiểm tra inpaint model
    inpaint_file = os.path.join(checkpoints_dir, 'InpaintingModel_gen.pth')
    if os.path.exists(inpaint_file):
        files_found.append(('InpaintingModel_gen.pth', inpaint_file))
        size = os.path.getsize(inpaint_file) / (1024 * 1024)  # MB
        print(f"✓ InpaintingModel_gen.pth: {size:.2f} MB")
    else:
        files_missing.append(('InpaintingModel_gen.pth', inpaint_file))
        print(f"✗ InpaintingModel_gen.pth: KHÔNG TỒN TẠI")
    
    # Kiểm tra config
    config_file = os.path.join(checkpoints_dir, 'config.yml')
    if os.path.exists(config_file):
        print(f"✓ config.yml: TỒN TẠI")
    else:
        print(f"✗ config.yml: KHÔNG TỒN TẠI")
    
    print()
    return len(files_missing) == 0


def check_input_images(input_dir, mask_dir):
    """Kiểm tra ảnh input và mask"""
    print("=" * 60)
    print("KIỂM TRA INPUT IMAGES VÀ MASKS")
    print("=" * 60)
    
    import glob
    
    # Kiểm tra ảnh
    image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(input_dir, '*.png'))
    
    if len(image_files) == 0:
        print(f"✗ Không tìm thấy ảnh trong: {input_dir}")
        return False
    
    print(f"✓ Tìm thấy {len(image_files)} ảnh")
    
    # Kiểm tra mask
    mask_files = glob.glob(os.path.join(mask_dir, '*.jpg')) + \
                 glob.glob(os.path.join(mask_dir, '*.png'))
    
    if len(mask_files) == 0:
        print(f"✗ Không tìm thấy mask trong: {mask_dir}")
        return False
    
    print(f"✓ Tìm thấy {len(mask_files)} mask")
    
    # Kiểm tra số lượng khớp
    if len(image_files) != len(mask_files):
        print(f"⚠ Cảnh báo: Số lượng ảnh ({len(image_files)}) và mask ({len(mask_files)}) không khớp")
    
    # Kiểm tra một vài ảnh mẫu
    print("\nKiểm tra ảnh mẫu:")
    for i, img_path in enumerate(image_files[:3]):
        try:
            img = imageio.imread(img_path)
            print(f"  [{i+1}] {os.path.basename(img_path)}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        except Exception as e:
            print(f"  ✗ Lỗi đọc {img_path}: {e}")
    
    print("\nKiểm tra mask mẫu:")
    for i, mask_path in enumerate(mask_files[:3]):
        try:
            mask = imageio.imread(mask_path)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] if mask.shape[2] >= 1 else mask
            print(f"  [{i+1}] {os.path.basename(mask_path)}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}")
            # Kiểm tra mask có đúng format không (0-255, vùng mask = 255)
            unique_values = np.unique(mask)
            print(f"      Unique values: {unique_values[:10]}...")  # Hiển thị 10 giá trị đầu
        except Exception as e:
            print(f"  ✗ Lỗi đọc {mask_path}: {e}")
    
    print()
    return True


def check_device():
    """Kiểm tra device (CPU/GPU)"""
    print("=" * 60)
    print("KIỂM TRA DEVICE")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("⚠ CUDA không available, sẽ sử dụng CPU (chậm hơn)")
    
    print()


def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
    print("=" * 60)
    print("KIỂM TRA DEPENDENCIES")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'imageio': 'imageio',
        'skimage': 'scikit-image',
        'yaml': 'PyYAML',
        'cv2': 'opencv-python'
    }
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'PIL':
                import PIL
                version = PIL.__version__
            elif module_name == 'cv2':
                import cv2
                version = cv2.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package_name}: {version}")
        except ImportError:
            print(f"✗ {package_name}: CHƯA CÀI ĐẶT")
    
    print()


def test_model_loading(checkpoints_dir):
    """Test load model"""
    print("=" * 60)
    print("KIỂM TRA LOAD MODEL")
    print("=" * 60)
    
    try:
        from src.config import Config
        from src.edge_connect import EdgeConnect
        
        config_path = os.path.join(checkpoints_dir, 'config.yml')
        if not os.path.exists(config_path):
            print(f"✗ Không tìm thấy config.yml tại {config_path}")
            return False
        
        config = Config(config_path)
        config.MODE = 2
        config.MODEL = 3
        config.PATH = checkpoints_dir
        
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
        else:
            config.DEVICE = torch.device("cpu")
        
        print(f"  Device: {config.DEVICE}")
        print(f"  Model type: {config.MODEL} (3 = edge-inpaint)")
        
        # Tạo model
        model = EdgeConnect(config)
        
        # Load model
        print("  Đang load model...")
        model.load()
        
        print("✓ Model đã load thành công!")
        
        # Kiểm tra model đã được load chưa
        model.edge_model.eval()
        model.inpaint_model.eval()
        
        print("✓ Model đã chuyển sang eval mode")
        
        return True
        
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kiểm tra setup EdgeConnect')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/celeba',
                       help='Đường dẫn đến thư mục checkpoints')
    parser.add_argument('--input', type=str, default='./examples/celeba/images',
                       help='Đường dẫn đến thư mục ảnh input')
    parser.add_argument('--mask', type=str, default='./examples/celeba/masks',
                       help='Đường dẫn đến thư mục mask')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("KIỂM TRA SETUP EDGECONNECT")
    print("=" * 60 + "\n")
    
    all_ok = True
    
    # Kiểm tra dependencies
    check_dependencies()
    
    # Kiểm tra device
    check_device()
    
    # Kiểm tra model files
    if not check_model_files(args.checkpoints):
        all_ok = False
        print("⚠ CẢNH BÁO: Thiếu một số file model!")
        print("   Hãy đảm bảo đã tải đầy đủ pre-trained models")
        print()
    
    # Kiểm tra input images
    if not check_input_images(args.input, args.mask):
        all_ok = False
        print("⚠ CẢNH BÁO: Có vấn đề với input images hoặc masks!")
        print()
    
    # Test load model
    if not test_model_loading(args.checkpoints):
        all_ok = False
        print("⚠ CẢNH BÁO: Không thể load model!")
        print()
    
    print("=" * 60)
    if all_ok:
        print("✓ TẤT CẢ KIỂM TRA ĐỀU PASS!")
        print("\nGợi ý để cải thiện chất lượng kết quả:")
        print("1. Đảm bảo mask đúng format: vùng mask = 255 (trắng), vùng không mask = 0 (đen)")
        print("2. Ảnh input nên có độ phân giải tốt (ít nhất 256x256)")
        print("3. Mask không nên quá lớn (không quá 50% diện tích ảnh)")
        print("4. Sử dụng model phù hợp với dataset (CelebA cho ảnh người, Places2 cho cảnh vật)")
    else:
        print("✗ CÓ MỘT SỐ VẤN ĐỀ CẦN KHẮC PHỤC!")
    print("=" * 60)


if __name__ == '__main__':
    main()

