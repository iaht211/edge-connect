"""
Script để tạo dữ liệu test cho EdgeConnect từ ảnh gốc
Tạo ảnh đã ghép với mask và ảnh mask tương ứng
"""
import os
import glob
import argparse
import numpy as np
import imageio
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize as sk_resize

# Import utils - hỗ trợ cả relative và absolute import
try:
    from .utils import create_mask
except ImportError:
    from utils import create_mask


def create_mask_image(width, height, mask_type='random', mask_path=None):
    """
    Tạo mask theo các loại khác nhau
    
    Args:
        width: chiều rộng ảnh
        height: chiều cao ảnh
        mask_type: loại mask ('random', 'half', 'external')
        mask_path: đường dẫn đến mask external (nếu mask_type='external')
    
    Returns:
        mask: numpy array mask (0-255, uint8)
    """
    if mask_type == 'random':
        # Random block mask
        mask_width = width // 2
        mask_height = height // 2
        mask = create_mask(width, height, mask_width, mask_height)
        
    elif mask_type == 'half':
        # Half mask (trái hoặc phải)
        side = np.random.choice(['left', 'right'])
        if side == 'left':
            mask = create_mask(width, height, width // 2, height, x=0, y=0)
        else:
            mask = create_mask(width, height, width // 2, height, x=width // 2, y=0)
            
    elif mask_type == 'external':
        # External mask từ file
        if mask_path is None or not os.path.exists(mask_path):
            raise ValueError(f"External mask path không tồn tại: {mask_path}")
        
        mask = imageio.imread(mask_path)
        # Resize mask về kích thước ảnh
        if len(mask.shape) == 3:
            mask = rgb2gray(mask)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        mask = sk_resize(mask, (height, width), preserve_range=True, anti_aliasing=True)
        mask = (mask > 0.5).astype(np.float32)  # Threshold
        
    else:
        raise ValueError(f"Mask type không hợp lệ: {mask_type}. Chọn 'random', 'half', hoặc 'external'")
    
    # Chuyển đổi mask sang uint8 (0-255)
    mask = (mask * 255).astype(np.uint8)
    return mask


def apply_mask_to_image(img, mask):
    """
    Áp dụng mask lên ảnh - vùng mask được fill bằng màu trắng (255)
    
    Args:
        img: numpy array ảnh (H, W, C)
        mask: numpy array mask (H, W) với giá trị 0-255
    
    Returns:
        masked_img: ảnh đã được áp dụng mask
    """
    masked_img = img.copy()
    mask_binary = (mask > 128).astype(bool)  # Threshold mask
    
    # Nếu ảnh là grayscale
    if len(img.shape) == 2:
        masked_img[mask_binary] = 255
    # Nếu ảnh là RGB/RGBA
    else:
        for c in range(img.shape[2]):
            masked_img[:, :, c][mask_binary] = 255
    
    return masked_img


def process_images(input_dir, output_images_dir, output_masks_dir, 
                   mask_type='random', external_mask_dir=None, 
                   prefix='celeba', max_images=None):
    """
    Xử lý ảnh từ thư mục input và tạo dữ liệu test
    
    Args:
        input_dir: thư mục chứa ảnh gốc
        output_images_dir: thư mục lưu ảnh đã mask
        output_masks_dir: thư mục lưu mask
        mask_type: loại mask ('random', 'half', 'external')
        external_mask_dir: thư mục chứa external mask (nếu mask_type='external')
        prefix: prefix cho tên file output
        max_images: số lượng ảnh tối đa để xử lý (None = tất cả)
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_files = glob.glob(os.path.join(input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(input_dir, '*.png')) + \
                  glob.glob(os.path.join(input_dir, '*.JPG')) + \
                  glob.glob(os.path.join(input_dir, '*.PNG'))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return
    
    # Giới hạn số lượng ảnh nếu cần
    if max_images is not None:
        image_files = image_files[:max_images]
    
    # Lấy danh sách external mask nếu cần
    external_masks = None
    if mask_type == 'external' and external_mask_dir:
        external_masks = glob.glob(os.path.join(external_mask_dir, '*.jpg')) + \
                        glob.glob(os.path.join(external_mask_dir, '*.png'))
        external_masks.sort()
        if len(external_masks) == 0:
            print(f"Cảnh báo: Không tìm thấy external mask trong {external_mask_dir}")
            print("Chuyển sang sử dụng random mask")
            mask_type = 'random'
    
    print(f"Bắt đầu xử lý {len(image_files)} ảnh với mask type: {mask_type}")
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Đọc ảnh
            img = imageio.imread(img_path)
            
            # Đảm bảo ảnh là RGB
            if len(img.shape) == 2:
                # Grayscale -> RGB
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                # RGBA -> RGB
                img = img[:, :, :3]
            
            height, width = img.shape[:2]
            
            # Tạo mask
            if mask_type == 'external' and external_masks:
                # Sử dụng mask tương ứng hoặc random
                mask_idx = (idx - 1) % len(external_masks)
                mask = create_mask_image(width, height, mask_type='external', 
                                        mask_path=external_masks[mask_idx])
            else:
                mask = create_mask_image(width, height, mask_type=mask_type)
            
            # Áp dụng mask lên ảnh
            masked_img = apply_mask_to_image(img, mask)
            
            # Tạo tên file output
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_name = f"{prefix}_{idx:02d}.png"
            
            # Lưu ảnh đã mask
            output_img_path = os.path.join(output_images_dir, output_name)
            imageio.imwrite(output_img_path, masked_img)
            
            # Lưu mask
            output_mask_path = os.path.join(output_masks_dir, output_name)
            imageio.imwrite(output_mask_path, mask)
            
            print(f"[{idx}/{len(image_files)}] Đã xử lý: {base_name} -> {output_name}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {str(e)}")
            continue
    
    print(f"\nHoàn thành! Đã tạo {len(image_files)} cặp ảnh và mask")
    print(f"Ảnh đã mask: {output_images_dir}")
    print(f"Mask: {output_masks_dir}")


def main():
    parser = argparse.ArgumentParser(description='Tạo dữ liệu test cho EdgeConnect từ ảnh gốc')
    parser.add_argument('--input', type=str, required=True,
                       help='Thư mục chứa ảnh gốc')
    parser.add_argument('--output-images', type=str, 
                       default='./examples/celeba/images',
                       help='Thư mục lưu ảnh đã mask (default: ./examples/celeba/images)')
    parser.add_argument('--output-masks', type=str,
                       default='./examples/celeba/masks',
                       help='Thư mục lưu mask (default: ./examples/celeba/masks)')
    parser.add_argument('--mask-type', type=str, choices=['random', 'half', 'external'],
                       default='random',
                       help='Loại mask: random, half, hoặc external (default: random)')
    parser.add_argument('--external-mask-dir', type=str, default=None,
                       help='Thư mục chứa external mask (chỉ dùng khi mask-type=external)')
    parser.add_argument('--prefix', type=str, default='celeba',
                       help='Prefix cho tên file output (default: celeba)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Số lượng ảnh tối đa để xử lý (default: tất cả)')
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input,
        output_images_dir=args.output_images,
        output_masks_dir=args.output_masks,
        mask_type=args.mask_type,
        external_mask_dir=args.external_mask_dir,
        prefix=args.prefix,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()

