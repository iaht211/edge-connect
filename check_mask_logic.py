"""
Script để kiểm tra logic mask và debug vấn đề
"""
import os
import torch
import numpy as np
import imageio
from PIL import Image
import torchvision.transforms.functional as F
from src.config import Config
from src.dataset import Dataset
from torch.utils.data import DataLoader


def check_mask_logic(checkpoints_dir, image_path, mask_path):
    """Kiểm tra logic mask từ đầu đến cuối"""
    
    print("=" * 70)
    print("KIỂM TRA LOGIC MASK")
    print("=" * 70)
    
    # 1. Đọc mask file gốc
    print("\n1. ĐỌC MASK FILE GỐC:")
    print("-" * 70)
    mask_original = imageio.imread(mask_path)
    print(f"   Shape: {mask_original.shape}")
    print(f"   Dtype: {mask_original.dtype}")
    print(f"   Min: {mask_original.min()}, Max: {mask_original.max()}")
    print(f"   Unique values: {np.unique(mask_original)[:10]}...")
    
    # Kiểm tra giá trị
    white_pixels = (mask_original > 128).sum() if len(mask_original.shape) == 2 else (mask_original[:, :, 0] > 128).sum()
    black_pixels = (mask_original <= 128).sum() if len(mask_original.shape) == 2 else (mask_original[:, :, 0] <= 128).sum()
    total_pixels = mask_original.shape[0] * mask_original.shape[1]
    
    print(f"   White pixels (>128): {white_pixels} ({white_pixels/total_pixels*100:.2f}%)")
    print(f"   Black pixels (<=128): {black_pixels} ({black_pixels/total_pixels*100:.2f}%)")
    print(f"   → White = vùng cần khôi phục, Black = vùng giữ nguyên")
    
    # 2. Xem cách dataset xử lý mask
    print("\n2. XỬ LÝ MASK TRONG DATASET:")
    print("-" * 70)
    
    config_path = os.path.join(checkpoints_dir, 'config.yml')
    config = Config(config_path)
    config.MODE = 2
    config.TEST_FLIST = image_path
    config.TEST_MASK_FLIST = mask_path
    config.INPUT_SIZE = 0
    config.PATH = checkpoints_dir
    
    test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST,
                          config.TEST_MASK_FLIST, augment=False, training=False)
    
    if len(test_dataset) == 0:
        print("✗ Không tải được dataset")
        return
    
    # Lấy một item
    item = test_dataset[0]
    images, images_gray, edges, masks = item
    
    print(f"   Mask tensor shape: {masks.shape}")
    print(f"   Mask tensor dtype: {masks.dtype}")
    print(f"   Mask tensor min: {masks.min():.4f}, max: {masks.max():.4f}")
    
    # Chuyển về numpy để kiểm tra
    mask_np = masks[0].numpy() if len(masks.shape) == 3 else masks.numpy()
    mask_white_region = (mask_np > 0.5).sum()
    mask_black_region = (mask_np <= 0.5).sum()
    
    print(f"   Mask > 0.5 (white region): {mask_white_region} pixels")
    print(f"   Mask <= 0.5 (black region): {mask_black_region} pixels")
    print(f"   → Sau normalize: 1.0 = vùng cần khôi phục, 0.0 = vùng giữ nguyên")
    
    # 3. Kiểm tra cách merge kết quả
    print("\n3. LOGIC MERGE KẾT QUẢ:")
    print("-" * 70)
    print("   Công thức: outputs_merged = (outputs * masks) + (images * (1 - masks))")
    print("   - Nếu mask = 1.0 (white): dùng outputs (kết quả inpainting)")
    print("   - Nếu mask = 0.0 (black): dùng images (ảnh gốc)")
    
    # Tạo dummy outputs để test
    dummy_outputs = torch.ones_like(images) * 0.5  # Màu xám để dễ nhận biết
    
    # Merge
    outputs_merged = (dummy_outputs * masks) + (images * (1 - masks))
    
    # Kiểm tra - xử lý đúng số chiều
    if len(outputs_merged.shape) == 4:  # [B, C, H, W]
        merged_np = outputs_merged[0].permute(1, 2, 0).numpy()
        original_np = images[0].permute(1, 2, 0).numpy()
    elif len(outputs_merged.shape) == 3:  # [C, H, W]
        merged_np = outputs_merged.permute(1, 2, 0).numpy()
        original_np = images.permute(1, 2, 0).numpy()
    else:
        merged_np = outputs_merged.numpy()
        original_np = images.numpy()
    
    # Xử lý mask để có 3 channels cho overlay
    if len(masks.shape) == 4:  # [B, C, H, W]
        mask_single = masks[0]
        if mask_single.shape[0] == 1:  # [1, H, W]
            mask_2d = mask_single[0].numpy()
        else:  # [C, H, W]
            mask_2d = mask_single[0].numpy() if mask_single.shape[0] >= 1 else mask_single.numpy()
    elif len(masks.shape) == 3:  # [C, H, W]
        if masks.shape[0] == 1:  # [1, H, W]
            mask_2d = masks[0].numpy()
        else:  # [C, H, W] với C > 1
            mask_2d = masks[0].numpy()
    else:  # [H, W]
        mask_2d = masks.numpy()
    
    # Tạo mask 3D từ mask 2D
    mask_3d = np.stack([mask_2d] * 3, axis=2)
    
    # So sánh
    diff = np.abs(merged_np - original_np)
    changed_pixels = (diff > 0.01).sum()
    
    print(f"   Pixels thay đổi (dùng outputs): {changed_pixels}")
    print(f"   Pixels giữ nguyên (dùng images): {diff.size - changed_pixels}")
    
    # 4. Kiểm tra xem có vấn đề gì không
    print("\n4. PHÁT HIỆN VẤN ĐỀ:")
    print("-" * 70)
    
    issues = []
    
    # Kiểm tra mask có đúng format không
    if mask_original.max() > 255 or mask_original.min() < 0:
        issues.append("✗ Mask có giá trị ngoài phạm vi [0, 255]")
    
    # Kiểm tra mask có đúng 2 giá trị không (0 và 255)
    unique_vals = np.unique(mask_original)
    if len(unique_vals) > 10:
        issues.append(f"⚠ Mask có nhiều giá trị khác nhau ({len(unique_vals)} giá trị), nên threshold rõ ràng hơn")
    
    # Kiểm tra mask coverage
    if white_pixels / total_pixels > 0.7:
        issues.append(f"⚠ Mask quá lớn ({white_pixels/total_pixels*100:.1f}%), khó khôi phục")
    elif white_pixels / total_pixels < 0.05:
        issues.append(f"⚠ Mask quá nhỏ ({white_pixels/total_pixels*100:.1f}%), có thể không cần inpainting")
    
    # Kiểm tra xem mask có bị đảo ngược không
    # Nếu mask có nhiều white hơn black, có thể đúng
    # Nhưng cần xem kết quả thực tế
    
    if len(issues) == 0:
        print("   ✓ Không phát hiện vấn đề rõ ràng")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    # 5. Lưu ảnh để kiểm tra trực quan
    print("\n5. LƯU ẢNH KIỂM TRA:")
    print("-" * 70)
    
    output_dir = "./mask_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu mask gốc
    if len(mask_original.shape) == 2:
        Image.fromarray(mask_original, mode='L').save(os.path.join(output_dir, "mask_original.png"))
    else:
        Image.fromarray(mask_original).save(os.path.join(output_dir, "mask_original.png"))
    print(f"   ✓ Đã lưu: mask_original.png")
    
    # Lưu mask sau normalize
    mask_normalized = (mask_np * 255).astype(np.uint8)
    if len(mask_normalized.shape) == 2:
        Image.fromarray(mask_normalized, mode='L').save(os.path.join(output_dir, "mask_normalized.png"))
    else:
        Image.fromarray(mask_normalized[:, :, 0], mode='L').save(os.path.join(output_dir, "mask_normalized.png"))
    print(f"   ✓ Đã lưu: mask_normalized.png")
    
    # Lưu ảnh gốc
    if len(images.shape) == 4:
        img_np = (images[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    elif len(images.shape) == 3:
        img_np = (images.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img_np = (images.numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(os.path.join(output_dir, "image_original.png"))
    print(f"   ✓ Đã lưu: image_original.png")
    
    # Lưu ảnh với mask overlay (màu đỏ ở vùng mask)
    img_with_mask = img_np.copy()
    mask_binary = (mask_np > 0.5)
    if len(mask_binary.shape) == 2:
        img_with_mask[mask_binary, 0] = 255  # Red channel
        img_with_mask[mask_binary, 1] = 0
        img_with_mask[mask_binary, 2] = 0
    Image.fromarray(img_with_mask).save(os.path.join(output_dir, "image_with_mask_overlay.png"))
    print(f"   ✓ Đã lưu: image_with_mask_overlay.png (màu đỏ = vùng mask)")
    
    # Lưu ảnh test merge
    if len(outputs_merged.shape) == 4:
        merged_test = (outputs_merged[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    elif len(outputs_merged.shape) == 3:
        merged_test = (outputs_merged.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        merged_test = (outputs_merged.numpy() * 255).astype(np.uint8)
    Image.fromarray(merged_test).save(os.path.join(output_dir, "test_merge.png"))
    print(f"   ✓ Đã lưu: test_merge.png (màu xám = vùng dùng outputs, màu gốc = vùng giữ nguyên)")
    
    print(f"\n✓ Tất cả ảnh debug đã được lưu vào: {output_dir}")
    print("\n" + "=" * 70)
    print("KẾT LUẬN:")
    print("=" * 70)
    print("1. Kiểm tra ảnh 'image_with_mask_overlay.png' - màu đỏ phải ở vùng cần khôi phục")
    print("2. Kiểm tra ảnh 'test_merge.png' - màu xám phải ở vùng cần khôi phục")
    print("3. Nếu màu đỏ/xám ở sai vị trí, mask có thể bị đảo ngược")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kiểm tra logic mask')
    parser.add_argument('--checkpoints', type=str, required=True,
                       help='Đường dẫn đến thư mục checkpoints')
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh input')
    parser.add_argument('--mask', type=str, required=True,
                       help='Đường dẫn đến mask')
    
    args = parser.parse_args()
    
    check_mask_logic(args.checkpoints, args.image, args.mask)


if __name__ == '__main__':
    main()

