"""
Script test một ảnh cụ thể và hiển thị chi tiết từng bước để debug
"""
import os
import torch
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from src.config import Config
from src.edge_connect import EdgeConnect
from src.dataset import Dataset
from torch.utils.data import DataLoader


def test_single_image(checkpoints_dir, image_path, mask_path, output_dir):
    """Test một ảnh cụ thể và hiển thị chi tiết"""
    
    print("=" * 60)
    print("TEST MỘT ẢNH CỤ THỂ")
    print("=" * 60)
    
    # Load config
    config_path = os.path.join(checkpoints_dir, 'config.yml')
    config = Config(config_path)
    config.MODE = 2
    config.MODEL = 3
    config.TEST_FLIST = image_path
    config.TEST_MASK_FLIST = mask_path
    config.INPUT_SIZE = 0
    config.PATH = checkpoints_dir
    
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")
    
    print(f"Device: {config.DEVICE}")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    
    # Load model
    model = EdgeConnect(config)
    model.load()
    model.edge_model.eval()
    model.inpaint_model.eval()
    
    # Load dataset
    test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST,
                          config.TEST_MASK_FLIST, augment=False, training=False)
    
    if len(test_dataset) == 0:
        print("✗ Không tải được ảnh từ dataset")
        return
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    # Lấy một batch
    items = next(iter(test_loader))
    images, images_gray, edges_canny, masks = model.cuda(*items)
    
    print("\n" + "-" * 60)
    print("THÔNG TIN INPUT")
    print("-" * 60)
    print(f"Images shape: {images.shape}, dtype: {images.dtype}, min: {images.min():.3f}, max: {images.max():.3f}")
    print(f"Images_gray shape: {images_gray.shape}, dtype: {images_gray.dtype}, min: {images_gray.min():.3f}, max: {images_gray.max():.3f}")
    print(f"Edges_canny shape: {edges_canny.shape}, dtype: {edges_canny.dtype}, min: {edges_canny.min():.3f}, max: {edges_canny.max():.3f}")
    print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}, min: {masks.min():.3f}, max: {masks.max():.3f}")
    
    # Kiểm tra mask coverage
    mask_coverage = (masks > 0.5).float().mean().item()
    print(f"Mask coverage: {mask_coverage * 100:.2f}%")
    
    if mask_coverage > 0.5:
        print("⚠ Cảnh báo: Mask quá lớn (>50%), có thể ảnh hưởng đến chất lượng")
    
    # Stage 1: Edge model
    print("\n" + "-" * 60)
    print("STAGE 1: EDGE MODEL")
    print("-" * 60)
    
    with torch.no_grad():
        edges_hallucinated = model.edge_model(images_gray, edges_canny, masks)
    
    print(f"Edges hallucinated shape: {edges_hallucinated.shape}")
    print(f"Edges hallucinated min: {edges_hallucinated.min():.3f}, max: {edges_hallucinated.max():.3f}")
    
    edges_merged = (edges_hallucinated * masks) + (edges_canny * (1 - masks))
    print(f"Edges merged min: {edges_merged.min():.3f}, max: {edges_merged.max():.3f}")
    
    # Stage 2: Inpaint model với Canny edges
    print("\n" + "-" * 60)
    print("STAGE 2: INPAINT MODEL (với Canny edges)")
    print("-" * 60)
    
    with torch.no_grad():
        outputs_canny = model.inpaint_model(images, edges_canny, masks)
    
    print(f"Outputs canny shape: {outputs_canny.shape}")
    print(f"Outputs canny min: {outputs_canny.min():.3f}, max: {outputs_canny.max():.3f}")
    
    # Stage 3: Inpaint model với hallucinated edges
    print("\n" + "-" * 60)
    print("STAGE 3: INPAINT MODEL (với hallucinated edges)")
    print("-" * 60)
    
    with torch.no_grad():
        outputs_final = model.inpaint_model(images, edges_merged, masks)
    
    print(f"Outputs final shape: {outputs_final.shape}")
    print(f"Outputs final min: {outputs_final.min():.3f}, max: {outputs_final.max():.3f}")
    
    # Merge outputs
    outputs_merged_canny = (outputs_canny * masks) + (images * (1 - masks))
    outputs_merged_final = (outputs_final * masks) + (images * (1 - masks))
    
    # Tính toán metrics
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    
    def calculate_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # So sánh với ảnh gốc (nếu có)
    # PSNR chỉ có ý nghĩa nếu có ground truth
    
    # Lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    
    def save_tensor_image(tensor_img, path, is_edge=False):
        """Lưu tensor image"""
        img = model.postprocess(tensor_img)[0]
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().astype(np.uint8)
        else:
            img_np = img.astype(np.uint8)
        
        if is_edge and len(img_np.shape) == 3 and img_np.shape[2] == 1:
            img_np = img_np[:, :, 0]
            Image.fromarray(img_np, mode='L').save(path)
        else:
            Image.fromarray(img_np).save(path)
    
    print("\n" + "-" * 60)
    print("LƯU KẾT QUẢ")
    print("-" * 60)
    
    base_name = os.path.basename(image_path).split('.')[0]
    
    save_tensor_image(images, os.path.join(output_dir, f'{base_name}_original.png'))
    print(f"✓ Đã lưu: {base_name}_original.png")
    
    save_tensor_image((images * (1 - masks) + masks), os.path.join(output_dir, f'{base_name}_masked.png'))
    print(f"✓ Đã lưu: {base_name}_masked.png")
    
    edges_canny_3ch = edges_canny.repeat(1, 3, 1, 1) if edges_canny.shape[1] == 1 else edges_canny
    save_tensor_image(1 - edges_canny_3ch, os.path.join(output_dir, f'{base_name}_canny_edges.png'), is_edge=True)
    print(f"✓ Đã lưu: {base_name}_canny_edges.png")
    
    edges_merged_3ch = edges_merged.repeat(1, 3, 1, 1) if edges_merged.shape[1] == 1 else edges_merged
    save_tensor_image(1 - edges_merged_3ch, os.path.join(output_dir, f'{base_name}_hallucinated_edges.png'), is_edge=True)
    print(f"✓ Đã lưu: {base_name}_hallucinated_edges.png")
    
    save_tensor_image(outputs_merged_canny, os.path.join(output_dir, f'{base_name}_result_canny.png'))
    print(f"✓ Đã lưu: {base_name}_result_canny.png")
    
    save_tensor_image(outputs_merged_final, os.path.join(output_dir, f'{base_name}_result_final.png'))
    print(f"✓ Đã lưu: {base_name}_result_final.png")
    
    print(f"\n✓ Tất cả kết quả đã được lưu vào: {output_dir}")
    
    # Gợi ý
    print("\n" + "=" * 60)
    print("GỢI Ý ĐỂ CẢI THIỆN CHẤT LƯỢNG:")
    print("=" * 60)
    print("1. Kiểm tra mask coverage - không nên quá lớn")
    print("2. So sánh kết quả với Canny edges và hallucinated edges")
    print("3. Đảm bảo model đã được load đúng (kiểm tra file .pth)")
    print("4. Thử với ảnh có độ phân giải khác nhau")
    print("5. Kiểm tra xem model có phù hợp với dataset không (CelebA vs Places2)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test một ảnh cụ thể để debug')
    parser.add_argument('--checkpoints', type=str, required=True,
                       help='Đường dẫn đến thư mục checkpoints')
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh input')
    parser.add_argument('--mask', type=str, required=True,
                       help='Đường dẫn đến mask')
    parser.add_argument('--output', type=str, default='./debug_output',
                       help='Thư mục lưu kết quả debug')
    
    args = parser.parse_args()
    
    test_single_image(args.checkpoints, args.image, args.mask, args.output)


if __name__ == '__main__':
    main()

