"""
Script để cải thiện chất lượng edges bằng cách:
1. Thử các giá trị SIGMA khác nhau cho Canny edge detector
2. Kiểm tra và so sánh chất lượng edges
3. Đề xuất tham số tốt nhất
"""
import os
import torch
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray
from src.config import Config
from src.edge_connect import EdgeConnect
from src.dataset import Dataset
from torch.utils.data import DataLoader


def test_different_sigma(checkpoints_dir, image_path, mask_path, output_dir, sigma_values=[1, 2, 3, 4]):
    """Test với các giá trị SIGMA khác nhau"""
    
    print("=" * 60)
    print("TEST CÁC GIÁ TRỊ SIGMA KHÁC NHAU CHO CANNY EDGE DETECTOR")
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
    
    # Load model
    model = EdgeConnect(config)
    model.load()
    model.edge_model.eval()
    model.inpaint_model.eval()
    
    # Load dataset với SIGMA mặc định
    test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST,
                          config.TEST_MASK_FLIST, augment=False, training=False)
    
    if len(test_dataset) == 0:
        print("✗ Không tải được ảnh từ dataset")
        return
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    items = next(iter(test_loader))
    images, images_gray, _, masks = model.cuda(*items)
    
    # Lấy ảnh gốc để tính Canny edges với các SIGMA khác nhau
    images_np = images[0].cpu().permute(1, 2, 0).numpy()
    images_gray_np = images_gray[0, 0].cpu().numpy()
    masks_np = masks[0, 0].cpu().numpy()
    
    # Normalize về 0-1 nếu cần
    if images_gray_np.max() > 1:
        images_gray_np = images_gray_np / 255.0
    
    # Mask để loại bỏ vùng mask khi tính Canny
    mask_bool = (masks_np < 0.5).astype(bool)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    
    print(f"\nẢnh: {base_name}")
    print(f"Mask coverage: {(masks_np > 0.5).mean() * 100:.2f}%")
    print("\n" + "-" * 60)
    
    best_sigma = None
    best_score = -1
    
    results = []
    
    for sigma in sigma_values:
        print(f"\nTesting SIGMA = {sigma}")
        
        # Tính Canny edges với SIGMA này
        canny_edges = canny(images_gray_np, sigma=sigma, mask=mask_bool).astype(np.float64)
        
        # Chuyển sang tensor
        canny_edges_tensor = torch.from_numpy(canny_edges).float().unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        
        # Chạy edge model
        with torch.no_grad():
            edges_hallucinated = model.edge_model(images_gray, canny_edges_tensor, masks)
        
        edges_merged = (edges_hallucinated * masks) + (canny_edges_tensor * (1 - masks))
        
        # Đánh giá chất lượng edges
        # Tính số lượng edges trong vùng mask
        mask_region = (masks > 0.5).float()
        edges_in_mask = (edges_hallucinated * mask_region).sum().item()
        edge_density = edges_in_mask / mask_region.sum().item() if mask_region.sum() > 0 else 0
        
        print(f"  Edge density in mask region: {edge_density:.4f}")
        
        # Chạy inpaint model với edges này
        with torch.no_grad():
            outputs = model.inpaint_model(images, edges_merged, masks)
        
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        
        # Lưu kết quả
        def save_tensor_image(tensor_img, path, is_edge=False):
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
        
        # Lưu Canny edges
        canny_3ch = canny_edges_tensor.repeat(1, 3, 1, 1) if canny_edges_tensor.shape[1] == 1 else canny_edges_tensor
        save_tensor_image(1 - canny_3ch, 
                         os.path.join(output_dir, f'{base_name}_canny_sigma{sigma}.png'), 
                         is_edge=True)
        
        # Lưu hallucinated edges
        edges_merged_3ch = edges_merged.repeat(1, 3, 1, 1) if edges_merged.shape[1] == 1 else edges_merged
        save_tensor_image(1 - edges_merged_3ch, 
                         os.path.join(output_dir, f'{base_name}_hallucinated_sigma{sigma}.png'), 
                         is_edge=True)
        
        # Lưu kết quả inpainting
        save_tensor_image(outputs_merged, 
                         os.path.join(output_dir, f'{base_name}_result_sigma{sigma}.png'))
        
        results.append({
            'sigma': sigma,
            'edge_density': edge_density,
            'canny_edges': canny_edges_tensor,
            'hallucinated_edges': edges_hallucinated,
            'result': outputs_merged
        })
        
        if edge_density > best_score:
            best_score = edge_density
            best_sigma = sigma
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"\nSIGMA tốt nhất: {best_sigma} (edge density: {best_score:.4f})")
    print("\nSo sánh các SIGMA:")
    for r in results:
        print(f"  SIGMA {r['sigma']}: Edge density = {r['edge_density']:.4f}")
    
    print(f"\n✓ Tất cả kết quả đã được lưu vào: {output_dir}")
    print(f"\nGợi ý: Thử chỉnh SIGMA trong config.yml thành {best_sigma} và chạy lại")


def improve_edge_quality(checkpoints_dir, image_path, mask_path, output_dir):
    """Cải thiện chất lượng edges bằng nhiều phương pháp"""
    
    print("=" * 60)
    print("CẢI THIỆN CHẤT LƯỢNG EDGES")
    print("=" * 60)
    
    # Test với các SIGMA khác nhau
    print("\n1. Testing với các giá trị SIGMA khác nhau...")
    test_different_sigma(checkpoints_dir, image_path, mask_path, 
                        os.path.join(output_dir, 'sigma_test'),
                        sigma_values=[1, 1.5, 2, 2.5, 3, 4])
    
    # Kiểm tra mask coverage
    mask = imageio.imread(mask_path)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask_coverage = (mask > 128).mean()
    
    print("\n" + "=" * 60)
    print("PHÂN TÍCH")
    print("=" * 60)
    print(f"Mask coverage: {mask_coverage * 100:.2f}%")
    
    if mask_coverage > 0.5:
        print("⚠ Cảnh báo: Mask quá lớn (>50%), khó khôi phục edges tốt")
        print("   Gợi ý: Thử với mask nhỏ hơn hoặc chia nhỏ mask thành nhiều phần")
    
    if mask_coverage < 0.1:
        print("⚠ Cảnh báo: Mask quá nhỏ (<10%), có thể không cần edge model")
    
    print("\n" + "=" * 60)
    print("CÁC PHƯƠNG PHÁP CẢI THIỆN")
    print("=" * 60)
    print("1. Điều chỉnh SIGMA trong config.yml (thử 1-4)")
    print("2. Sử dụng external edges nếu có (EDGE=2 trong config)")
    print("3. Giảm kích thước mask nếu quá lớn")
    print("4. Đảm bảo Canny edges ban đầu có chất lượng tốt")
    print("5. Kiểm tra model đã được train đúng với dataset chưa")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cải thiện chất lượng edges')
    parser.add_argument('--checkpoints', type=str, required=True,
                       help='Đường dẫn đến thư mục checkpoints')
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh input')
    parser.add_argument('--mask', type=str, required=True,
                       help='Đường dẫn đến mask')
    parser.add_argument('--output', type=str, default='./improve_edges_output',
                       help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    improve_edge_quality(args.checkpoints, args.image, args.mask, args.output)


if __name__ == '__main__':
    main()

