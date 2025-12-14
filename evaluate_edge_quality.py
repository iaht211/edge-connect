"""
Script đánh giá chất lượng edges và kết quả inpainting một cách toàn diện
Không chỉ dựa vào edge density mà còn xem xét chất lượng thực tế
"""
import os
import torch
import numpy as np
import imageio
from PIL import Image
from src.config import Config
from src.edge_connect import EdgeConnect
from src.dataset import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_metrics(img1, img2):
    """Tính toán các metrics để đánh giá chất lượng"""
    # Chuyển về numpy và normalize về 0-1
    if isinstance(img1, torch.Tensor):
        img1_np = img1.cpu().numpy()
    else:
        img1_np = img1
    
    if isinstance(img2, torch.Tensor):
        img2_np = img2.cpu().numpy()
    else:
        img2_np = img2
    
    # Đảm bảo cùng shape và dtype
    if img1_np.shape != img2_np.shape:
        return None
    
    # Normalize về 0-1
    if img1_np.max() > 1:
        img1_np = img1_np / 255.0
    if img2_np.max() > 1:
        img2_np = img2_np / 255.0
    
    # Chuyển về grayscale nếu là RGB
    if len(img1_np.shape) == 4:  # Batch dimension
        img1_np = img1_np[0]
    if len(img2_np.shape) == 4:
        img2_np = img2_np[0]
    
    if len(img1_np.shape) == 3:
        img1_gray = np.mean(img1_np, axis=0) if img1_np.shape[0] == 3 else np.mean(img1_np, axis=2)
    else:
        img1_gray = img1_np
    
    if len(img2_np.shape) == 3:
        img2_gray = np.mean(img2_np, axis=0) if img2_np.shape[0] == 3 else np.mean(img2_np, axis=2)
    else:
        img2_gray = img2_np
    
    # Kiểm tra kích thước ảnh
    min_dim = min(img1_gray.shape[0], img1_gray.shape[1])
    
    # Tính SSIM với window size phù hợp
    if min_dim < 7:
        # Ảnh quá nhỏ, không thể tính SSIM
        ssim_value = 0.0
    else:
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1  # Phải là số lẻ
        try:
            ssim_value = ssim(img1_gray, img2_gray, data_range=1.0, win_size=win_size)
        except Exception as e:
            print(f"Warning: Could not calculate SSIM: {e}")
            ssim_value = 0.0
    
    # Tính PSNR
    mse = np.mean((img1_gray - img2_gray) ** 2)
    if mse == 0:
        psnr_value = float('inf')
    else:
        try:
            psnr_value = psnr(img1_gray, img2_gray, data_range=1.0)
        except Exception as e:
            print(f"Warning: Could not calculate PSNR: {e}")
            psnr_value = 0.0
    
    return {
        'ssim': ssim_value,
        'psnr': psnr_value,
        'mse': mse
    }


def evaluate_sigma_quality(checkpoints_dir, image_path, mask_path, output_dir, sigma_values=[1, 1.5, 2, 2.5, 3, 4]):
    """Đánh giá chất lượng với các SIGMA khác nhau một cách toàn diện"""
    
    print("=" * 70)
    print("ĐÁNH GIÁ CHẤT LƯỢNG EDGES VÀ INPAINTING")
    print("=" * 70)
    
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
    
    # Load dataset
    test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST,
                          config.TEST_MASK_FLIST, augment=False, training=False)
    
    if len(test_dataset) == 0:
        print("✗ Không tải được ảnh từ dataset")
        return
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    items = next(iter(test_loader))
    images, images_gray, _, masks = model.cuda(*items)
    
    # Lấy ảnh gốc
    images_gray_np = images_gray[0, 0].cpu().numpy()
    masks_np = masks[0, 0].cpu().numpy()
    
    # Normalize về 0-1 nếu cần
    if images_gray_np.max() > 1:
        images_gray_np = images_gray_np / 255.0
    
    mask_bool = (masks_np < 0.5).astype(bool)
    mask_coverage = (masks_np > 0.5).mean()
    
    print(f"\nẢnh: {os.path.basename(image_path)}")
    print(f"Mask coverage: {mask_coverage * 100:.2f}%")
    print(f"\n{'SIGMA':<8} {'Edge Density':<15} {'Edge Quality':<15} {'Inpaint Quality':<20}")
    print("-" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    
    results = []
    
    for sigma in sigma_values:
        from skimage.feature import canny
        
        # Tính Canny edges
        canny_edges = canny(images_gray_np, sigma=sigma, mask=mask_bool).astype(np.float64)
        canny_edges_tensor = torch.from_numpy(canny_edges).float().unsqueeze(0).unsqueeze(0).to(config.DEVICE)
        
        # Chạy edge model
        with torch.no_grad():
            edges_hallucinated = model.edge_model(images_gray, canny_edges_tensor, masks)
        
        edges_merged = (edges_hallucinated * masks) + (canny_edges_tensor * (1 - masks))
        
        # Tính edge density
        mask_region = (masks > 0.5).float()
        edges_in_mask = (edges_hallucinated * mask_region).sum().item()
        edge_density = edges_in_mask / mask_region.sum().item() if mask_region.sum() > 0 else 0
        
        # Đánh giá chất lượng edges (so sánh với Canny edges ở vùng không mask)
        # Edges tốt nên có tính nhất quán với edges ở vùng không mask
        non_mask_region = (masks < 0.5).float()
        canny_in_non_mask = (canny_edges_tensor * non_mask_region).sum().item()
        hallucinated_in_mask = (edges_hallucinated * mask_region).sum().item()
        
        # Tỷ lệ edges giữa vùng mask và không mask (nên tương đương)
        if canny_in_non_mask > 0:
            edge_ratio = hallucinated_in_mask / canny_in_non_mask
        else:
            edge_ratio = 0
        
        # Chạy inpaint model
        with torch.no_grad():
            outputs = model.inpaint_model(images, edges_merged, masks)
        
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        
        # Đánh giá chất lượng inpainting
        # So sánh toàn bộ ảnh (không chỉ vùng mask) để đánh giá tổng thể
        # Hoặc có thể so sánh chỉ vùng được inpaint
        
        # Tính metrics trên toàn bộ ảnh để đánh giá tổng thể
        # (vì chúng ta không có ground truth cho vùng mask)
        metrics = calculate_metrics(images, outputs_merged)
        
        # Đánh giá tổng thể
        edge_quality_score = edge_density * (1 + min(edge_ratio, 1.0))  # Kết hợp density và ratio
        inpaint_quality_score = metrics['ssim'] if metrics else 0
        
        results.append({
            'sigma': sigma,
            'edge_density': edge_density,
            'edge_ratio': edge_ratio,
            'edge_quality_score': edge_quality_score,
            'ssim': metrics['ssim'] if metrics else 0,
            'psnr': metrics['psnr'] if metrics else 0,
            'result': outputs_merged
        })
        
        # In kết quả
        edge_quality_str = f"{edge_quality_score:.4f}"
        inpaint_quality_str = f"SSIM: {metrics['ssim']:.4f}" if metrics else "N/A"
        print(f"{sigma:<8} {edge_density:<15.4f} {edge_quality_str:<15} {inpaint_quality_str:<20}")
    
    # Tìm SIGMA tốt nhất dựa trên nhiều tiêu chí
    print("\n" + "=" * 70)
    print("PHÂN TÍCH VÀ ĐÁNH GIÁ")
    print("=" * 70)
    
    # Tìm SIGMA tốt nhất cho từng tiêu chí
    best_by_density = max(results, key=lambda x: x['edge_density'])
    best_by_ssim = max(results, key=lambda x: x['ssim'])
    best_by_edge_quality = max(results, key=lambda x: x['edge_quality_score'])
    
    # Tính điểm tổng hợp (weighted score)
    for r in results:
        # Normalize các scores về 0-1
        max_density = max(x['edge_density'] for x in results)
        max_ssim = max(x['ssim'] for x in results)
        max_edge_quality = max(x['edge_quality_score'] for x in results)
        
        normalized_density = r['edge_density'] / max_density if max_density > 0 else 0
        normalized_ssim = r['ssim'] / max_ssim if max_ssim > 0 else 0
        normalized_edge_quality = r['edge_quality_score'] / max_edge_quality if max_edge_quality > 0 else 0
        
        # Weighted score: ưu tiên chất lượng inpainting (SSIM) hơn
        r['overall_score'] = (
            0.2 * normalized_density +      # Edge density: 20%
            0.3 * normalized_edge_quality + # Edge quality: 30%
            0.5 * normalized_ssim           # Inpainting quality: 50%
        )
    
    best_overall = max(results, key=lambda x: x['overall_score'])
    
    print(f"\n✓ SIGMA tốt nhất theo Edge Density: {best_by_density['sigma']} (density: {best_by_density['edge_density']:.4f})")
    print(f"✓ SIGMA tốt nhất theo Edge Quality: {best_by_edge_quality['sigma']} (score: {best_by_edge_quality['edge_quality_score']:.4f})")
    print(f"✓ SIGMA tốt nhất theo Inpainting Quality (SSIM): {best_by_ssim['sigma']} (SSIM: {best_by_ssim['ssim']:.4f})")
    print(f"\n🏆 SIGMA TỐT NHẤT TỔNG THỂ: {best_overall['sigma']} (overall score: {best_overall['overall_score']:.4f})")
    
    print("\n" + "-" * 70)
    print("CHI TIẾT TỪNG SIGMA:")
    print("-" * 70)
    print(f"{'SIGMA':<8} {'Density':<12} {'Edge Ratio':<12} {'SSIM':<10} {'Overall':<10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['overall_score'], reverse=True):
        print(f"{r['sigma']:<8} {r['edge_density']:<12.4f} {r['edge_ratio']:<12.4f} {r['ssim']:<10.4f} {r['overall_score']:<10.4f}")
    
    # Đánh giá chất lượng
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ CHẤT LƯỢNG")
    print("=" * 70)
    
    avg_ssim = np.mean([r['ssim'] for r in results])
    max_ssim = max(r['ssim'] for r in results)
    
    if max_ssim > 0.8:
        print("✓ Chất lượng inpainting: RẤT TỐT (SSIM > 0.8)")
    elif max_ssim > 0.6:
        print("⚠ Chất lượng inpainting: TỐT (SSIM > 0.6)")
    elif max_ssim > 0.4:
        print("⚠ Chất lượng inpainting: TRUNG BÌNH (SSIM > 0.4)")
    else:
        print("✗ Chất lượng inpainting: KÉM (SSIM < 0.4)")
    
    if mask_coverage > 0.5:
        print(f"⚠ Cảnh báo: Mask quá lớn ({mask_coverage*100:.1f}%), khó khôi phục tốt")
    
    print(f"\n💡 KHUYẾN NGHỊ:")
    print(f"   Sử dụng SIGMA = {best_overall['sigma']} trong config.yml")
    print(f"   Hoặc thử SIGMA = {best_by_ssim['sigma']} nếu muốn ưu tiên chất lượng inpainting")
    
    # Lưu kết quả tốt nhất
    def save_tensor_image(tensor_img, path):
        img = model.postprocess(tensor_img)[0]
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().astype(np.uint8)
        else:
            img_np = img.astype(np.uint8)
        Image.fromarray(img_np).save(path)
    
    save_tensor_image(best_overall['result'], 
                     os.path.join(output_dir, f'{base_name}_best_sigma{best_overall["sigma"]}.png'))
    save_tensor_image(best_by_ssim['result'], 
                     os.path.join(output_dir, f'{base_name}_best_ssim_sigma{best_by_ssim["sigma"]}.png'))
    
    print(f"\n✓ Đã lưu kết quả tốt nhất vào: {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Đánh giá chất lượng edges và inpainting')
    parser.add_argument('--checkpoints', type=str, required=True,
                       help='Đường dẫn đến thư mục checkpoints')
    parser.add_argument('--image', type=str, required=True,
                       help='Đường dẫn đến ảnh input')
    parser.add_argument('--mask', type=str, required=True,
                       help='Đường dẫn đến mask')
    parser.add_argument('--output', type=str, default='./evaluate_output',
                       help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    evaluate_sigma_quality(args.checkpoints, args.image, args.mask, args.output)


if __name__ == '__main__':
    main()

