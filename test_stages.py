"""
Script để test và hiển thị kết quả từng giai đoạn của EdgeConnect:
1. Ảnh gốc với mask
2. Edge map ban đầu (Canny)
3. Edge map sau edge model (hallucinated edges)
4. Kết quả inpainting từ edge model
5. Kết quả cuối cùng từ edge-inpaint model
"""
import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from PIL import Image
from src.config import Config
from src.edge_connect import EdgeConnect
from src.utils import create_dir, imsave
from skimage.feature import canny
from skimage.color import rgb2gray


def test_all_stages(config_path, input_dir, mask_dir, output_dir, model_path):
    """
    Test và lưu kết quả từng giai đoạn
    
    Args:
        config_path: đường dẫn đến config.yml
        input_dir: thư mục chứa ảnh input
        mask_dir: thư mục chứa mask
        output_dir: thư mục lưu kết quả
        model_path: đường dẫn đến thư mục chứa model checkpoints
    """
    # Load config
    config = Config(config_path)
    config.MODE = 2
    config.MODEL = 3  # edge-inpaint model
    config.TEST_FLIST = input_dir
    config.TEST_MASK_FLIST = mask_dir
    config.RESULTS = output_dir
    config.INPUT_SIZE = 0
    config.PATH = model_path
    
    # Set device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")
    
    # Set random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # Build model
    model = EdgeConnect(config)
    model.load()
    
    # Create output directories
    create_dir(output_dir)
    stage1_dir = os.path.join(output_dir, 'stage1_original_masked')
    stage2_dir = os.path.join(output_dir, 'stage2_canny_edges')
    stage3_dir = os.path.join(output_dir, 'stage3_hallucinated_edges')
    stage4_dir = os.path.join(output_dir, 'stage4_inpaint_from_canny')
    stage5_dir = os.path.join(output_dir, 'stage5_final_result')
    comparison_dir = os.path.join(output_dir, 'comparison')
    
    for d in [stage1_dir, stage2_dir, stage3_dir, stage4_dir, stage5_dir, comparison_dir]:
        create_dir(d)
    
    # Test dataset
    from src.dataset import Dataset
    test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, 
                          config.TEST_MASK_FLIST, augment=False, training=False)
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
    print("\nBắt đầu test và lưu kết quả từng giai đoạn...\n")
    
    model.edge_model.eval()
    model.inpaint_model.eval()
    
    index = 0
    for items in test_loader:
        name = test_dataset.load_name(index)
        images, images_gray, edges_canny, masks = model.cuda(*items)
        index += 1
        
        print(f"Đang xử lý [{index}/{len(test_dataset)}]: {name}")
        
        # Stage 1: Ảnh gốc với mask (màu trắng ở vùng mask)
        masked_image = (images * (1 - masks) + masks)
        stage1 = model.postprocess(masked_image)[0]
        imsave(stage1, os.path.join(stage1_dir, name))
        
        # Stage 2: Canny edges ban đầu
        # Edge là grayscale, cần expand để có 3 channels
        edges_canny_3ch = edges_canny.repeat(1, 3, 1, 1) if edges_canny.shape[1] == 1 else edges_canny
        stage2 = model.postprocess(1 - edges_canny_3ch)[0]  # Invert để hiển thị rõ hơn
        imsave(stage2, os.path.join(stage2_dir, name))
        
        # Stage 3: Hallucinated edges từ edge model
        with torch.no_grad():
            edges_hallucinated = model.edge_model(images_gray, edges_canny, masks)
        edges_merged = (edges_hallucinated * masks) + (edges_canny * (1 - masks))
        edges_merged_3ch = edges_merged.repeat(1, 3, 1, 1) if edges_merged.shape[1] == 1 else edges_merged
        stage3 = model.postprocess(1 - edges_merged_3ch)[0]  # Invert để hiển thị rõ hơn
        imsave(stage3, os.path.join(stage3_dir, name))
        
        # Stage 4: Inpainting chỉ với Canny edges (không dùng hallucinated edges)
        with torch.no_grad():
            outputs_canny_only = model.inpaint_model(images, edges_canny, masks)
        outputs_merged_canny = (outputs_canny_only * masks) + (images * (1 - masks))
        stage4 = model.postprocess(outputs_merged_canny)[0]
        imsave(stage4, os.path.join(stage4_dir, name))
        
        # Stage 5: Kết quả cuối cùng với hallucinated edges
        with torch.no_grad():
            outputs_final = model.inpaint_model(images, edges_merged, masks)
        outputs_merged_final = (outputs_final * masks) + (images * (1 - masks))
        stage5 = model.postprocess(outputs_merged_final)[0]
        imsave(stage5, os.path.join(stage5_dir, name))
        
        # Tạo ảnh so sánh (stitch tất cả các giai đoạn)
        
        # Lấy các ảnh đã postprocess
        img_original = model.postprocess(images)
        img_masked = model.postprocess(masked_image)
        edges_canny_3ch = edges_canny.repeat(1, 3, 1, 1) if edges_canny.shape[1] == 1 else edges_canny
        edges_merged_3ch = edges_merged.repeat(1, 3, 1, 1) if edges_merged.shape[1] == 1 else edges_merged
        img_canny = model.postprocess(1 - edges_canny_3ch)
        img_hallucinated = model.postprocess(1 - edges_merged_3ch)
        img_final = model.postprocess(outputs_merged_final)
        
        # Chuyển sang numpy và PIL Image để stitch
        def tensor_to_pil(tensor_img):
            if isinstance(tensor_img, torch.Tensor):
                img_np = tensor_img[0].cpu().numpy().astype(np.uint8)
            else:
                img_np = tensor_img[0].astype(np.uint8)
            if len(img_np.shape) == 2:
                return Image.fromarray(img_np, mode='L').convert('RGB')
            return Image.fromarray(img_np)
        
        imgs = [
            tensor_to_pil(img_original),
            tensor_to_pil(img_masked),
            tensor_to_pil(img_canny),
            tensor_to_pil(img_hallucinated),
            tensor_to_pil(img_final)
        ]
        
        # Tạo ảnh so sánh dọc (vertical)
        width, height = imgs[0].size
        gap = 5
        total_height = height * len(imgs) + gap * (len(imgs) - 1)
        comparison = Image.new('RGB', (width, total_height))
        
        y_offset = 0
        for img in imgs:
            comparison.paste(img, (0, y_offset))
            y_offset += height + gap
        
        fname, fext = name.split('.')
        comparison.save(os.path.join(comparison_dir, fname + '_comparison.' + fext))
        
        print(f"  ✓ Đã lưu kết quả cho {name}\n")
    
    print(f"\n✓ Hoàn thành! Kết quả được lưu tại: {output_dir}")
    print(f"\nCấu trúc thư mục:")
    print(f"  - stage1_original_masked/: Ảnh gốc với mask (màu trắng)")
    print(f"  - stage2_canny_edges/: Edge map từ Canny detector")
    print(f"  - stage3_hallucinated_edges/: Edge map sau khi qua edge model")
    print(f"  - stage4_inpaint_from_canny/: Kết quả inpainting chỉ dùng Canny edges")
    print(f"  - stage5_final_result/: Kết quả cuối cùng với hallucinated edges")
    print(f"  - comparison/: Ảnh so sánh tất cả các giai đoạn")


def main():
    parser = argparse.ArgumentParser(description='Test EdgeConnect và hiển thị kết quả từng giai đoạn')
    parser.add_argument('--checkpoints', '--path', type=str, required=True,
                       help='Đường dẫn đến thư mục chứa model checkpoints')
    parser.add_argument('--input', type=str, required=True,
                       help='Thư mục chứa ảnh input')
    parser.add_argument('--mask', type=str, required=True,
                       help='Thư mục chứa mask')
    parser.add_argument('--output', type=str, default=None,
                       help='Thư mục lưu kết quả (mặc định: checkpoints/results_stages)')
    
    args = parser.parse_args()
    
    # Tạo đường dẫn config
    config_path = os.path.join(args.checkpoints, 'config.yml')
    
    # Tạo config nếu chưa có
    if not os.path.exists(config_path):
        if os.path.exists('./config.yml.example'):
            copyfile('./config.yml.example', config_path)
            print(f"Đã tạo file config từ template: {config_path}")
        else:
            print(f"Lỗi: Không tìm thấy config.yml tại {config_path}")
            return
    
    # Tạo output directory
    if args.output is None:
        output_dir = os.path.join(args.checkpoints, 'results_stages')
    else:
        output_dir = args.output
    
    test_all_stages(
        config_path=config_path,
        input_dir=args.input,
        mask_dir=args.mask,
        output_dir=output_dir,
        model_path=args.checkpoints
    )


if __name__ == '__main__':
    main()

