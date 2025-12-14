"""
Script wrapper để tạo dữ liệu test cho EdgeConnect
Chạy script này từ thư mục gốc của project
"""
import sys
import os

# Thêm thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from create_data import main

if __name__ == '__main__':
    main()

