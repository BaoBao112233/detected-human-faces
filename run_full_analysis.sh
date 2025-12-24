#!/bin/bash

echo "======================================================"
echo "CHẠY PHÂN TÍCH ĐẦY ĐỦ VIDEO test_new.avi"
echo "======================================================"
echo ""
echo "Đang xử lý toàn bộ 836 frames..."
echo "Mỗi frame sẽ tạo:"
echo "  - Ảnh crop cho mỗi person"
echo "  - File .txt chứa thông số chi tiết"
echo "  - Ảnh crop cho mỗi face (nếu có)"
echo ""
echo "Thời gian ước tính: ~5-10 phút"
echo ""

python run_detailed_analysis.py \
    --input /home/baobao/sshfs/orange-pi/Projects/detected-human-faces/input/test_new.avi \
    --output output/test_new_full_crops

echo ""
echo "======================================================"
echo "HOÀN THÀNH!"
echo "======================================================"
echo ""
echo "Kết quả đã lưu tại: output/test_new_full_crops/"
echo ""
echo "Thống kê file:"
ls -lh output/test_new_full_crops/ | tail -5
echo ""
echo "Tổng số file:"
ls output/test_new_full_crops/*.jpg 2>/dev/null | wc -l | xargs echo "- Person/Face crops:"
ls output/test_new_full_crops/*.txt 2>/dev/null | grep -v detailed | wc -l | xargs echo "- File thông số .txt:"
