import cv2
import numpy as np

def perspective_transform(image_path, angle=30):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # 定义原始图片的四个角点
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 定义变换后的四个点（模拟3D投影）
    shift = 0.2 * w  # 控制透视强度
    pts2 = np.float32([
        [0 + shift, 0],       # 左上角
        [w - shift, 0],       # 右上角
        [0, h],               # 左下角
        [w, h]                # 右下角
    ])
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 应用变换
    warped = cv2.warpPerspective(img, M, (w, h))
    
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("3D Projection", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 示例
perspective_transform("annotated_region.png", angle=30)

# 示例
# mayavi_3d_projection("annotated_region.png", scale=0.1)