import os
import sys
import numpy as np
import open3d as o3d


def visualize_xyzc(input_path):
    """
    input_path:
        - 可以是資料夾
        - 或是直接 .npy 檔案
    """

    # ====== 判斷路徑 ======
    if os.path.isdir(input_path):
        npy_path = os.path.join(input_path, "xyzc.npy")
    else:
        npy_path = input_path

    if not os.path.exists(npy_path):
        raise ValueError(f"❌ 找不到檔案: {npy_path}")

    print(f"✅ Loading: {npy_path}")

    # ====== 讀取 npy ======
    data = np.load(npy_path)
    # print("Original shape:", data.shape)

    # ====== 處理 5 種 rendering ======
    if data.ndim == 3:
        if data.shape[2] == 5:
            print("Detected format: (N, 4, 5)")
            data = data[:, :, 0]
        elif data.shape[0] == 5:
            print("Detected format: (5, N, 4)")
            data = data[0, :, :]
        else:
            raise ValueError("❌ 無法辨識的 3D npy 格式")

    elif data.ndim == 2:
        print("Detected format: (N, C)")

    else:
        raise ValueError("❌ 不支援的 npy 維度")

    print("Processed shape:", data.shape)

    # ====== 解析 xyz + color ======
    if data.shape[1] >= 6:
        xyz = data[:, :3]
        rgb = data[:, 3:6]

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

    elif data.shape[1] == 4:
        xyz = data[:, :3]
        labels = data[:, 3]

        unique_labels = np.unique(labels)
        colors_map = {l: np.random.rand(3) for l in unique_labels}
        rgb = np.array([colors_map[l] for l in labels])

    else:
        raise ValueError("❌ 不支援的 channel 格式")

    # ====== 建立 PointCloud ======
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # ====== 正規化 ======
    center = pcd.get_center()
    pcd.translate(-center)

    scale = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
    )
    pcd.scale(1.0 / scale, center=(0, 0, 0))

    # ====== 顯示 ======
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw(pcd)


# ====== CLI entry ======
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("請提供 input path（資料夾或 npy 檔）")

    input_path = sys.argv[1]
    visualize_xyzc(input_path)