# #### Step 1: 載入必要的函式庫

# import os, cv2, json, tqdm, torch, gc, re, glob
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from matplotlib.lines import Line2D
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# import plotly.graph_objects as go
# import open3d as o3d

# # 啟用 OpenEXR 支援
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# # 選擇 GPU (0: RTX 6000 Ada, 1/2: RTX PRO 6000 Blackwell, 3: RTX 3090)
# GPU_INDEX = 2

# if torch.cuda.is_available():
#     n_gpus = torch.cuda.device_count()
#     # print(f"\n=== Available GPUs ({n_gpus}) ===")
#     for i in range(n_gpus):
#         props = torch.cuda.get_device_properties(i)
#         mem_gb = props.total_memory / 1024**3
#         marker = " <-- selected" if i == GPU_INDEX else ""
#         # print(f"  cuda:{i}  {props.name}  ({mem_gb:.1f} GB){marker}")

#     assert GPU_INDEX < n_gpus, f"GPU_INDEX={GPU_INDEX} 超出範圍 (只有 {n_gpus} 張)"
#     torch.cuda.set_device(GPU_INDEX)
#     device = torch.device(f"cuda:{GPU_INDEX}")
#     # print(f"\nUsing device: {device} ({torch.cuda.get_device_name(GPU_INDEX)})")
# else:
#     device = torch.device("cpu")
#     print(f"Using device: {device}")

# #### Step 2: 定義核心函數

# def get_intrinsic_matrix(height, width):
#     """
#     獲取相機內參矩陣 K
    
#     G-Objaverse 使用固定焦距 fx=fy=1422.222 (基於 1024x1024 解析度)
#     主點位於影像中心
    
#     K = | fx  0  cx |
#         | 0  fy  cy |
#         | 0   0   1 |
#     """
#     fx = fy = 1422.222
#     res_raw = 1024
    
#     # 根據實際解析度縮放焦距
#     f_x = f_y = fx * height / res_raw
#     cx = width / 2
#     cy = height / 2
    
#     K = np.array([
#         [f_x, 0, cx],
#         [0, f_y, cy],
#         [0, 0, 1]
#     ], dtype=np.float32)
    
#     return K

# def read_camera_matrix(json_file):
#     """
#     讀取相機外參矩陣 (camera-to-world, c2w)
    
#     JSON 檔案包含:
#     - x: 相機 X 軸方向 (right)
#     - y: 相機 Y 軸方向 (up)
#     - z: 相機 Z 軸方向 (forward/backward)
#     - origin: 相機位置
    
#     注意: 需要 flip Y 和 Z 軸來轉換座標系統
#     """
#     with open(json_file, 'r', encoding='utf8') as f:
#         data = json.load(f)
    
#     c2w = np.eye(4, dtype=np.float32)
#     c2w[:3, 0] = np.array(data['x'])       # X 軸
#     c2w[:3, 1] = -np.array(data['y'])      # Y 軸 (flipped)
#     c2w[:3, 2] = -np.array(data['z'])      # Z 軸 (flipped)
#     c2w[:3, 3] = np.array(data['origin'])  # 相機位置
    
#     return c2w

# def read_depth_from_exr(exr_path, camera_position):
#     """
#     從 EXR 檔案讀取深度圖
    
#     *_nd.exr 檔案格式:
#     - Channel 0-2: Normal (世界座標系)
#     - Channel 3: Depth
    
#     深度值需要過濾掉 near plane 之前的無效值
#     """
#     # 計算相機距離原點的距離
#     cam_distance = np.linalg.norm(camera_position)
    
#     # near plane 設定 (物體被normalize到單位球內)
#     near = 0.867  # sqrt(3) * 0.5
#     near_distance = cam_distance - near
    
#     # 讀取 EXR 檔案
#     normald = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
#     depth = normald[..., 3]  # 第4個通道是深度
    
#     # 過濾無效深度值
#     depth[depth < near_distance] = 0
    
#     return depth, normald[..., :3]  # 回傳深度和法向量

# def read_rgb_image(img_path):
#     """讀取 RGB 影像並轉換到 [0, 1] 範圍"""
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {img_path}")
    
#     # BGR to RGB
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # 處理 RGBA
#     if img.shape[-1] == 4:
#         img = img[..., :3]
    
#     return img.astype(np.float32) / 255.0

# #### Step 3: 2D → 3D 反投影函數 (核心！)

# def unproject_depth_to_pointcloud(depth, K, c2w, rgb=None):
#     """
#     將深度圖反投影到 3D 世界座標點雲
    
#     數學公式:
#     1. 像素座標 (u, v) + depth → 相機座標 P_cam
#        P_cam = K^(-1) @ [u, v, 1]^T * depth
    
#     2. 相機座標 → 世界座標
#        P_world = c2w @ [P_cam, 1]^T
    
#     Args:
#         depth: [H, W] 深度圖
#         K: [3, 3] 內參矩陣
#         c2w: [4, 4] camera-to-world 矩陣
#         rgb: [H, W, 3] 可選的顏色圖
    
#     Returns:
#         points: [N, 3] 世界座標點雲
#         colors: [N, 3] 對應的顏色 (如果有提供 rgb)
#     """
#     H, W = depth.shape
    
#     # Step 1: 建立像素座標網格
#     u = np.arange(W)
#     v = np.arange(H)
#     u, v = np.meshgrid(u, v)  # [H, W]
    
#     # Step 2: 只取有效深度的點
#     valid_mask = depth > 0
#     u_valid = u[valid_mask]
#     v_valid = v[valid_mask]
#     depth_valid = depth[valid_mask]
    
#     # Step 3: 像素座標 → 齊次座標
#     # [u, v, 1]^T
#     ones = np.ones_like(u_valid)
#     pixel_coords = np.stack([u_valid, v_valid, ones], axis=0)  # [3, N]
    
#     # Step 4: 反投影到相機座標系
#     # P_cam = K^(-1) @ [u, v, 1]^T * depth
#     K_inv = np.linalg.inv(K)
#     cam_coords = K_inv @ pixel_coords  # [3, N]
#     cam_coords = cam_coords * depth_valid[np.newaxis, :]  # 乘以深度
    
#     # Step 5: 相機座標 → 世界座標
#     # P_world = c2w @ [P_cam, 1]^T
#     ones_row = np.ones((1, cam_coords.shape[1]))
#     cam_coords_homo = np.vstack([cam_coords, ones_row])  # [4, N]
#     world_coords = c2w @ cam_coords_homo  # [4, N]
#     points = world_coords[:3, :].T  # [N, 3]
    
#     # Step 6: 獲取顏色
#     colors = None
#     if rgb is not None:
#         colors = rgb[valid_mask]  # [N, 3]
    
#     return points, colors, valid_mask


# def convert_pose_flip_yz(c2w):
#     """
#     Flip Y 和 Z 軸 (用於座標系統轉換)
#     這是因為不同軟體使用不同的座標系統慣例
#     """
#     flip_yz = np.eye(4)
#     flip_yz[1, 1] = -1
#     flip_yz[2, 2] = -1
#     return c2w @ flip_yz

# #### Step 4: 設定資料路徑

# # 選擇 output_components.json 中第幾筆資料
# DATA_INDEX = 49

# # 讀進 object_id, question, answer
# with open("output_components.json", "r") as f:
#     output_components = json.load(f)

# object_oc = output_components[DATA_INDEX]
# id, name, question, answer = (
#     object_oc[k] for k in ["object_id", "object_name", "question", "answer"]
# )

# # 讀進 point_cloud, multi-view images
# with open("./dataset/gobjaverse_to_affogato_nj.json", "r") as f:
#     gobjaverse_to_affogato = json.load(f)

# object_gta = gobjaverse_to_affogato[DATA_INDEX]
# pc_path = f'{object_gta["dst"]}/xyzc.npy'
# img_folder_path = object_gta["src"]

# print(f"name: {name}\nid: {id}\nquestion: {question}\nanswer: {answer}")
# print(f"pc_path: {pc_path}\nimg_folder_path: {img_folder_path}")

# #### Step 5: 讀取單一視角的資料並視覺化

# # 選擇要讀取的視角
# VIEW_IDX = 0

# # 建立檔案路徑
# img_path = f'{img_folder_path}/{VIEW_IDX:05d}/{VIEW_IDX:05d}.png'
# depth_path = f'{img_folder_path}/{VIEW_IDX:05d}/{VIEW_IDX:05d}_nd.exr'
# json_path = f'{img_folder_path}/{VIEW_IDX:05d}/{VIEW_IDX:05d}.json'

# # 讀取相機參數
# c2w = read_camera_matrix(json_path)
# camera_position = c2w[:3, 3]

# # 讀取 RGB 和深度
# rgb = read_rgb_image(img_path)
# depth, normal = read_depth_from_exr(depth_path, camera_position)

# #### Step 6: 將單一視角反投影到 3D 點雲

# # 獲取內參矩陣
# H, W = depth.shape
# K = get_intrinsic_matrix(H, W)

# # 執行反投影
# c2w_converted = convert_pose_flip_yz(c2w)
# points, colors, valid_mask = unproject_depth_to_pointcloud(depth, K, c2w_converted, rgb)

# #### Step 7: 視覺化單視角 3D 點雲

# def visualize_point_cloud_3d(points, colors=None, title="Point Cloud", 
#                               camera_positions=None, subsample=5):
#     """
#     使用 matplotlib 視覺化 3D 點雲
    
#     Args:
#         points: [N, 3] 點雲座標
#         colors: [N, 3] 點的顏色
#         camera_positions: list of [3,] 相機位置 (可選)
#         subsample: 降採樣倍率 (加速視覺化)
#     """
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Downsampling 以加速視覺化
#     idx = np.arange(0, len(points), subsample)
#     pts = points[idx]
    
#     if colors is not None:
#         cols = colors[idx]
#         ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
#                    c=cols, s=1, alpha=0.6)
#     else:
#         ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
#                    s=1, alpha=0.6)
    
#     # 繪製相機位置
#     if camera_positions is not None:
#         cam_pos = np.array(camera_positions)
#         ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
#                    c='red', s=100, marker='^', label='Cameras')
#         ax.legend()
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(title)
    
#     # 設定相等的軸比例
#     max_range = np.max(np.abs(pts)) * 1.1
#     ax.set_xlim([-max_range, max_range])
#     ax.set_ylim([-max_range, max_range])
#     ax.set_zlim([-max_range, max_range])
    
#     plt.tight_layout()
#     return fig, ax

# #### Step 8: 合併多個視角的點雲

# def load_all_views(data_root, num_views=40):
#     """
#     載入所有視角的資料並合併成完整點雲
    
#     Args:
#         data_root: 資料根目錄
#         num_views: 視角數量 (預設 40)
    
#     Returns:
#         all_points: 合併的點雲
#         all_colors: 對應的顏色
#         camera_positions: 所有相機位置
#     """
#     all_points = []
#     all_colors = []
#     camera_positions = []
    
#     for view_idx in tqdm.trange(num_views, desc=f"Loading {num_views} views"):
#         # 檔案路徑
#         img_path = f'{data_root}/{view_idx:05d}/{view_idx:05d}.png'
#         depth_path = f'{data_root}/{view_idx:05d}/{view_idx:05d}_nd.exr'
#         json_path = f'{data_root}/{view_idx:05d}/{view_idx:05d}.json'
        
#         # 檢查檔案是否存在
#         if not all(os.path.exists(p) for p in [img_path, depth_path, json_path]):
#             print(f"  Skipping view {view_idx} (files not found)")
#             continue
        
#         # 讀取資料
#         c2w = read_camera_matrix(json_path)
#         camera_pos = c2w[:3, 3]
        
#         rgb = read_rgb_image(img_path)
#         depth, _ = read_depth_from_exr(depth_path, camera_pos)
        
#         # 內參
#         H, W = depth.shape
#         K = get_intrinsic_matrix(H, W)
        
#         # 反投影
#         c2w_converted = convert_pose_flip_yz(c2w)
#         points, colors, _ = unproject_depth_to_pointcloud(depth, K, c2w_converted, rgb)
        
#         all_points.append(points)
#         all_colors.append(colors)
#         camera_positions.append(camera_pos)
    
#     # 合併
#     all_points = np.vstack(all_points)
#     all_colors = np.vstack(all_colors)
#     camera_positions = np.array(camera_positions)
    
#     pcd = o3d.geometry.PointCloud()

#     pcd.points = o3d.utility.Vector3dVector(all_points)
#     pcd.colors = o3d.utility.Vector3dVector(all_colors)

#     pcd_filtered, ind = pcd.remove_statistical_outlier(
#         nb_neighbors=20,
#         std_ratio=1.0
#     )

#     all_points = np.asarray(pcd_filtered.points).astype(np.float32)
#     all_colors = np.asarray(pcd_filtered.colors).astype(np.float32)
    
#     return all_points, all_colors, camera_positions

# #### Step 11: 顯示 24 個 View Images

# # 載入多個視角 (這裡只載入前 8 個視角作為示範，可調整)
# NUM_VIEWS_TO_LOAD = 24  # 可以改成 40 載入所有視角

# all_points, all_colors, all_camera_positions = load_all_views(img_folder_path, NUM_VIEWS_TO_LOAD)

# # print(f"\n=== Summary ===")
# # print(f"Total cameras: {len(all_camera_positions)}, Total points: {len(all_points):,}")

# def save_pointcloud_npz(save_path, points, colors, camera_positions):
#     """
#     將 point cloud + camera positions 存成 npz
#     """
#     np.savez_compressed(
#         save_path,
#         points=points.astype(np.float32),
#         colors=colors.astype(np.float32),
#         camera_positions=camera_positions.astype(np.float32)
#     )
#     print(f"[INFO] Saved to {save_path}")
    
# def load_pointcloud_npz(npz_path):
#     """
#     讀取 npz point cloud
#     """
#     data = np.load(npz_path)

#     points = data["points"]
#     colors = data["colors"]
#     camera_positions = data["camera_positions"]

#     print(f"[INFO] Loaded from {npz_path}")
#     print(f"       points: {points.shape}, colors: {colors.shape}")

#     return points, colors, camera_positions

# os.makedirs(f'output/{id}', exist_ok=True)
# save_path = f'output/{id}/recon.npz'
# save_pointcloud_npz(
#     save_path,
#     all_points,
#     all_colors,
#     all_camera_positions
# )
# all_points, all_colors, all_camera_positions = load_pointcloud_npz(save_path)

# # 載入並顯示所有 24 view 的 RGB 影像
# NUM_VIEWS_TO_LOAD = 24

# view_images = []       # 存放 PIL Image (給後續 Molmo / SAM2 使用)
# view_images_np = []    # numpy 格式

# for view_idx in range(NUM_VIEWS_TO_LOAD):
#     img_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.png'
#     img = Image.open(img_path).convert('RGB')                      # 讀取為 PIL Image, 並確保是 RGB 模式
#     view_images.append(img)         
#     view_images_np.append(np.array(img))                           # 存成 numpy 格式以便 matplotlib 顯示

# #### Step 12: 使用 Molmo2 VLM 標註 Affordance 點

# MOLMO_MODEL_ID = "allenai/MolmoPoint-8B"

# # 載入 MolmoPoint-8B
# print(f"\nLoading Molmo2 model: {MOLMO_MODEL_ID} ...")

# molmo_processor = AutoProcessor.from_pretrained(
#     MOLMO_MODEL_ID,
#     trust_remote_code=True,
# )

# molmo_model = AutoModelForImageTextToText.from_pretrained(
#     MOLMO_MODEL_ID,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
# )

# ROLE_QUERIES = {
#     "role_move":   f"Point to the part of the {name} that is the {answer[0]}.",
#     "role_action": f"Point to the part of the {name} that is the {answer[1]}.",
# }

# ROLES = list(ROLE_QUERIES.keys())

# _DEFAULT_COLORS = ["red", "deepskyblue", "lime", "orange", "magenta"]
# _DEFAULT_SCALES = ["Reds", "Blues", "Greens", "Oranges", "Purples"]
# ROLE_COLORS      = {r: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, r in enumerate(ROLES)}
# ROLE_COLORSCALES = {r: _DEFAULT_SCALES[i % len(_DEFAULT_SCALES)] for i, r in enumerate(ROLES)}

# # MolmoPoint-8B: 從已載入的模型模組中取得 extract_image_points
# _molmo_module = type(molmo_model).__module__
# _mod = __import__(_molmo_module, fromlist=['extract_image_points'])
# extract_image_points = _mod.extract_image_points
# # print(f"Loaded extract_image_points from {_molmo_module}")

# def molmo_query_image(img, query_text):
#     messages = [{"role": "user", "content": [
#         {"type": "image", "image": img},
#         {"type": "text", "text": query_text},
#     ]}]
    
#     prompt_text = molmo_processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
    
#     inputs = molmo_processor(
#         text=prompt_text,
#         images=[img],
#         return_pointing_metadata=True,
#         return_tensors="pt",
#     )
    
#     metadata = inputs.pop("metadata")
#     inputs = {k: v.to(molmo_model.device) for k, v in inputs.items()}

#     with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
#         generated_ids = molmo_model.generate(**inputs, max_new_tokens=200)

#     generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
#     generated_text = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)

#     extracted = extract_image_points(
#         output_text=generated_text,
#         pooling=metadata["token_pooling"],
#         mappings=metadata["subpatch_mapping"],
#         no_more_points_class=molmo_model.config.no_more_points_class,
#         location=molmo_model.config.patch_location is not None,
#         image_sizes=metadata["image_sizes"],
#     )
    
#     # Molmo 解碼順序即 confidence 順序，只保留最強的一個
#     if len(extracted) > 0:
#         _, _, x, y = extracted[0]
#         pts = [np.array([x, y])]
#     else:
#         pts = []

#     del inputs, generated_ids, generated_tokens
#     gc.collect()
#     torch.cuda.empty_cache()

#     return pts, generated_text

# # Per-role × per-view: 跑 Molmo affordance query (top-1 per role)
# molmo_points_per_view_per_role = {role: [[] for _ in range(len(view_images))] for role in ROLES}
# molmo_raw_outputs_per_role     = {role: [""] * len(view_images)               for role in ROLES}

# for role in ROLES:
#     query_text = ROLE_QUERIES[role]
#     # print(f"\n=== Role: {role} ===\nQuery: {query_text}")

#     for view_idx in tqdm.trange(len(view_images), desc=f"MolmoPoint [{role}]"):
#         pts, raw = molmo_query_image(view_images[view_idx], query_text)
#         molmo_points_per_view_per_role[role][view_idx] = pts
#         molmo_raw_outputs_per_role[role][view_idx]     = raw

#     n_views_with = sum(1 for p in molmo_points_per_view_per_role[role] if len(p) > 0)
#     n_pts = sum(len(p) for p in molmo_points_per_view_per_role[role])
#     # print(f"  → views with points: {n_views_with}/{len(view_images)}, total points: {n_pts}")

# # 視覺化: 在同一組 grid 上畫所有 role 的點 (top-1 per role)，
# #         以 ROLE_COLORS 區分，legend 標註對應的 query

# fig, axes = plt.subplots(4, 6, figsize=(20, 13))

# for i, ax in enumerate(axes.flat):
#     if i < len(view_images_np):
#         ax.imshow(view_images_np[i])
#         for role in ROLES:
#             pts = molmo_points_per_view_per_role[role][i]
#             for pt in pts:
#                 ax.plot(pt[0], pt[1], '*', markersize=14,
#                         c=ROLE_COLORS[role],
#                         markeredgecolor='white', markeredgewidth=0.6)
#         counts = ", ".join(
#             f'{role}:{len(molmo_points_per_view_per_role[role][i])}' for role in ROLES
#         )
#         ax.set_title(f'View {i}  ({counts})', fontsize=8)
#     ax.axis('off')

# # 一個 legend，把 query 文字也帶上
# legend_handles = [
#     Line2D([0], [0], marker='*', linestyle='None',
#            markerfacecolor=ROLE_COLORS[role],
#            markeredgecolor='white', markeredgewidth=0.6,
#            markersize=14,
#            label=f'[{role}] "{ROLE_QUERIES[role]}"')
#     for role in ROLES
# ]
# fig.legend(
#     handles=legend_handles,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 0.95),
#     ncol=1,
#     fontsize=12,
#     frameon=True,
# )

# # plt.suptitle("Molmo Affordance Points (top-1 per role)", fontsize=14, y=0.965)
# # plt.tight_layout(rect=[0, 0, 1, 0.92])
# # plt.show()

# save_path = f'output/{id}/molmo_point.png'
# fig.savefig(save_path, dpi=200, bbox_inches="tight")
# # print(f"[INFO] saved figure to {save_path}")

# #### Step 13: SAM2 Point Prompt → Mask → Probabilistic Heatmap

# SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# # print(f"Loading SAM2: {SAM2_CHECKPOINT} ...")

# sam2_model = build_sam2(model_cfg, SAM2_CHECKPOINT, device=device)
# sam2_predictor = SAM2ImagePredictor(sam2_model)
# # print("SAM2 loaded!")

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))

# def make_gaussian_weight(H, W, points, sigma=50):
#     """
#     建立以 Molmo 點為中心的 Gaussian weight map。
#     多個點取 max（union of Gaussians）。sigma 越小越聚焦。
#     """
#     u = np.arange(W)[np.newaxis, :]  # [1, W]
#     v = np.arange(H)[:, np.newaxis]  # [H, 1]

#     weight = np.zeros((H, W), dtype=np.float32)
#     for pt in points:
#         dx = u - pt[0]
#         dy = v - pt[1]
#         g = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
#         weight = np.maximum(weight, g)
#     return weight


# # ====== 參數設定 ======
# GAUSSIAN_SIGMA = 77             # Gaussian 半徑 (像素)
# MASK_SELECT = "best_iou"        # "smallest" or "best_iou"

# # ====== Per-role × per-view: Molmo 點 → SAM2 mask logits → Gaussian-weighted heatmap ======
# heatmaps_per_view_per_role = {role: [] for role in ROLES}

# for role in ROLES:
#     # print(f"\n=== SAM2 heatmaps for [{role}]  (mask={MASK_SELECT}, sigma={GAUSSIAN_SIGMA}) ===")
#     points_per_view = molmo_points_per_view_per_role[role]

#     for view_idx in tqdm.trange(len(view_images), desc=f"Heatmaps [{role}]"):
#         points = points_per_view[view_idx]
#         img_np = view_images_np[view_idx]

#         if len(points) == 0:
#             H, W = img_np.shape[:2]
#             heatmaps_per_view_per_role[role].append(np.zeros((H, W), dtype=np.float32))
#             continue

#         sam2_predictor.set_image(img_np)

#         point_coords = np.array(points, dtype=np.float32)
#         point_labels = np.ones(len(points), dtype=np.int32)

#         masks, iou_scores, low_res_logits = sam2_predictor.predict(
#             point_coords=point_coords,
#             point_labels=point_labels,
#             multimask_output=True,
#             return_logits=True,
#         )

#         if MASK_SELECT == "smallest":
#             mask_areas = [(m > 0).sum() for m in masks]
#             best_idx = np.argmin(mask_areas)
#         else:
#             best_idx = np.argmax(iou_scores)

#         best_logits = masks[best_idx]
#         heatmap = sigmoid(best_logits).astype(np.float32)

#         H, W = img_np.shape[:2]
#         gauss_w = make_gaussian_weight(H, W, points, sigma=GAUSSIAN_SIGMA)
#         heatmap = heatmap * gauss_w

#         heatmaps_per_view_per_role[role].append(heatmap)
#         sam2_predictor.reset_predictor()

# # 視覺化: 每個 role 各畫一張 heatmap overlay grid
# for role in ROLES:
#     fig, axes = plt.subplots(4, 6, figsize=(18, 12))
#     heatmaps = heatmaps_per_view_per_role[role]

#     for i, ax in enumerate(axes.flat):
#         if i < len(view_images_np):
#             ax.imshow(view_images_np[i])
#             ax.imshow(heatmaps[i], cmap='jet', alpha=0.5, vmin=0, vmax=1)
#             ax.set_title(f'View {i}', fontsize=9)
#         ax.axis('off')

#     plt.suptitle(f'[{role}] SAM2 Probabilistic Heatmaps  —  Query: "{ROLE_QUERIES[role]}"', fontsize=15)
#     plt.tight_layout()
#     # plt.show()
#     save_path = f'output/{id}/sam2_{role}_heatmap.png'
#     fig.savefig(save_path, dpi=200, bbox_inches="tight")

# #### Step 14: 多視角 Voting — 把 2D Heatmap 投回原始 3D 點雲

# def project_and_sample_heatmaps(points, heatmaps, cameras, K_list, depths, depth_tolerance=0.05):
#     """
#     把原始 3D 點雲投影到每個 view 的 2D heatmap 上，做 voting aggregation。

#     Args:
#         points: [N, 3] 原始 3D 點雲 (來自 Step 8)
#         heatmaps: list of [H, W] 每個 view 的 2D heatmap
#         cameras: list of (c2w, w2c) tuples for each view
#         K_list: list of [3, 3] 內參矩陣
#         depths: list of [H, W] 深度圖 (用於可見性檢查)
#         depth_tolerance: 深度容差 (相對值)，用於判斷點是否被遮擋

#     Returns:
#         scores: [N] 每個點的 aggregated affordance score
#         counts: [N] 每個點被多少個 view 看到
#     """
#     N = len(points)
#     num_views = len(heatmaps)

#     score_sum = np.zeros(N, dtype=np.float32)
#     view_counts = np.zeros(N, dtype=np.int32)

#     points_homo = np.hstack([points, np.ones((N, 1))])  # [N, 4]

#     for view_idx in range(num_views):
#         c2w, w2c = cameras[view_idx]
#         K = K_list[view_idx]
#         heatmap = heatmaps[view_idx]
#         depth_map = depths[view_idx]
#         H, W = heatmap.shape

#         cam_coords = (w2c @ points_homo.T).T
#         cam_xyz = cam_coords[:, :3]

#         in_front = cam_xyz[:, 2] > 0

#         pixel_homo = (K @ cam_xyz.T).T
#         pixel_uv = pixel_homo[:, :2] / pixel_homo[:, 2:3]

#         u = pixel_uv[:, 0]
#         v = pixel_uv[:, 1]
#         in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

#         proj_depth = cam_xyz[:, 2]

#         u_int = np.clip(u.astype(np.int32), 0, W - 1)
#         v_int = np.clip(v.astype(np.int32), 0, H - 1)
#         sampled_depth = depth_map[v_int, u_int]

#         visible = (sampled_depth > 0) & (proj_depth < sampled_depth * (1 + depth_tolerance))

#         valid_mask = in_front & in_bounds & visible

#         sampled_scores = np.zeros(N, dtype=np.float32)
#         sampled_scores[valid_mask] = heatmap[v_int[valid_mask], u_int[valid_mask]]

#         score_sum += sampled_scores
#         view_counts += valid_mask.astype(np.int32)

#     final_scores = np.zeros(N, dtype=np.float32)
#     seen_mask = view_counts > 0
#     final_scores[seen_mask] = score_sum[seen_mask] / view_counts[seen_mask]

#     return final_scores, view_counts


# # ====== 準備投影所需的相機參數 (所有 role 共用) ======
# cameras = []      # list of (c2w, w2c)
# K_list = []
# depth_maps = []

# for view_idx in tqdm.trange(NUM_VIEWS_TO_LOAD, desc="Preparing camera matrices"):
#     json_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.json'
#     depth_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}_nd.exr'

#     c2w = read_camera_matrix(json_path)
#     c2w_converted = convert_pose_flip_yz(c2w)
#     w2c = np.linalg.inv(c2w_converted)

#     camera_pos = c2w[:3, 3]
#     depth, _ = read_depth_from_exr(depth_path, camera_pos)

#     H, W = depth.shape
#     K = get_intrinsic_matrix(H, W)

#     cameras.append((c2w, w2c))
#     K_list.append(K)
#     depth_maps.append(depth)


# # ====== Per-role voting aggregation ======
# affordance_scores_per_role = {}
# visibility_counts_per_role = {}

# for role in ROLES:
#     print(f"\n=== Voting Aggregation for [{role}] ===")
#     print(f"Projecting {len(gt_points):,} GT points to {len(heatmaps_per_view_per_role[role])} views...")
#     scores, counts = project_and_sample_heatmaps(
#         gt_points,
#         heatmaps_per_view_per_role[role],
#         cameras, K_list, depth_maps,
#         depth_tolerance=0.15,
#     )
#     affordance_scores_per_role[role] = scores
#     visibility_counts_per_role[role] = counts

#     # print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]   mean: {scores.mean():.4f}")
#     # print(f"  visibility:  min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f} views/point")
#     # print(f"  points seen by 0 views: {(counts == 0).sum():,}")

# # ====== 載入 reconstruction point cloud ======
# recon_data = np.load(f'output/{id}/recon.npz')
# recon_points = recon_data["points"].astype(np.float32)

# print(f"Loaded reconstruction points: {recon_points.shape}")


# # ====== 準備投影所需的相機參數 ======
# cameras = []      # list of (c2w, w2c)
# K_list = []
# depth_maps = []

# for view_idx in tqdm.trange(NUM_VIEWS_TO_LOAD, desc="Preparing camera matrices"):
#     json_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.json'
#     depth_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}_nd.exr'

#     c2w = read_camera_matrix(json_path)
#     c2w_converted = convert_pose_flip_yz(c2w)
#     w2c = np.linalg.inv(c2w_converted)

#     camera_pos = c2w[:3, 3]
#     depth, _ = read_depth_from_exr(depth_path, camera_pos)

#     H, W = depth.shape
#     K = get_intrinsic_matrix(H, W)

#     cameras.append((c2w, w2c))
#     K_list.append(K)
#     depth_maps.append(depth)


# # ====== Per-role voting aggregation on reconstruction ======
# affordance_scores_per_role = {}
# visibility_counts_per_role = {}

# for role in ROLES:
#     print(f"\n=== Voting Aggregation for [{role}] ===")
#     print(f"Projecting {len(recon_points):,} reconstruction points to {len(heatmaps_per_view_per_role[role])} views...")

#     scores, counts = project_and_sample_heatmaps(
#         recon_points,
#         heatmaps_per_view_per_role[role],
#         cameras,
#         K_list,
#         depth_maps,
#         depth_tolerance=0.15,
#     )

#     affordance_scores_per_role[role] = scores
#     visibility_counts_per_role[role] = counts

#     print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
#     print(f"  visible points: {(counts > 0).sum():,} / {len(counts):,}")

# # ====== 3D 視覺化: 每個 role 一張 plot (Reconstruction Version) ======
# import plotly.graph_objects as go

# subsample = 20

# idx = np.arange(0, len(recon_points), subsample)
# pts_sub = recon_points[idx]

# print(f"Subsampled: {len(pts_sub):,} points (1/{subsample})")

# for role in ROLES:
#     scores_sub = affordance_scores_per_role[role][idx]
#     counts_sub = visibility_counts_per_role[role][idx]

#     fig = go.Figure()

#     fig.add_trace(go.Scatter3d(
#         x=pts_sub[:, 0],
#         y=-pts_sub[:, 1],
#         z=pts_sub[:, 2],
#         mode='markers',
#         marker=dict(
#             size=1,
#             color=scores_sub,
#             colorscale=ROLE_COLORSCALES[role],
#             cmin=0,
#             cmax=1,
#             colorbar=dict(title=f'{role}<br>score'),
#             opacity=0.85,
#         ),
#         text=[
#             f'score={s:.3f}, views={c}'
#             for s, c in zip(scores_sub, counts_sub)
#         ],
#         hoverinfo='text',
#         name=role,
#     ))

#     fig.update_layout(
#         title=(
#             f'[{role}] Affordance Heatmap  —  '
#             f'Query: "{ROLE_QUERIES[role]}"<br>'
#             f'({len(pts_sub):,} points, subsampled 1/{subsample})'
#         ),
#         scene=dict(
#             aspectmode='data',
#             xaxis_title='X',
#             yaxis_title='Y',
#             zaxis_title='Z',
#         ),
#         width=900,
#         height=700,
#     )

#     fig.show()

#### Step 1: 載入必要的函式庫

import os, cv2, json, tqdm, torch, gc, re, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from matplotlib.lines import Line2D
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import plotly.graph_objects as go
import open3d as o3d

# 啟用 OpenEXR 支援
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# ==============================================================================
# Step 2: 定義核心函數 (相機、深度、影像 I/O)
# ==============================================================================

def get_intrinsic_matrix(height, width):
    """
    獲取相機內參矩陣 K

    G-Objaverse 使用固定焦距 fx=fy=1422.222 (基於 1024x1024 解析度)
    主點位於影像中心

    K = | fx  0  cx |
        | 0  fy  cy |
        | 0   0   1 |
    """
    fx = fy = 1422.222
    res_raw = 1024

    f_x = f_y = fx * height / res_raw
    cx = width / 2
    cy = height / 2

    K = np.array([
        [f_x, 0, cx],
        [0, f_y, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def read_camera_matrix(json_file):
    """
    讀取相機外參矩陣 (camera-to-world, c2w)

    JSON 檔案包含:
    - x: 相機 X 軸方向 (right)
    - y: 相機 Y 軸方向 (up)
    - z: 相機 Z 軸方向 (forward/backward)
    - origin: 相機位置

    注意: 需要 flip Y 和 Z 軸來轉換座標系統
    """
    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = np.array(data['x'])       # X 軸
    c2w[:3, 1] = -np.array(data['y'])      # Y 軸 (flipped)
    c2w[:3, 2] = -np.array(data['z'])      # Z 軸 (flipped)
    c2w[:3, 3] = np.array(data['origin'])  # 相機位置

    return c2w


def read_depth_from_exr(exr_path, camera_position):
    """
    從 EXR 檔案讀取深度圖

    *_nd.exr 檔案格式:
    - Channel 0-2: Normal (世界座標系)
    - Channel 3: Depth

    深度值需要過濾掉 near plane 之前的無效值
    """
    cam_distance = np.linalg.norm(camera_position)

    # near plane 設定 (物體被 normalize 到單位球內)
    near = 0.867  # sqrt(3) * 0.5
    near_distance = cam_distance - near

    normald = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[..., 3]  # 第 4 個通道是深度

    depth[depth < near_distance] = 0

    return depth, normald[..., :3]  # 回傳深度和法向量


def read_rgb_image(img_path):
    """讀取 RGB 影像並轉換到 [0, 1] 範圍"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[-1] == 4:
        img = img[..., :3]

    return img.astype(np.float32) / 255.0


# ==============================================================================
# Step 3: 2D → 3D 反投影函數 (核心！)
# ==============================================================================

def unproject_depth_to_pointcloud(depth, K, c2w, rgb=None):
    """
    將深度圖反投影到 3D 世界座標點雲

    數學公式:
    1. 像素座標 (u, v) + depth → 相機座標 P_cam
       P_cam = K^(-1) @ [u, v, 1]^T * depth

    2. 相機座標 → 世界座標
       P_world = c2w @ [P_cam, 1]^T

    Args:
        depth: [H, W] 深度圖
        K: [3, 3] 內參矩陣
        c2w: [4, 4] camera-to-world 矩陣
        rgb: [H, W, 3] 可選的顏色圖

    Returns:
        points: [N, 3] 世界座標點雲
        colors: [N, 3] 對應的顏色 (如果有提供 rgb)
        valid_mask: [H, W] 有效像素的 mask
    """
    H, W = depth.shape

    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)  # [H, W]

    valid_mask = depth > 0
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]

    ones = np.ones_like(u_valid)
    pixel_coords = np.stack([u_valid, v_valid, ones], axis=0)  # [3, N]

    K_inv = np.linalg.inv(K)
    cam_coords = K_inv @ pixel_coords          # [3, N]
    cam_coords = cam_coords * depth_valid[np.newaxis, :]

    ones_row = np.ones((1, cam_coords.shape[1]))
    cam_coords_homo = np.vstack([cam_coords, ones_row])  # [4, N]
    world_coords = c2w @ cam_coords_homo       # [4, N]
    points = world_coords[:3, :].T             # [N, 3]

    colors = None
    if rgb is not None:
        colors = rgb[valid_mask]               # [N, 3]

    return points, colors, valid_mask


def convert_pose_flip_yz(c2w):
    """
    Flip Y 和 Z 軸 (用於座標系統轉換)
    這是因為不同軟體使用不同的座標系統慣例
    """
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return c2w @ flip_yz


# ==============================================================================
# Step 4: 讀取資料設定
# ==============================================================================

def load_data_config(data_index):
    """
    從 output_components.json 與 gobjaverse_to_affogato_nj.json
    讀取指定 index 的物件資訊與路徑設定。

    Returns:
        object_id, object_name, question, answer, pc_path, img_folder_path
    """
    with open("output_components.json", "r") as f:
        output_components = json.load(f)

    object_oc = output_components[data_index]
    
    if not object_oc.get("operable", False):
        print(f"[SKIP] DATA_INDEX={data_index} ({object_oc.get('object_name', '')}) is not operable.")
        return None
    
    object_id, name, question, answer = (
        object_oc[k] for k in ["object_id", "object_name", "question", "answer"]
    )

    with open("./dataset/gobjaverse_to_affogato_nj.json", "r") as f:
        gobjaverse_to_affogato = json.load(f)

    object_gta = gobjaverse_to_affogato[data_index]
    pc_path = f'{object_gta["dst"]}/xyzc.npy'
    img_folder_path = object_gta["src"]

    print(f"name: {name}\nid: {object_id}\nquestion: {question}\nanswer: {answer}")
    print(f"pc_path: {pc_path}\nimg_folder_path: {img_folder_path}")

    return object_id, name, question, answer, pc_path, img_folder_path


# ==============================================================================
# Step 5 & 6: 讀取單一視角並反投影
# ==============================================================================

def load_and_unproject_single_view(img_folder_path, view_idx):
    """
    讀取單一視角的 RGB、深度、相機參數，並反投影為 3D 點雲。

    Returns:
        points, colors, valid_mask, rgb, depth, normal, c2w
    """
    img_path   = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.png'
    depth_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}_nd.exr'
    json_path  = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.json'

    c2w = read_camera_matrix(json_path)
    camera_position = c2w[:3, 3]

    rgb = read_rgb_image(img_path)
    depth, normal = read_depth_from_exr(depth_path, camera_position)

    H, W = depth.shape
    K = get_intrinsic_matrix(H, W)

    c2w_converted = convert_pose_flip_yz(c2w)
    points, colors, valid_mask = unproject_depth_to_pointcloud(depth, K, c2w_converted, rgb)

    return points, colors, valid_mask, rgb, depth, normal, c2w


# ==============================================================================
# Step 7: 視覺化單視角 3D 點雲 (函數定義，不呼叫)
# ==============================================================================

def visualize_point_cloud_3d(points, colors=None, title="Point Cloud",
                              camera_positions=None, subsample=5):
    """
    使用 matplotlib 視覺化 3D 點雲

    Args:
        points: [N, 3] 點雲座標
        colors: [N, 3] 點的顏色
        camera_positions: list of [3,] 相機位置 (可選)
        subsample: 降採樣倍率 (加速視覺化)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    idx = np.arange(0, len(points), subsample)
    pts = points[idx]

    if colors is not None:
        cols = colors[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=cols, s=1, alpha=0.6)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1, alpha=0.6)

    if camera_positions is not None:
        cam_pos = np.array(camera_positions)
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
                   c='red', s=100, marker='^', label='Cameras')
        ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = np.max(np.abs(pts)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    return fig, ax


# ==============================================================================
# Step 8: 合併多個視角的點雲
# ==============================================================================

def load_all_views(data_root, num_views=40):
    """
    載入所有視角的資料並合併成完整點雲

    Args:
        data_root: 資料根目錄
        num_views: 視角數量 (預設 40)

    Returns:
        all_points: 合併的點雲
        all_colors: 對應的顏色
        camera_positions: 所有相機位置
    """
    all_points = []
    all_colors = []
    camera_positions = []

    for view_idx in tqdm.trange(num_views, desc=f"Loading {num_views} views"):
        img_path   = f'{data_root}/{view_idx:05d}/{view_idx:05d}.png'
        depth_path = f'{data_root}/{view_idx:05d}/{view_idx:05d}_nd.exr'
        json_path  = f'{data_root}/{view_idx:05d}/{view_idx:05d}.json'

        if not all(os.path.exists(p) for p in [img_path, depth_path, json_path]):
            print(f"  Skipping view {view_idx} (files not found)")
            continue

        c2w = read_camera_matrix(json_path)
        camera_pos = c2w[:3, 3]

        rgb = read_rgb_image(img_path)
        depth, _ = read_depth_from_exr(depth_path, camera_pos)

        H, W = depth.shape
        K = get_intrinsic_matrix(H, W)

        c2w_converted = convert_pose_flip_yz(c2w)
        points, colors, _ = unproject_depth_to_pointcloud(depth, K, c2w_converted, rgb)

        all_points.append(points)
        all_colors.append(colors)
        camera_positions.append(camera_pos)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    camera_positions = np.array(camera_positions)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    pcd_filtered, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=1.0
    )

    all_points = np.asarray(pcd_filtered.points).astype(np.float32)
    all_colors = np.asarray(pcd_filtered.colors).astype(np.float32)

    return all_points, all_colors, camera_positions


# ==============================================================================
# Step 11: 存取點雲 & 載入視角影像
# ==============================================================================

def save_pointcloud_npz(save_path, points, colors, camera_positions):
    """將 point cloud + camera positions 存成 npz"""
    np.savez_compressed(
        save_path,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        camera_positions=camera_positions.astype(np.float32)
    )
    print(f"[INFO] Saved to {save_path}")


def load_pointcloud_npz(npz_path):
    """讀取 npz point cloud"""
    data = np.load(npz_path)

    points = data["points"]
    colors = data["colors"]
    camera_positions = data["camera_positions"]

    print(f"[INFO] Loaded from {npz_path}")
    print(f"       points: {points.shape}, colors: {colors.shape}")

    return points, colors, camera_positions


def load_view_images(img_folder_path, num_views):
    """
    載入多個視角的 RGB 影像。

    Returns:
        view_images: list of PIL Image
        view_images_np: list of numpy array
    """
    view_images = []
    view_images_np = []

    for view_idx in range(num_views):
        img_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.png'
        img = Image.open(img_path).convert('RGB')
        view_images.append(img)
        view_images_np.append(np.array(img))

    return view_images, view_images_np


# ==============================================================================
# Step 12: Molmo2 VLM — 模型載入 & Affordance 標註
# ==============================================================================

def load_molmo_model(model_id):
    """
    載入 MolmoPoint 模型與 processor，並取得 extract_image_points 函式。

    Returns:
        molmo_processor, molmo_model, extract_image_points
    """
    print(f"\nLoading Molmo2 model: {model_id} ...")

    molmo_processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    molmo_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    _molmo_module = type(molmo_model).__module__
    _mod = __import__(_molmo_module, fromlist=['extract_image_points'])
    extract_image_points = _mod.extract_image_points

    return molmo_processor, molmo_model, extract_image_points


def molmo_query_image(img, query_text, molmo_processor, molmo_model, extract_image_points):
    """
    對單張影像執行 MolmoPoint affordance query，回傳 top-1 點座標。

    Returns:
        pts: list of np.array([x, y])，最多 1 個點
        generated_text: 模型原始輸出文字
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": query_text},
    ]}]

    prompt_text = molmo_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = molmo_processor(
        text=prompt_text,
        images=[img],
        return_pointing_metadata=True,
        return_tensors="pt",
    )

    metadata = inputs.pop("metadata")
    inputs = {k: v.to(molmo_model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        generated_ids = molmo_model.generate(**inputs, max_new_tokens=200)

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)

    extracted = extract_image_points(
        output_text=generated_text,
        pooling=metadata["token_pooling"],
        mappings=metadata["subpatch_mapping"],
        no_more_points_class=molmo_model.config.no_more_points_class,
        location=molmo_model.config.patch_location is not None,
        image_sizes=metadata["image_sizes"],
    )

    # Molmo 解碼順序即 confidence 順序，只保留最強的一個
    if len(extracted) > 0:
        _, _, x, y = extracted[0]
        pts = [np.array([x, y])]
    else:
        pts = []

    del inputs, generated_ids, generated_tokens
    gc.collect()
    torch.cuda.empty_cache()

    return pts, generated_text


def run_molmo_all_views(view_images, view_images_np, roles, role_queries, role_colors,
                        molmo_processor, molmo_model, extract_image_points, output_dir, object_id):
    """
    對所有視角、所有 role 執行 Molmo affordance query，
    並儲存視覺化結果。

    Returns:
        molmo_points_per_view_per_role: dict[role] -> list of list of np.array
    """
    molmo_points_per_view_per_role = {role: [[] for _ in range(len(view_images))] for role in roles}
    molmo_raw_outputs_per_role     = {role: [""] * len(view_images)               for role in roles}

    for role in roles:
        query_text = role_queries[role]

        for view_idx in tqdm.trange(len(view_images), desc=f"MolmoPoint [{role}]"):
            pts, raw = molmo_query_image(
                view_images[view_idx], query_text,
                molmo_processor, molmo_model, extract_image_points
            )
            molmo_points_per_view_per_role[role][view_idx] = pts
            molmo_raw_outputs_per_role[role][view_idx]     = raw

        n_views_with = sum(1 for p in molmo_points_per_view_per_role[role] if len(p) > 0)
        n_pts = sum(len(p) for p in molmo_points_per_view_per_role[role])
        print(f"  [{role}] views with points: {n_views_with}/{len(view_images)}, total points: {n_pts}")

    # 視覺化: 每個 view 單獨存一張，畫所有 role 的點
    molmo_dir = os.path.join(output_dir, "molmo")
    os.makedirs(molmo_dir, exist_ok=True)

    for i in range(len(view_images_np)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(view_images_np[i])
        for role in roles:
            pts = molmo_points_per_view_per_role[role][i]
            for pt in pts:
                ax.plot(pt[0], pt[1], '*', markersize=14,
                        c=role_colors[role],
                        markeredgecolor='white', markeredgewidth=0.6)
        counts = ", ".join(
            f'{role}:{len(molmo_points_per_view_per_role[role][i])}' for role in roles
        )
        ax.set_title(f'View {i}  ({counts})', fontsize=9)
        ax.axis('off')

        legend_handles = [
            Line2D([0], [0], marker='*', linestyle='None',
                   markerfacecolor=role_colors[role],
                   markeredgecolor='white', markeredgewidth=0.6,
                   markersize=14,
                   label=f'[{role}] "{role_queries[role]}"')
            for role in roles
        ]
        ax.legend(handles=legend_handles, fontsize=8, frameon=True)

        save_path = os.path.join(molmo_dir, f"molmo_view_{i}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] Saved Molmo visualizations to {molmo_dir}/")

    return molmo_points_per_view_per_role


# ==============================================================================
# Step 13: SAM2 — 模型載入 & Probabilistic Heatmap
# ==============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_gaussian_weight(H, W, points, sigma=50):
    """
    建立以 Molmo 點為中心的 Gaussian weight map。
    多個點取 max（union of Gaussians）。sigma 越小越聚焦。
    """
    u = np.arange(W)[np.newaxis, :]  # [1, W]
    v = np.arange(H)[:, np.newaxis]  # [H, 1]

    weight = np.zeros((H, W), dtype=np.float32)
    for pt in points:
        dx = u - pt[0]
        dy = v - pt[1]
        g = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        weight = np.maximum(weight, g)
    return weight


def load_sam2_model(checkpoint, model_cfg, device):
    """
    載入 SAM2 模型與 predictor。

    Returns:
        sam2_predictor
    """
    print(f"Loading SAM2: {checkpoint} ...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 loaded!")
    return sam2_predictor


def run_sam2_all_views(view_images_np, molmo_points_per_view_per_role,
                       roles, role_queries, sam2_predictor,
                       gaussian_sigma, mask_select,
                       output_dir, object_id):
    """
    對所有視角、所有 role 執行 SAM2 point prompt → heatmap，
    並儲存視覺化結果。

    Returns:
        heatmaps_per_view_per_role: dict[role] -> list of [H, W] heatmap
    """
    heatmaps_per_view_per_role = {role: [] for role in roles}

    for role in roles:
        points_per_view = molmo_points_per_view_per_role[role]

        for view_idx in tqdm.trange(len(view_images_np), desc=f"Heatmaps [{role}]"):
            points = points_per_view[view_idx]
            img_np = view_images_np[view_idx]

            if len(points) == 0:
                H, W = img_np.shape[:2]
                heatmaps_per_view_per_role[role].append(np.zeros((H, W), dtype=np.float32))
                continue

            sam2_predictor.set_image(img_np)

            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.ones(len(points), dtype=np.int32)

            masks, iou_scores, low_res_logits = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                return_logits=True,
            )

            if mask_select == "smallest":
                mask_areas = [(m > 0).sum() for m in masks]
                best_idx = np.argmin(mask_areas)
            else:
                best_idx = np.argmax(iou_scores)

            best_logits = masks[best_idx]
            heatmap = sigmoid(best_logits).astype(np.float32)

            H, W = img_np.shape[:2]
            gauss_w = make_gaussian_weight(H, W, points, sigma=gaussian_sigma)
            heatmap = heatmap * gauss_w

            heatmaps_per_view_per_role[role].append(heatmap)
            sam2_predictor.reset_predictor()

        # 視覺化: 每個 view 單獨存一張 heatmap overlay
        sam2_dir = os.path.join(output_dir, "sam2", role)
        os.makedirs(sam2_dir, exist_ok=True)
        heatmaps = heatmaps_per_view_per_role[role]

        for i in range(len(view_images_np)):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(view_images_np[i])
            ax.imshow(heatmaps[i], cmap='jet', alpha=0.5, vmin=0, vmax=1)
            ax.set_title(f'[{role}] View {i}', fontsize=9)
            ax.axis('off')

            save_path = os.path.join(sam2_dir, f"sam2_view_{i}.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"[INFO] Saved SAM2 heatmaps to {sam2_dir}/")

    return heatmaps_per_view_per_role


# ==============================================================================
# Step 14: 多視角 Voting — 把 2D Heatmap 投回 3D 點雲
# ==============================================================================

def project_and_sample_heatmaps(points, heatmaps, cameras, K_list, depths, depth_tolerance=0.05):
    """
    把原始 3D 點雲投影到每個 view 的 2D heatmap 上，做 voting aggregation。

    Args:
        points: [N, 3] 原始 3D 點雲
        heatmaps: list of [H, W] 每個 view 的 2D heatmap
        cameras: list of (c2w, w2c) tuples for each view
        K_list: list of [3, 3] 內參矩陣
        depths: list of [H, W] 深度圖 (用於可見性檢查)
        depth_tolerance: 深度容差 (相對值)，用於判斷點是否被遮擋

    Returns:
        scores: [N] 每個點的 aggregated affordance score
        counts: [N] 每個點被多少個 view 看到
    """
    N = len(points)
    num_views = len(heatmaps)

    score_sum = np.zeros(N, dtype=np.float32)
    view_counts = np.zeros(N, dtype=np.int32)

    points_homo = np.hstack([points, np.ones((N, 1))])  # [N, 4]

    for view_idx in range(num_views):
        c2w, w2c = cameras[view_idx]
        K = K_list[view_idx]
        heatmap = heatmaps[view_idx]
        depth_map = depths[view_idx]
        H, W = heatmap.shape

        cam_coords = (w2c @ points_homo.T).T
        cam_xyz = cam_coords[:, :3]

        in_front = cam_xyz[:, 2] > 0

        pixel_homo = (K @ cam_xyz.T).T
        pixel_uv = pixel_homo[:, :2] / pixel_homo[:, 2:3]

        u = pixel_uv[:, 0]
        v = pixel_uv[:, 1]
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        proj_depth = cam_xyz[:, 2]

        u_int = np.clip(u.astype(np.int32), 0, W - 1)
        v_int = np.clip(v.astype(np.int32), 0, H - 1)
        sampled_depth = depth_map[v_int, u_int]

        visible = (sampled_depth > 0) & (proj_depth < sampled_depth * (1 + depth_tolerance))

        valid_mask = in_front & in_bounds & visible

        sampled_scores = np.zeros(N, dtype=np.float32)
        sampled_scores[valid_mask] = heatmap[v_int[valid_mask], u_int[valid_mask]]

        score_sum += sampled_scores
        view_counts += valid_mask.astype(np.int32)

    final_scores = np.zeros(N, dtype=np.float32)
    seen_mask = view_counts > 0
    final_scores[seen_mask] = score_sum[seen_mask] / view_counts[seen_mask]

    return final_scores, view_counts


def prepare_camera_params(img_folder_path, num_views):
    """
    讀取所有視角的相機參數與深度圖，供 voting aggregation 使用。

    Returns:
        cameras: list of (c2w, w2c)
        K_list: list of [3, 3]
        depth_maps: list of [H, W]
    """
    cameras = []
    K_list = []
    depth_maps = []

    for view_idx in tqdm.trange(num_views, desc="Preparing camera matrices"):
        json_path  = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}.json'
        depth_path = f'{img_folder_path}/{view_idx:05d}/{view_idx:05d}_nd.exr'

        c2w = read_camera_matrix(json_path)
        c2w_converted = convert_pose_flip_yz(c2w)
        w2c = np.linalg.inv(c2w_converted)

        camera_pos = c2w[:3, 3]
        depth, _ = read_depth_from_exr(depth_path, camera_pos)

        H, W = depth.shape
        K = get_intrinsic_matrix(H, W)

        cameras.append((c2w, w2c))
        K_list.append(K)
        depth_maps.append(depth)

    return cameras, K_list, depth_maps


def run_voting_aggregation(target_points, heatmaps_per_view_per_role, roles,
                           cameras, K_list, depth_maps, depth_tolerance=0.15):
    """
    對指定點雲執行 per-role voting aggregation。

    Returns:
        affordance_scores_per_role: dict[role] -> [N] scores
        visibility_counts_per_role: dict[role] -> [N] counts
    """
    affordance_scores_per_role = {}
    visibility_counts_per_role = {}

    for role in roles:
        print(f"\n=== Voting Aggregation for [{role}] ===")
        print(f"Projecting {len(target_points):,} points to {len(heatmaps_per_view_per_role[role])} views...")

        scores, counts = project_and_sample_heatmaps(
            target_points,
            heatmaps_per_view_per_role[role],
            cameras, K_list, depth_maps,
            depth_tolerance=depth_tolerance,
        )
        affordance_scores_per_role[role] = scores
        visibility_counts_per_role[role] = counts

        print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  visible points: {(counts > 0).sum():,} / {len(counts):,}")

    return affordance_scores_per_role, visibility_counts_per_role


def save_affordance_plots(recon_points, affordance_scores_per_role, visibility_counts_per_role,
                          roles, role_queries, role_colorscales, output_dir, subsample=20):
    """
    對 reconstruction 點雲的 affordance score 繪製 3D Plotly 圖並儲存為 HTML。
    """
    idx = np.arange(0, len(recon_points), subsample)
    pts_sub = recon_points[idx]
    print(f"Subsampled: {len(pts_sub):,} points (1/{subsample})")

    for role in roles:
        scores_sub = affordance_scores_per_role[role][idx]
        counts_sub = visibility_counts_per_role[role][idx]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=pts_sub[:, 0],
            y=-pts_sub[:, 1],
            z=pts_sub[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=scores_sub,
                colorscale=role_colorscales[role],
                cmin=0,
                cmax=1,
                colorbar=dict(title=f'{role}<br>score'),
                opacity=0.85,
            ),
            text=[
                f'score={s:.3f}, views={c}'
                for s, c in zip(scores_sub, counts_sub)
            ],
            hoverinfo='text',
            name=role,
        ))

        fig.update_layout(
            title=(
                f'[{role}] Affordance Heatmap  —  '
                f'Query: "{role_queries[role]}"<br>'
                f'({len(pts_sub):,} points, subsampled 1/{subsample})'
            ),
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            width=900,
            height=700,
        )

        save_path = f'{output_dir}/affordance_{role}_3d.html'
        fig.write_html(save_path)
        print(f"[INFO] Saved 3D affordance plot to {save_path}")


# ==============================================================================
# main
# ==============================================================================

def setup_device(gpu_index):
    """初始化 CUDA 裝置"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            marker = " <-- selected" if i == gpu_index else ""
            print(f"  cuda:{i}  {props.name}  ({mem_gb:.1f} GB){marker}")

        assert gpu_index < n_gpus, f"GPU_INDEX={gpu_index} 超出範圍 (只有 {n_gpus} 張)"
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        print(f"\nUsing device: {device} ({torch.cuda.get_device_name(gpu_index)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return device


def main():
    # ──────────────────────────────────────────
    # 參數設定
    # ──────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=5)
    parser.add_argument("--gpu",   type=int, default=2)
    args = parser.parse_args()
    
    GPU_INDEX         = args.gpu
    DATA_START        = args.start    # 起始 index (inclusive)
    DATA_END          = args.end      # 結束 index (exclusive)
    VIEW_IDX          = 0             # 單視角預覽用
    NUM_VIEWS_TO_LOAD = 24

    MOLMO_MODEL_ID   = "allenai/MolmoPoint-8B"
    SAM2_CHECKPOINT  = "checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"

    GAUSSIAN_SIGMA   = 77
    MASK_SELECT      = "best_iou"   # "smallest" or "best_iou"
    SUBSAMPLE        = 20

    _DEFAULT_COLORS = ["red", "deepskyblue", "lime", "orange", "magenta"]
    _DEFAULT_SCALES = ["Reds", "Blues", "Greens", "Oranges", "Purples"]

    # ──────────────────────────────────────────
    # 模型載入 (只做一次，在迴圈外)
    # ──────────────────────────────────────────
    device = setup_device(GPU_INDEX)

    molmo_processor, molmo_model, extract_image_points = load_molmo_model(MOLMO_MODEL_ID)
    sam2_predictor = load_sam2_model(SAM2_CHECKPOINT, SAM2_MODEL_CFG, device)

    # ──────────────────────────────────────────
    # 資料迴圈
    # ──────────────────────────────────────────
    for DATA_INDEX in range(DATA_START, DATA_END):
        print(f"\n{'='*60}")
        print(f"Processing DATA_INDEX={DATA_INDEX}  ({DATA_INDEX - DATA_START + 1}/{DATA_END - DATA_START})")
        print(f"{'='*60}")

        # Step 4: 讀取資料設定
        result = load_data_config(DATA_INDEX)
        if result is None:
            continue
        object_id, name, question, answer, pc_path, img_folder_path = result
        output_dir = f'output/{object_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Step 5 & 6: 單視角反投影 (預覽用)
        points, colors, valid_mask, rgb, depth, normal, c2w = load_and_unproject_single_view(
            img_folder_path, VIEW_IDX
        )

        # Step 11: 多視角點雲重建 & 儲存
        all_points, all_colors, all_camera_positions = load_all_views(img_folder_path, NUM_VIEWS_TO_LOAD)

        recon_path = f'{output_dir}/recon.npz'
        save_pointcloud_npz(recon_path, all_points, all_colors, all_camera_positions)
        all_points, all_colors, all_camera_positions = load_pointcloud_npz(recon_path)

        view_images, view_images_np = load_view_images(img_folder_path, NUM_VIEWS_TO_LOAD)

        # Step 12: Molmo affordance 標註
        ROLE_QUERIES = {
            "role_move":   f"Point to the part of the {name} that is the {answer[0]}.",
            "role_action": f"Point to the part of the {name} that is the {answer[1]}.",
        }
        ROLES = list(ROLE_QUERIES.keys())
        ROLE_COLORS      = {r: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, r in enumerate(ROLES)}
        ROLE_COLORSCALES = {r: _DEFAULT_SCALES[i % len(_DEFAULT_SCALES)] for i, r in enumerate(ROLES)}

        molmo_points_per_view_per_role = run_molmo_all_views(
            view_images, view_images_np,
            ROLES, ROLE_QUERIES, ROLE_COLORS,
            molmo_processor, molmo_model, extract_image_points,
            output_dir, object_id,
        )

        # Step 13: SAM2 heatmap
        heatmaps_per_view_per_role = run_sam2_all_views(
            view_images_np, molmo_points_per_view_per_role,
            ROLES, ROLE_QUERIES, sam2_predictor,
            GAUSSIAN_SIGMA, MASK_SELECT,
            output_dir, object_id,
        )

        # Step 14: Voting Aggregation (GT 點雲)
        cameras, K_list, depth_maps = prepare_camera_params(img_folder_path, NUM_VIEWS_TO_LOAD)

        gt_points = np.load(pc_path)[:, :3].astype(np.float32)
        # Align AFFOGATO points to the G-Objaverse camera frame before voting:
        # swap Y/Z then flip the new Y -> (x, -z, y). Matches point_cloud_from_depth.ipynb
        # CELL 17. Without this, every GT point projects to the wrong pixel and samples
        # the wrong heatmap value. (NOTE: affordance_scores_gt is currently not saved by
        # main() below — only the recon plots are. To output GT-aligned bimanual heatmaps,
        # save affordance_scores_gt here, e.g. np.savez to output/{object_id}.)
        gt_points = gt_points[:, [0, 2, 1]]
        gt_points[:, 1] *= -1
        affordance_scores_gt, visibility_counts_gt = run_voting_aggregation(
            gt_points, heatmaps_per_view_per_role, ROLES,
            cameras, K_list, depth_maps,
        )

        # Step 14 (續): Voting Aggregation (Reconstruction 點雲)
        recon_data = np.load(f'{output_dir}/recon.npz')
        recon_points = recon_data["points"].astype(np.float32)
        print(f"Loaded reconstruction points: {recon_points.shape}")

        cameras, K_list, depth_maps = prepare_camera_params(img_folder_path, NUM_VIEWS_TO_LOAD)

        affordance_scores_recon, visibility_counts_recon = run_voting_aggregation(
            recon_points, heatmaps_per_view_per_role, ROLES,
            cameras, K_list, depth_maps,
        )

        # 儲存 3D affordance plot (HTML)
        save_affordance_plots(
            recon_points, affordance_scores_recon, visibility_counts_recon,
            ROLES, ROLE_QUERIES, ROLE_COLORSCALES,
            output_dir, subsample=SUBSAMPLE,
        )

        print(f"\n[INFO] DATA_INDEX={DATA_INDEX} ({object_id}) done.")


if __name__ == "__main__":
    main()