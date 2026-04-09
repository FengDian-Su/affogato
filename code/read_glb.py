import open3d as o3d

o3d.visualization.webrtc_server.enable_webrtc()

mesh = o3d.io.read_triangle_mesh("model.glb")
mesh.compute_vertex_normals()

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"  # 👈 很重要

o3d.visualization.draw([{
    "name": "mesh",
    "geometry": mesh,
    "material": mat
}])

# 現在讓我們來定義一下完整的抓資料的 pipeline：
# 1. 在 "/mnt/home/sufengdian/affogato/dataset/affogato/affogato_all_part000" 這個資料夾下有很多標示著 object id 的資料夾，比方說 "0a0a8274693445a6b533dce7f97f747c"