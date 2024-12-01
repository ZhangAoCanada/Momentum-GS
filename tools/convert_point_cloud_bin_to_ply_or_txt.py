import os
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_points3D_binary, read_points3D_text


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output_type", type=str, required=True)
    args = parser.parse_args()
    path = args.path
    output_type = args.output_type

    ply_path = os.path.join(path, "points3D.ply")
    bin_path = os.path.join(path, "points3D.bin")
    txt_path = os.path.join(path, "points3D.txt")
    if not os.path.exists(ply_path) or not os.path.exists(txt_path):
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            print("Could not find points3D.bin file")
        if output_type == "ply":
            print("Converting point3d.bin to .ply")
            storePly(ply_path, xyz, rgb)
        elif output_type == "txt":
            print("Converting point3d.bin to .txt")
            pc = np.concatenate((xyz, rgb), axis=1)
            np.savetxt(txt_path, pc, fmt='%.3f')
        else:
            print("Invalid output type. Please specify either 'ply' or 'txt'.")

    print("Done.")