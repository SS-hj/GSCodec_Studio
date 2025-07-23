import json
import os
import subprocess
from dataclasses import dataclass, field, InitVar
import glob
import shutil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from gsplat import compression
from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats
from gsplat.utils import inverse_log_transform, log_transform
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


@dataclass
class SeqHevcCompressionBlockSort:
    """Uses quantization and sorting to compress splats into mp4 files via libx265
      and uses K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `imageio <https://pypi.org/project/imageio/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
    """

    use_sort: bool = True
    verbose: bool = True
    qp: Dict[str, Union[int, Dict[str, Any]]] = field(default_factory=lambda: {
        "means": -1,
        "opacities": 4,
        "quats": 4,
        "scales": 4,
        "sh0": 16,
        "shN":{
            "sh1": 20,
            "sh2": 24,
            "sh3": 28
        }
    })
    n_clusters: int = 32768
    debug: bool = False
    use_all_intra: bool = False

    block_size: int = 8
    threshold_size: int = 7

    attribute_codec_registry: InitVar[Optional[Dict[str, str]]] = None

    compress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _compress_video_hevc_16bit,
        "scales": _compress_video_hevc,
        "quats": _compress_quats_video_hevc,
        "opacities": _compress_video_hevc,
        "sh0": _compress_video_hevc,
        "shN": _compress_shN_video_hevc
        # "shN": _compress_masked_kmeans,
    })
    decompress_fn_map: Dict[str, Callable] = field(default_factory=lambda: {
        "means": _decompress_video_hevc_16bit,
        "scales": _decompress_video_hevc,
        "quats": _decompress_quats_video_hevc,
        "opacities": _decompress_video_hevc,
        "sh0": _decompress_video_hevc,
        "shN": _decompress_shN_video_hevc
        # "shN": _decompress_masked_kmeans,
    })


    def __post_init__(self, attribute_codec_registry):
        if attribute_codec_registry:
            available_functions = {
                "_compress_video_hevc_16bit": _compress_video_hevc_16bit,
                "_compress_video_hevc": _compress_video_hevc,
                "_compress_quats_video_hevc": _compress_quats_video_hevc,
                "_compress_shN_video_hevc": _compress_shN_video_hevc,
                # "_compress_masked_kmeans": _compress_masked_kmeans,

                "_decompress_video_hevc_16bit": _decompress_video_hevc_16bit,
                "_decompress_video_hevc": _decompress_video_hevc,
                "_decompress_quats_video_hevc": _decompress_quats_video_hevc,
                "_decompress_shN_video_hevc": _decompress_shN_video_hevc
                # "_decompress_masked_kmeans": _decompress_masked_kmeans,
            }

            for attr_name, attr_codec in attribute_codec_registry.items(): # go through the registry
                if attr_name in self.compress_fn_map and "encode" in attr_codec:
                    if attr_codec["encode"] in available_functions:
                        self.compress_fn_map[attr_name] = available_functions[attr_codec["encode"]]
                    else:
                        print(f"Warning: Unknown func: {attr_codec['encode']}")

                if attr_name in self.decompress_fn_map and "decode" in attr_codec:
                    if attr_codec["decode"] in available_functions:
                        self.decompress_fn_map[attr_name] = available_functions[attr_codec["decode"]]
                    else:
                        print(f"Warning: Unknown func: {attr_codec['decode']}")

    def _get_compress_fn(self, param_name: str) -> Callable:
        if param_name in self.compress_fn_map:
            return self.compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        if param_name in self.decompress_fn_map:
            return self.decompress_fn_map[param_name]
        else:
            return _decompress_npz
    
    def compress(self, compress_dir: str) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
        """

        # Param-specific preprocessing
        # splats["means"] = log_transform(splats["means"])
        self.splats_videos["quats"] = F.normalize(self.splats_videos["quats"], dim=-1)

        meta = {}
        for param_name in self.splats_videos.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": int(self.splats_videos["means"].size(1)),
                "qp": self.qp[param_name],
                "use_all_intra": self.use_all_intra,
                "debug": self.debug
            }
            meta[param_name] = compress_fn(
                compress_dir, param_name, self.splats_videos[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def decompress(self, compress_dir: str) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        for param_name, param_meta in meta.items():
            decompress_fn = self._get_decompress_fn(param_name)
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        # splats["means"] = inverse_log_transform(splats["means"])
        return splats
    
    def sort_with_frame_index(self, splats_list: List[Dict], frame_id: int = 0) -> Tensor:
        """Organize the list of splats into several sequences of attributs

        Args:

        """
        splats_to_be_sorted = splats_list[frame_id]

        n_gs = len(splats_to_be_sorted["means"])
        n_sidelen = int(np.ceil(n_gs**0.5))
        n_pad = n_sidelen**2 - n_gs
        if n_pad != 0:
            # splats = _crop_n_splats(splats, n_crop)
            splats_to_be_sorted = _pad_n_splats(splats_to_be_sorted, n_pad)
            print(
                f"Warning: Number of Gaussians was not square. Padded {n_pad} Gaussians."
            )
        
        _, sorted_indices = sort_splats(splats_to_be_sorted, return_indices=True, sort_with_shN=False)
        print(f"Finsh the sorting with frame {frame_id}.")

        return sorted_indices
    
    def splats_list_to_attribute_seq(self, splats_list: List[Dict]) -> Dict[str, Tensor]:
        sample_splat = splats_list[0]
        attribute_names = list(sample_splat.keys())

        splats_sequences = {}
        for attr_name in attribute_names:
            attr_seq = [splat[attr_name] for splat in splats_list if attr_name in splat]
            attr_seq = torch.stack(attr_seq, dim=0)

            splats_sequences[attr_name] = attr_seq
        
        return splats_sequences

    
    def pad_attr_seq(self, splats_videos: Dict[str, Tensor], n_pad) -> Dict[str, Tensor]:
        if n_pad != 0:
            print(
                f"Warning: Number of Gaussians was not square. Padded {n_pad} Gaussians."
            )
            for attr_name, splats_video in splats_videos.items():
                pad_shape = list(splats_video.shape)
                pad_shape[1] = n_pad
                if attr_name == "opacities":
                    pad_splats_video = -5 * torch.ones(pad_shape, dtype=splats_video.dtype, device=splats_video.device)
                elif attr_name == "scales":
                    pad_splats_video = -10 * torch.ones(pad_shape, dtype=splats_video.dtype, device=splats_video.device)
                else:
                    pad_splats_video = torch.zeros(pad_shape, dtype=splats_video.dtype, device=splats_video.device)

                splats_videos[attr_name] = torch.cat([splats_video, pad_splats_video], dim=1)
        
        return splats_videos
    
    def reorganize(self, splats_list: List[Dict]) -> Dict[str, Tensor]:  
        """
        Efficient 8x8 block clustering for maintaining high-quality compression
        """
        import gc  
        from torchpq.clustering import KMeans  
        import time  
        import numpy as np  
        import torch.nn.functional as F  

        # Clear GPU cache and enable expandable segments
        torch.cuda.empty_cache()  
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  
        
        # Convert list of splats into attribute sequences
        seq_attr_dict = self.splats_list_to_attribute_seq(splats_list)

        # Retrieve frame count and number of gaussians per frame
        T, N = seq_attr_dict["means"].shape[:2]  
        device = seq_attr_dict["means"].device
        block_size = self.block_size
        threshold_size = self.threshold_size

        # Greedy placement of cluster blocks
        def find_next_empty_region(grid, height, width):
            rows, cols = grid.shape
            # 왼쪽 상단부터 순차적으로 탐색
            for y in range(0, rows - height + 1, block_size):
                for x in range(0, cols - width + 1, block_size):
                    if np.all(grid[y:y+height, x:x+width] == -1):
                        return y, x
            return None

        # Only proceed if spherical harmonic data is present
        if "shN" in seq_attr_dict:  
            print(f"Creating high-quality {block_size}x{block_size} block layout...")  

            with torch.no_grad(): 
                # Use specified number of clusters
                n_clusters = self.n_clusters  
                print(f"Using {n_clusters} clusters as specified")
                self.splats_videos = {}  

                if self.use_all_intra:

                    # 1. Efficient feature extraction and clustering per frame
                    print("Efficient feature extraction...")  
                    for t in range(T):  
                        print(f"Processing frame {t}/{T}...")
                        
                        # Extract spherical harmonic features for this frame
                        sample_sh = seq_attr_dict["shN"][t]
                        
                        # Flatten and transpose for clustering
                        features = sample_sh.reshape(sample_sh.shape[0], -1).permute(1, 0).contiguous()  
                        
                        # Perform K-Means clustering using Manhattan distance
                        kmeans = KMeans(  
                            n_clusters=n_clusters,   
                            distance="manhattan",
                        )  
                        
                        full_labels = kmeans.fit(features)

                        # Group gaussians by cluster label  
                        print("Grouping by cluster...")  
                        clusters = {}  
                        for c in range(n_clusters):  
                            cluster_mask = (full_labels == c)  
                            clusters[c] = torch.nonzero(cluster_mask).squeeze(1)  
                            print(f"Cluster {c}: {len(clusters[c])} gaussians")

                        # Ensure all gaussians are assigned
                        total_assigned = sum(len(indices) for indices in clusters.values())  
                        if total_assigned != N:  
                            print(f"Warning: Not all gaussians assigned! {total_assigned}/{N}")
                            missing_mask = (full_labels == -1)  
                            missing_indices = torch.nonzero(missing_mask).squeeze(1)  
                            if len(missing_indices) > 0:  
                                print(f"Assigning {len(missing_indices)} missing gaussians to random clusters")  
                                for idx in missing_indices:  
                                    c = np.random.randint(0, n_clusters)  
                                    if c not in clusters:  
                                        clusters[c] = []  
                                    clusters[c] = torch.cat([clusters[c], idx.unsqueeze(0)])  
                        
                        # Clean up intermediate tensors
                        del features, full_labels  
                        gc.collect()  
                
                        # 2. Calculate block requirements per cluster
                        print("Computing optimal block layout...")  
                        cluster_blocks = {}  # {cluster_id: (width, height)}  
                        total_blocks_needed = 0  
                        for c, indices in clusters.items():
                            elements_count = len(indices)  
                            blocks_needed = (elements_count + (block_size**2) - 1) // (block_size**2)
                            blocks_side = int(math.ceil(math.sqrt(blocks_needed))) * block_size
                            cluster_blocks[c] = (blocks_side, blocks_side)  
                            total_blocks_needed += blocks_side * blocks_side  
                            print(f"Cluster {c}: {blocks_side}x{blocks_side} blocks needed ({elements_count} elements)")
                        
                        # Determine overall grid size (multiple of block_size)
                        grid_blocks_side = math.ceil(math.sqrt(N)) 
                        grid_blocks_side = ((grid_blocks_side + 7) // block_size) * block_size 
                        print(f"Grid dimensions: {grid_blocks_side}x{grid_blocks_side} pixels")  
                    
                        # 3. Initialize grid and sort clusters by size
                        print("Arranging cluster blocks...")
                        grid = np.full((grid_blocks_side, grid_blocks_side), -1, dtype=np.int32)
                        sorted_clusters = sorted(
                            [(c, len(indices)) for c, indices in clusters.items()],
                            key=lambda x: x[1], reverse=True
                        )
                        cluster_positions = {c: [] for c in range(n_clusters)}  
                        placed_clusters = set()  

                        print("Placing clusters greedily...")
                        for c, _ in sorted_clusters:
                            blocks_width, blocks_height = cluster_blocks[c]
                            region = find_next_empty_region(grid, blocks_height, blocks_width)
                            if region is None:
                                break
                            y0, x0 = region

                            # Fill cluster pixels row by row
                            n_pixels = len(clusters[c])
                            filled = 0
                            cluster_positions[c] = []
                            for dy in range(blocks_height):
                                for dx in range(blocks_width):
                                    yy, xx = y0 + dy, x0 + dx
                                    if filled < n_pixels:
                                        grid[yy, xx] = c
                                        cluster_positions[c].append((yy, xx))
                                        filled += 1
                                    else:
                                        grid[yy, xx] = -2
                            placed_clusters.add(c)
                            print(f"Cluster {c} placed at ({y0}, {x0}) with size {blocks_height}x{blocks_width}")

                        # 4. Fill remaining clusters based on centroid similarity
                        centroids = kmeans.centroids.permute(1, 0)  
                        centroids_norm = centroids / centroids.norm(dim=1, keepdim=True)  
                        similarity_matrix = torch.mm(centroids_norm, centroids_norm.t())  
                        similarity_matrix.fill_diagonal_(0)

                        unplaced_clusters = list(set(clusters.keys()) - placed_clusters)
                        print("Filling remaining clusters by similarity...")
                        while unplaced_clusters:
                            c = unplaced_clusters.pop(0)
                            pixels_to_fill = len(clusters[c])
                            candidates = sorted(
                                placed_clusters,
                                key=lambda pc: similarity_matrix[c, pc],
                                reverse=True
                            )
                            # Attempt to fill near similar clusters
                            for pc in candidates:
                                if pixels_to_fill <= 0:
                                    break
                                y0, x0 = cluster_positions[pc][0]
                                blocks_width, blocks_height = cluster_blocks[pc]
                                
                                for y in range(y0, y0 + blocks_height):
                                    for x in range(x0, x0 + blocks_width):
                                        if grid[y, x] == -2:
                                            grid[y, x] = c
                                            cluster_positions[c].append((y, x))
                                            pixels_to_fill -= 1
                                            if pixels_to_fill <= 0:
                                                break
                                    if pixels_to_fill <= 0:
                                        break
                                
                            print(f"Cluster {c} filled {len(cluster_positions[c])} pixels, "
                                    f"remaining {pixels_to_fill} pixels to fill")
                            

                            # Fill any remaining holes sequentially
                            if pixels_to_fill > 0:
                                empties = np.argwhere(grid == -2)
                                for (y, x) in empties:
                                    if pixels_to_fill <= 0:
                                        break
                                    grid[y, x] = c
                                    cluster_positions[c].append((y, x))
                                    pixels_to_fill -= 1
                                print(f"Cluster {c} remaining {pixels_to_fill} pixels to fill")

                        print("\nCluster batch result:")
                        print(f"{np.sum(grid != -2)}/{grid.size} "  
                            f"({np.sum(grid != -2)/grid.size*100:.2f}%)")  

                        # 5. Map attributes into the 2D grid for each frame
                        print("Mapping attributes into 2D frames...")
                        for attr_name, tensor in seq_attr_dict.items():  
                            print(f"\n--- Processing attribute: {attr_name} ---")  
                            print(f"Tensor shape: {tensor.shape}")  
                            
                            if attr_name not in self.splats_videos:  
                                self.splats_videos[attr_name] = []
                            rest_dims = tensor.shape[2:]  
                            frame_2d = torch.zeros((grid_blocks_side, grid_blocks_side, *rest_dims),   
                                                    dtype=tensor.dtype, device=device)
                            frame_flat = tensor[t].reshape(-1, *rest_dims)
                            for c, positions in cluster_positions.items():  
                                if c in clusters:  
                                    indices = clusters[c]  
                                    for idx, (y, x) in zip(indices, positions):  
                                        # 해당 위치에 데이터 할당  
                                        frame_2d[y][x] = frame_flat[idx]

                            # Use same padding conventions
                            if attr_name == "opacities":
                                frame_2d[frame_2d == 0] = -5
                            elif attr_name == "scales":
                                frame_2d[frame_2d == 0] = -10
                            self.splats_videos[attr_name].append(frame_2d)


                        # 6. Sort splats within each cluster block if enabled
                        print("Sorting splats within cluster blocks...")
                        if self.use_sort:
                            attrs = list(self.splats_videos.keys())

                            flat_attr = {}
                            rest_shapes = {}
                            for attr in attrs:
                                full = self.splats_videos[attr][t]                # shape (H, W, *rest)
                                rest = full.shape[2:]                             # e.g. () or (3,) or (4,)
                                rest_shapes[attr] = rest
                                flat_attr[attr] = full.reshape(-1, *rest).contiguous()

                            # now for each cluster
                            for c in placed_clusters:
                                
                                h, w = cluster_blocks[c]  # (height, width) of the cluster block
                                y, x = cluster_positions[c][0]  # (y, x) of the first pixel in the cluster

                                if h > threshold_size:
                                    idxs = [(y+i)*grid_blocks_side + x+j for i in range(h) for j in range(w)]
                                    to_sort = {attr: flat_attr[attr][idxs] for attr in attrs}
                                    
                                    _, sorted_idx = sort_splats(to_sort, return_indices=True, sort_with_shN=False)
                                    
                                    for attr_name, splats in flat_attr.items():
                                        splats_block = flat_attr[attr_name][idxs].clone()
                                        reordered = splats_block[sorted_idx]
                                        flat_attr[attr_name][idxs] = reordered.to(flat_attr[attr_name].device)

                            for attr in attrs:
                                rest = rest_shapes[attr]
                                self.splats_videos[attr][t] = (
                                    flat_attr[attr]
                                    .reshape(grid_blocks_side, grid_blocks_side, *rest)
                                )
                        
                        del clusters, sample_sh
                        gc.collect()

                else:
                    # 1. Efficient feature extraction and clustering per frame
                    print("Efficient feature extraction...")  
                        
                    # Extract spherical harmonic features for this frame
                    sample_sh = seq_attr_dict["shN"][0]
                    
                    # Flatten and transpose for clustering
                    features = sample_sh.reshape(sample_sh.shape[0], -1).permute(1, 0).contiguous()  
                    
                    # Perform K-Means clustering using Manhattan distance
                    kmeans = KMeans(  
                        n_clusters=n_clusters,   
                        distance="manhattan",
                    )  
                    
                    full_labels = kmeans.fit(features)

                    # Group gaussians by cluster label  
                    print("Grouping by cluster...")  
                    clusters = {}  
                    for c in range(n_clusters):  
                        cluster_mask = (full_labels == c)  
                        clusters[c] = torch.nonzero(cluster_mask).squeeze(1)  
                        print(f"Cluster {c}: {len(clusters[c])} gaussians")

                    # Ensure all gaussians are assigned
                    total_assigned = sum(len(indices) for indices in clusters.values())  
                    if total_assigned != N:  
                        print(f"Warning: Not all gaussians assigned! {total_assigned}/{N}")
                        missing_mask = (full_labels == -1)  
                        missing_indices = torch.nonzero(missing_mask).squeeze(1)  
                        if len(missing_indices) > 0:  
                            print(f"Assigning {len(missing_indices)} missing gaussians to random clusters")  
                            for idx in missing_indices:  
                                c = np.random.randint(0, n_clusters)  
                                if c not in clusters:  
                                    clusters[c] = []  
                                clusters[c] = torch.cat([clusters[c], idx.unsqueeze(0)])  
                    
                    # Clean up intermediate tensors
                    del features, full_labels  
                    gc.collect()  
            
                    # 2. Calculate block requirements per cluster
                    print("Computing optimal block layout...")  
                    cluster_blocks = {}  # {cluster_id: (width, height)}  
                    total_blocks_needed = 0  
                    for c, indices in clusters.items():
                        elements_count = len(indices)  
                        blocks_needed = (elements_count + (block_size**2) - 1) // (block_size**2)
                        blocks_side = int(math.ceil(math.sqrt(blocks_needed))) * block_size
                        cluster_blocks[c] = (blocks_side, blocks_side)  
                        total_blocks_needed += blocks_side * blocks_side  
                        print(f"Cluster {c}: {blocks_side}x{blocks_side} blocks needed ({elements_count} elements)")
                    
                    # Determine overall grid size (multiple of block_size)
                    grid_blocks_side = math.ceil(math.sqrt(N)) 
                    grid_blocks_side = ((grid_blocks_side + 7) // block_size) * block_size
                    print(f"Grid dimensions: {grid_blocks_side}x{grid_blocks_side} pixels")  
                
                    # 3. Initialize grid and sort clusters by size
                    print("Arranging cluster blocks...")
                    grid = np.full((grid_blocks_side, grid_blocks_side), -1, dtype=np.int32)
                    sorted_clusters = sorted(
                        [(c, len(indices)) for c, indices in clusters.items()],
                        key=lambda x: x[1], reverse=True
                    )
                    cluster_positions = {c: [] for c in range(n_clusters)}  
                    placed_clusters = set()  

                    print("Placing clusters greedily...")
                    for c, _ in sorted_clusters:
                        blocks_width, blocks_height = cluster_blocks[c]
                        region = find_next_empty_region(grid, blocks_height, blocks_width)
                        if region is None:
                            break
                        y0, x0 = region

                        # Fill cluster pixels row by row
                        n_pixels = len(clusters[c])
                        filled = 0
                        cluster_positions[c] = []
                        for dy in range(blocks_height):
                            for dx in range(blocks_width):
                                yy, xx = y0 + dy, x0 + dx
                                if filled < n_pixels:
                                    grid[yy, xx] = c
                                    cluster_positions[c].append((yy, xx))
                                    filled += 1
                                else:
                                    grid[yy, xx] = -2
                        placed_clusters.add(c)
                        print(f"Cluster {c} placed at ({y0}, {x0}) with size {blocks_height}x{blocks_width}")

                    # 4. Fill remaining clusters based on centroid similarity
                    centroids = kmeans.centroids.permute(1, 0)  
                    centroids_norm = centroids / centroids.norm(dim=1, keepdim=True)  
                    similarity_matrix = torch.mm(centroids_norm, centroids_norm.t())  
                    similarity_matrix.fill_diagonal_(0)

                    unplaced_clusters = list(set(clusters.keys()) - placed_clusters)
                    print("Filling remaining clusters by similarity...")
                    while unplaced_clusters:
                        c = unplaced_clusters.pop(0)
                        pixels_to_fill = len(clusters[c])
                        candidates = sorted(
                            placed_clusters,
                            key=lambda pc: similarity_matrix[c, pc],
                            reverse=True
                        )
                        # Attempt to fill near similar clusters
                        for pc in candidates:
                            if pixels_to_fill <= 0:
                                break
                            y0, x0 = cluster_positions[pc][0]
                            blocks_width, blocks_height = cluster_blocks[pc]
                            
                            for y in range(y0, y0 + blocks_height):
                                for x in range(x0, x0 + blocks_width):
                                    if grid[y, x] == -2:
                                        grid[y, x] = c
                                        cluster_positions[c].append((y, x))
                                        pixels_to_fill -= 1
                                        if pixels_to_fill <= 0:
                                            break
                                if pixels_to_fill <= 0:
                                    break
                            
                        print(f"Cluster {c} filled {len(cluster_positions[c])} pixels, "
                                f"remaining {pixels_to_fill} pixels to fill")
                        

                        # Fill any remaining holes sequentially
                        if pixels_to_fill > 0:
                            empties = np.argwhere(grid == -2)
                            for (y, x) in empties:
                                if pixels_to_fill <= 0:
                                    break
                                grid[y, x] = c
                                cluster_positions[c].append((y, x))
                                pixels_to_fill -= 1
                            print(f"Cluster {c} remaining {pixels_to_fill} pixels to fill")

                    print("\nCluster batch result:")
                    print(f"{np.sum(grid != -2)}/{grid.size} "  
                        f"({np.sum(grid != -2)/grid.size*100:.2f}%)")  

                    # 5. Map attributes into the 2D grid for each frame
                    print("Mapping attributes into 2D frames...")
                    for t in range(T):
                        print(f"\n--- Frame {t} ---")
                        for attr_name, tensor in seq_attr_dict.items():  
                            print(f"Processing attribute: {attr_name}, tensor shape: {tensor.shape}")

                            if attr_name not in self.splats_videos:  
                                self.splats_videos[attr_name] = []
                            rest_dims = tensor.shape[2:]
                            # prepare a blank grid for this frame
                            frame_2d = torch.zeros((grid_blocks_side, grid_blocks_side, *rest_dims),
                                                    dtype=tensor.dtype, device=device)
                            # now use tensor[t], not tensor[0]
                            frame_flat = tensor[t].reshape(-1, *rest_dims)

                            for c, positions in cluster_positions.items():  
                                if c in clusters:  
                                    indices = clusters[c]
                                    for idx, (y, x) in zip(indices, positions):
                                        frame_2d[y, x] = frame_flat[idx]

                            # padding conventions
                            if attr_name == "opacities":
                                frame_2d[frame_2d == 0] = -5
                            elif attr_name == "scales":
                                frame_2d[frame_2d == 0] = -10

                            # append this frame
                            self.splats_videos[attr_name].append(frame_2d)

                    # 6. Sort splats within each cluster block if enabled
                    print("Sorting splats within cluster blocks...")
                    if self.use_sort:
                        attrs = list(self.splats_videos.keys())

                        flat_attr = {}
                        rest_shapes = {}
                        for attr in attrs:
                            full = self.splats_videos[attr][0]                # shape (H, W, *rest)
                            rest = full.shape[2:]                             # e.g. () or (3,) or (4,)
                            rest_shapes[attr] = rest
                            flat_attr[attr] = full.reshape(-1, *rest).contiguous()

                        
                        # now compute sorted_idx per cluster for frame 0
                        sorted_indices_per_cluster = {}
                        for c in placed_clusters:
                            h, w = cluster_blocks[c]
                            y, x = cluster_positions[c][0]
                            if h > threshold_size:
                                idxs = [(y+i)*grid_blocks_side + x+j for i in range(h) for j in range(w)]
                                to_sort = {attr: flat_attr[attr][idxs] for attr in attrs}

                                _, sorted_idx = sort_splats(to_sort, return_indices=True, sort_with_shN=False)
                                sorted_indices_per_cluster[c] = idxs, sorted_idx

                            
                        for t in range(T):
                            print(f"Applying precomputed sorting to frame {t}...")
                            # rebuild flat_attr for this frame
                            flat_attr = {}
                            for attr in attrs:
                                full = self.splats_videos[attr][t]
                                flat_attr[attr] = full.reshape(-1, *rest_shapes[attr]).contiguous()

                            # for each cluster, re‐order using the stored indices
                            for c, (idxs, sorted_idx) in sorted_indices_per_cluster.items():
                                for attr in attrs:
                                    block = flat_attr[attr][idxs].clone()
                                    reordered = block[sorted_idx]
                                    flat_attr[attr][idxs] = reordered.to(flat_attr[attr].device)

                            # write back into splats_videos[t]
                            for attr in attrs:
                                self.splats_videos[attr][t] = flat_attr[attr] \
                                    .reshape(grid_blocks_side, grid_blocks_side, *rest_shapes[attr])
                                            
                    del clusters, sample_sh
                    gc.collect()


        # 7. Stack per-frame grids into final tensors
        self.splats_videos = {  
            attr_name: torch.stack(frames, dim=0)  
            for attr_name, frames in self.splats_videos.items()  
        }

        for attr_name, tensor in seq_attr_dict.items():  
            print(f"Final shape for {attr_name}: {self.splats_videos[attr_name].shape}")

        # Normalize quaternion attribute if present  
        if "quats" in self.splats_videos:  
            self.splats_videos["quats"] = F.normalize(self.splats_videos["quats"], dim=-1)  

        # Save grid metadata
        self.cluster_grid_info = {
            "grid_blocks_side": int(grid_blocks_side),  
            "block_size": int(block_size),  
            "n_clusters": int(n_clusters),  
            "original_n_gaussians": int(N)  
        }  
        self.grid_blocks_side = grid_blocks_side

        return self.splats_videos  


    def deorganize(self, splats_videos: Dict[str, Tensor]) -> List[Dict]:
        flattened_splats_videos = {}
        for attr_name, splats_video in splats_videos.items():
            ori_shape = list(splats_video.shape)
            new_shape = [ori_shape[0], ori_shape[1] * ori_shape[2]] + ori_shape[3:]

            flattened_splats_videos[attr_name] = splats_video.reshape(new_shape)

        n_frames = flattened_splats_videos["means"].size(0)
        splats_list = []
        
        for frame_idx in range(n_frames):
            splat_dict = {}
            for attr_name, attr_seq in flattened_splats_videos.items():
                splat_dict[attr_name] = attr_seq[frame_idx, ...]
            
            splats_list.append(splat_dict)

        return splats_list

def _pad_n_splats(splats: Dict[str, Tensor], n_pad: int) -> Dict[str, Tensor]:
    for k, v in splats.items():
        pad_shape = list(v.shape)
        pad_shape[0] += n_pad
        padded_v = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        padded_v[:v.shape[0]] = v
        splats[k] = padded_v
    return splats

def _compress_video_hevc(
        compress_dir: str, 
        param_name: str, 
        params: Tensor, 
        n_sidelen: int, 
        qp: int = 10, 
        debug: bool = False,
        use_all_intra: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**8 - 1)).round().astype(np.uint8)
    if video.shape[-1] != 3:
        video = video[..., 0]
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video)

    # save each frame
    # if len(video.shape) == 2:  
    #     imageio.imwrite(os.path.join(compress_dir, f"{param_name}_frame000.png"), video)
    # else:  
    for i in range(n_frames):
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_frame{i:03d}.png"), video[i])
    
    # run ffmpeg libx265 to compress PNG file
    file_extension = ".265" if debug else ".mp4"
    video_file = os.path.join(compress_dir, f"{param_name}.{file_extension[1:]}")

    print(f"QP value of {param_name} is: {qp}")
    pix_fmt = "-pix_fmt gray" if param_name == "opacities" else ""
    intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""
    cmd = f"ffmpeg -i {compress_dir}/{param_name}_frame%03d.png -c:v libx265 {pix_fmt} -x265-params \"qp={qp}{intra_params}\" {video_file}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_frame*.png")))
    for png_file in png_files:
        os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_video_hevc(compress_dir: str, param_name: str, meta: Dict[str, Any]):
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader = imageio.get_reader(os.path.join(compress_dir, f"{param_name}.{file_extension[1:]}"), format='FFMPEG')

    frames = []
    for i, frame in enumerate(reader):
        frames.append(frame)
    
    video = np.stack(frames, axis=0)
    if param_name == "opacities":
        video = video[..., 0]
    
    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))

    video_norm = video / (2**8 - 1)

    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_video_hevc_16bit(
        compress_dir: str, 
        param_name: str, 
        params: Tensor, 
        n_sidelen: int, 
        qp: int = 10, 
        debug: bool = False,
        use_all_intra: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**16 - 1)).round().astype(np.uint16)
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video,)

    video_l = video & 0xFF
    video_u = (video >> 8) & 0xFF

    # save each frame
    for i in range(len(video)):
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_l_frame{i:03d}.png"), video_l[i].astype(np.uint8)
        )
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_u_frame{i:03d}.png"), video_u[i].astype(np.uint8)
        )
    
    for byte_select in ['l', 'u']:
        file_extension = ".265" if debug else ".mp4"
        video_file = os.path.join(compress_dir, f"{param_name}_{byte_select}.{file_extension[1:]}")

        # old
        intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_{byte_select}_frame%03d.png -c:v libx265 -x265-params \"lossless=1:preset=veryslow{intra_params}\" {video_file}"
        
        # new
        # intra_params = "-intra" if use_all_intra else ""
        # cmd = f"ffmpeg -i {compress_dir}/{param_name}_{byte_select}_frame%03d.png -c:v libx265 {intra_params} -x265-params \"lossless=1:preset=veryslow\" {video_file}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_video_hevc_16bit(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
) -> Tensor:
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader_l = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_l.{file_extension[1:]}"), format='FFMPEG')
    reader_u = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_u.{file_extension[1:]}"), format='FFMPEG')

    frames = []    
    for i, (frame_l, frame_u) in enumerate(zip(reader_l, reader_u)):
        frame_u = frame_u.astype(np.uint16)
        frame = (frame_u << 8) + frame_l
        frames.append(frame)
    
    video = np.stack(frames, axis=0)

    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(65535) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))

    video_norm = video / (2**16 - 1)

    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params    

def _compress_quats_video_hevc(
        compress_dir: str, 
        param_name: str, 
        params: Tensor, 
        n_sidelen: int, 
        qp: int = 10, 
        debug: bool = False,
        use_all_intra: bool = True
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    grid = params.reshape((n_frames, n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    video_norm = grid_norm.detach().cpu().numpy()

    video = (video_norm * (2**8 - 1)).round().astype(np.uint8)
    # video = video.squeeze()
    np.save(os.path.join(compress_dir, f"{param_name}.npy"), video,)

    video_w = video[..., 0] # [T, H, W]
    video_xyz = video[..., 1:] # [T, H, W, 3]

    for i in range(len(video)):
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_w_frame{i:03d}.png"), video_w[i])
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}_xyz_frame{i:03d}.png"), video_xyz[i])

    # run ffmpeg libx265 to compress PNG file
    file_extension = ".265" if debug else ".mp4"
    intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""

    video_file = os.path.join(compress_dir, f"{param_name}_w.{file_extension[1:]}")
    print(f"QP value of {param_name}_w is: {qp}")
    cmd = f"ffmpeg -i {compress_dir}/{param_name}_w_frame%03d.png -c:v libx265 -pix_fmt gray -x265-params \"qp={qp}{intra_params}\" {video_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    video_file = os.path.join(compress_dir, f"{param_name}_xyz.{file_extension[1:]}")
    print(f"QP value of {param_name}_xyz is: {qp}")
    cmd = f"ffmpeg -i {compress_dir}/{param_name}_xyz_frame%03d.png -c:v libx265 -x265-params \"qp={qp}{intra_params}\" {video_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta    

def _decompress_quats_video_hevc(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
):
    import imageio.v2 as imageio

    file_extension = meta["file_extension"]
    reader_w = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_w.{file_extension[1:]}"), format='FFMPEG')
    reader_xyz = imageio.get_reader(os.path.join(compress_dir, f"{param_name}_xyz.{file_extension[1:]}"), format='FFMPEG')

    frames = []
    for frame_w, frame_xyz in zip(reader_w, reader_xyz):
        frame = np.concatenate([frame_w[..., 0:1], frame_xyz], axis=-1)
        frames.append(frame)

    video = np.stack(frames, axis=0)

    # report the PSNR between reconstructed videos and original videos
    raw_video = np.load(os.path.join(compress_dir, f"{param_name}.npy"))
    cal_psnr = lambda x, y: float('inf') if (d := np.mean((x-y)**2)) == 0 else 20*np.log10(255) - 10*np.log10(d)
    print(f"PSNR of \"{param_name}\" map after video coding: {cal_psnr(raw_video, video)} dB")
    os.remove(os.path.join(compress_dir, f"{param_name}.npy"))   

    video_norm = video / (2**8 - 1)
    grid_norm = torch.tensor(video_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    # ori_params = torch.load(os.path.join(compress_dir, "quats.ckpt"))

    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params    

def _compress_shN_video_hevc(
        compress_dir: str, 
        param_name: str, 
        params: Tensor, 
        n_sidelen: int, 
        qp: Dict[str, int], 
        debug: bool = False,
        use_all_intra: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    n_frames = int(params.size(0))

    shN_name_list = []
    for degree in range(1,4):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"sh{degree}_{level}")

    grid = params # [T, H, W, 15, 3]
    mins = torch.amin(grid, dim=(0, 1, 2))
    maxs = torch.amax(grid, dim=(0, 1, 2))
    grid_norm = (grid - mins) / (maxs - mins)
    shN_norm = grid_norm.detach().cpu().numpy()

    shN_norm = (shN_norm * (2**8 - 1)).round().astype(np.uint8)
    # shN_norm = shN_norm.squeeze()

    for f_id in range(n_frames):
        for shN_id, shN_name in enumerate(shN_name_list):
            image = shN_norm[f_id,:,:,shN_id,:]
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{shN_name}_frame{f_id:03d}.png"), image)
    
    file_extension = ".265" if debug else ".mp4"
    intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""
    for shN_id, shN_name in enumerate(shN_name_list):
        print(f"QP value of {shN_name} is: {qp[shN_name[0:3]]}")
        video_file = os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}")
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_{shN_name}_frame%03d.png -c:v libx265 -x265-params \"qp={qp[shN_name[0:3]]}{intra_params}\" {video_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # remove png files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "file_extension": file_extension
    }
    return meta

def _decompress_shN_video_hevc(
        compress_dir: str, param_name: str, meta: Dict[str, Any]
):
    import imageio.v2 as imageio

    shN_name_list = []
    for degree in range(1,4):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"sh{degree}_{level}")

    file_extension = meta["file_extension"]

    shN_reader_list = []
    for shN_name in shN_name_list:
        shN_reader_list.append(imageio.get_reader(os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}"), format='FFMPEG'))

    shN_video_list = []
    for shN_reader in shN_reader_list: # loop on shN components
        shN_frames = []
        for shN_frame in shN_reader: # loop on frames
            shN_frames.append(shN_frame)
        shN_video = np.stack(shN_frames, axis=0)
        shN_video_list.append(shN_video)
    shN_videos = np.stack(shN_video_list, axis=3)

    shN_norm = shN_videos / (2**8 -1)

    grid_norm = torch.tensor(shN_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params 

def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any]) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params