import json
import os
import subprocess
from dataclasses import dataclass, field, InitVar
import glob
import shutil
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
from sympy import im
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from gsplat import compression
from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats
import math

import numpy as np


def flip_sh_coefficients_by_cluster_block(
    splats_videos: Dict[str, List[torch.Tensor]],
    cluster_positions: Dict[int, List[Tuple[int, int]]],
    cluster_blocks: Dict[int, Tuple[int, int]],
    placed_clusters: List[int],
    device: torch.device
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]:
    """
    Performs SH coefficient flipping by cluster block unit and generates cluster block metadata.
    Each cluster's first position (y0, x0) through block size (width, height) area is considered as one block.
    
    Returns:
        Tuple[Dict[str, List[torch.Tensor]], Dict[str, Any]]: 
            - Flipped data
            - flip_info dictionary (shN flip information and cluster_blocks_meta included)
    """
    # Basic checks
    if "shN" not in splats_videos:
        print("No shN attribute found in splats_videos")
        return splats_videos, {}
    
    # Get basic information
    sh_frames = splats_videos["shN"]
    if not sh_frames:
        print("Empty shN frames list")
        return splats_videos, {}
        
    T = len(sh_frames)
    H, W, K, C = sh_frames[0].shape
    print(f"Processing {T} frames with shape H={H}, W={W}, K={K}, C={C}")
    
    # Initialize flip mask
    to_flip = torch.zeros((T, H, W, K), dtype=torch.int8, device=device)
    
    # Initialize cluster block metadata dictionary
    cluster_blocks_meta = {}
    total_pixels_covered = 0
    
    # Process each cluster block
    for block_id in placed_clusters:
        # Get cluster block's starting position and size
        y0, x0 = cluster_positions[block_id][0]
        blocks_width, blocks_height = cluster_blocks[block_id]
        
        # Adjust boundaries to valid region
        y1 = min(y0 + blocks_height - 1, H - 1)
        x1 = min(x0 + blocks_width - 1, W - 1)
        
        # Create block mask
        block_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                block_mask[y, x] = True
        
        # Calculate pixel count
        pixel_count = block_mask.sum().item()
        total_pixels_covered += pixel_count
        
        if pixel_count == 0:
            continue
        
        # Collect SH coefficient statistics for each frame
        all_mins = []
        all_maxs = []
        
        # Process frames for determining flip pattern
        for t in range(T):
            frame = sh_frames[t]
            
            # Extract SH coefficients within block
            block_sh = frame[block_mask]  # [num_pixels, K, C]
            
            # Calculate min/max values of SH coefficients within block
            mins = torch.min(block_sh, dim=0)[0]  # [K, C]
            maxs = torch.max(block_sh, dim=0)[0]  # [K, C]
            all_mins.append(mins)
            all_maxs.append(maxs)
            
            # Calculate mean for each SH basis and RGB channel within block
            block_means = block_sh.mean(dim=0)  # [K, C]
            
            # Count negative channels for each SH basis
            neg_channel_counts = (block_means < 0).sum(dim=-1)  # [K]
            
            # Decision: flip if 2 or more RGB channels are negative
            bases_to_flip = (neg_channel_counts >= 2).to(torch.int8)  # [K]
            
            # Apply same flip decision to all pixels within block
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    to_flip[t, y, x] = bases_to_flip
        
        # Calculate final min/max values across all frames
        if all_mins and all_maxs:
            stacked_mins = torch.stack(all_mins, dim=0)  # [T, K, C]
            stacked_maxs = torch.stack(all_maxs, dim=0)  # [T, K, C]
            final_mins = torch.min(stacked_mins, dim=0)[0]  # [K, C]
            final_maxs = torch.max(stacked_maxs, dim=0)[0]  # [K, C]
            
            # Save cluster block metadata
            cluster_blocks_meta[str(block_id)] = {
                "boundary": [int(y0), int(x0), int(y1), int(x1)],
                "size": [int(blocks_height), int(blocks_width)],
                "position": [int(y0), int(x0)],
                "pixel_count": int(pixel_count),
                "mins": final_mins.cpu().numpy().tolist(),
                "maxs": final_maxs.cpu().numpy().tolist()
            }
    
    print(f"Total pixels covered by all blocks: {total_pixels_covered} out of {H*W} ({total_pixels_covered/(H*W)*100:.2f}%)")
    
    # Apply flipping to generate new frames
    result_frames = []
    for t in range(T):
        # Generate sign tensor
        sign = torch.where(
            to_flip[t].unsqueeze(-1) == 1,
            torch.tensor(-1, device=device, dtype=sh_frames[t].dtype),
            torch.tensor(1, device=device, dtype=sh_frames[t].dtype),
        )
        
        # Apply flipping
        flipped_frame = sh_frames[t] * sign
        result_frames.append(flipped_frame)
    
    # Return result and flip information
    result = {k: v.copy() if isinstance(v, list) else v for k, v in splats_videos.items()}
    result["shN"] = result_frames
    
    # Save flip information and cluster block metadata
    flip_info = {
        "shN": to_flip.reshape(T, H*W, K),  # [T, H*W, K]
        "cluster_blocks_meta": cluster_blocks_meta  # Cluster block metadata
    }
    
    print(f"SH coefficient flipping completed for {T} frames with {len(cluster_blocks_meta)} valid blocks")
    return result, flip_info

@dataclass
class SeqHevcCompressionBlockSortUD:
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
            print(param_name)
            compress_fn = self._get_compress_fn(param_name)
            if param_name == "shN":
                # Use cluster_blocks_meta directly from flip_info
                # cluster_blocks_meta is already created and added in flip_sh_coefficients_by_cluster_block function
                kwargs = {
                    "n_clusters": self.n_clusters,
                    "flip_info": self.flip_info,
                    "qp": self.qp[param_name],
                    "use_all_intra": self.use_all_intra,
                    "debug": self.debug
                }
            else:
                kwargs = {
                    "n_sidelen": int(self.splats_videos["means"].size(1)),
                    "qp": self.qp[param_name],
                    "use_all_intra": self.use_all_intra,
                    "debug": self.debug
                }
            meta[param_name] = compress_fn(
                compress_dir, param_name, self.splats_videos[param_name], **kwargs
            )

        # Compress flip_info (only if exists)  
        if hasattr(self, 'flip_info') and self.flip_info:  
            flip_meta = {}  
            for attr_name, flip_data in self.flip_info.items():  
                print(f"Compressing flip_info for {attr_name}...")  
                
                # Convert to tensor if it's a list
                if isinstance(flip_data, list):  
                    flip_tensor = torch.stack(flip_data)
                    # Use NPZ compression
                    flip_meta[attr_name] = _compress_npz(  
                        compress_dir,   
                        f"flip_info_{attr_name}",   
                        flip_tensor  
                    )
                # Save directly if it's a dictionary
                elif isinstance(flip_data, dict):
                    # Save dictionary as JSON through modified NPZ function
                    flip_meta[attr_name] = _compress_npz(
                        compress_dir,
                        f"flip_info_{attr_name}",
                        flip_data
                    )
                # Save as is if it's a tensor
                else:
                    flip_tensor = flip_data
                    # Use NPZ compression
                    flip_meta[attr_name] = _compress_npz(  
                        compress_dir,   
                        f"flip_info_{attr_name}",   
                        flip_tensor  
                    )  
                    
                print(f"Flip info for {attr_name} compressed successfully")  
            
            # Add flip_info information to metadata  
            meta['flip_info'] = flip_meta  

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
        flip_info = {}  # Initialize dictionary to store flip information  
    
        for param_name, param_meta in meta.items():  
            if param_name == 'flip_info':  
                # Load flip_info data  
                print("Loading flip_info data...")  
                for attr_name, attr_meta in param_meta.items():  
                    flip_info[attr_name] = _decompress_npz(  
                        compress_dir, f"flip_info_{attr_name}", attr_meta  
                    )  
                    # Output appropriate information based on type
                    if isinstance(flip_info[attr_name], dict):
                        print(f"Loaded flip_info for {attr_name} (dictionary type)")
                    elif hasattr(flip_info[attr_name], 'shape'):
                        print(f"Loaded flip_info for {attr_name} with shape {flip_info[attr_name].shape}")
                    else:
                        print(f"Loaded flip_info for {attr_name} (type: {type(flip_info[attr_name])})")  
            else:  
                # Load general attributes  
                decompress_fn = self._get_decompress_fn(param_name)  
                splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)

        # Param-specific postprocessing
        # splats["means"] = inverse_log_transform(splats["means"])
        return splats, flip_info
    
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
            # Search sequentially from top-left
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
                                        # Assign data to corresponding position
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
                    
                    # Apply SH flipping after all frame mapping
                    if "shN" in self.splats_videos:  
                        print("Flipping SH coefficients by cluster block...")  
                        
                        # Apply block-based flipping function
                        flipped_tensors, flip_info = flip_sh_coefficients_by_cluster_block(  
                            splats_videos=self.splats_videos,  
                            cluster_positions=cluster_positions,  
                            cluster_blocks=cluster_blocks,  
                            placed_clusters=placed_clusters,  
                            device=device  
                        )  
                        
                        # Put results back into original data structure
                        for attr, tensor in flipped_tensors.items():  
                            for t in range(T):  
                                self.splats_videos[attr][t] = tensor[t]  
                        
                        # Save flip information
                        self.flip_info = flip_info

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

        # Save grid metadata and cluster block information
        self.cluster_grid_info = {
            "grid_blocks_side": int(grid_blocks_side),  
            "block_size": int(block_size),  
            "n_clusters": int(n_clusters),  
            "original_n_gaussians": int(N)
        }
        
        # Add cluster block boundary information
        cluster_blocks_meta = {}
        for cluster_id in placed_clusters:
            if cluster_id in cluster_positions and cluster_positions[cluster_id]:
                y0, x0 = cluster_positions[cluster_id][0]  # Top-left coordinates
                blocks_width, blocks_height = cluster_blocks[cluster_id]
                y1, x1 = min(y0 + blocks_height - 1, grid_blocks_side - 1), min(x0 + blocks_width - 1, grid_blocks_side - 1)  # Bottom-right coordinates
                
                # Calculate shN min/max values for each cluster block
                if "shN" in self.splats_videos:
                    shN_frames = self.splats_videos["shN"]  # [T, H, W, shN_dim, 3]
                    # Create mask with correct size
                    block_mask = torch.zeros((grid_blocks_side, grid_blocks_side), dtype=torch.bool, device=shN_frames.device)
                    
                    # Generate block mask
                    for y in range(y0, y1 + 1):
                        for x in range(x0, x1 + 1):
                            if 0 <= y < grid_blocks_side and 0 <= x < grid_blocks_side:
                                block_mask[y, x] = True
                    
                    # Calculate min/max values of shN values within block across all frames
                    mins_list = []
                    maxs_list = []
                    
                    for t in range(len(shN_frames)):
                        shN_frame = shN_frames[t]  # [H, W, shN_dim, 3]
                        
                        # Apply mask to select values within block and adjust shape
                        masked_values = shN_frame[block_mask]  # [num_masked_points, shN_dim, 3]
                        
                        if len(masked_values) > 0:  # Only if block has values
                            # Calculate min/max values for each SH coefficient and RGB channel
                            mins = torch.min(masked_values, dim=0)[0]  # [shN_dim, 3]
                            maxs = torch.max(masked_values, dim=0)[0]  # [shN_dim, 3]
                            
                            mins_list.append(mins)
                            maxs_list.append(maxs)
                    
                    if mins_list and maxs_list:
                        # Calculate min/max values across all frames
                        all_mins = torch.stack(mins_list, dim=0)  # [T, shN_dim]
                        all_maxs = torch.stack(maxs_list, dim=0)  # [T, shN_dim]
                        
                        final_mins = torch.min(all_mins, dim=0)[0]  # [shN_dim]
                        final_maxs = torch.max(all_maxs, dim=0)[0]  # [shN_dim]
                        
                        # Convert to numpy for storage
                        mins_np = final_mins.cpu().numpy().tolist()
                        maxs_np = final_maxs.cpu().numpy().tolist()
                    else:
                        # Set default values for empty block
                        shN_dim = shN_frames.shape[-2]  # Correct dimension: [T, H, W, shN_dim, 3]
                        rgb_dim = shN_frames.shape[-1]  # RGB channels: 3
                        mins_np = [[0] * rgb_dim for _ in range(shN_dim)]  # [shN_dim, 3] shape
                        maxs_np = [[1] * rgb_dim for _ in range(shN_dim)]  # [shN_dim, 3] shape
                else:
                    # Set default values when shN is not available
                    mins_np = [0]
                    maxs_np = [1]
                
                # Meta information for each cluster block
                cluster_blocks_meta[str(cluster_id)] = {
                    "boundary": [int(y0), int(x0), int(y1), int(x1)],
                    "size": [int(blocks_height), int(blocks_width)],
                    "position": [int(y0), int(x0)],
                    "mins": mins_np,  # Add shN minimum values
                    "maxs": maxs_np   # Add shN maximum values
                }
        
        # Save cluster block information
        self.cluster_blocks_meta = cluster_blocks_meta
        self.grid_blocks_side = grid_blocks_side
        
        # Add cluster block metadata to flip_info
        if hasattr(self, 'flip_info'):
            self.flip_info["cluster_blocks_meta"] = cluster_blocks_meta
        else:
            self.flip_info = {"cluster_blocks_meta": cluster_blocks_meta}

        return self.splats_videos  



    def deorganize(self, splats_videos: Dict[str, Tensor], flip_info=None) -> List[Dict]:
        """
        Convert 2D grid-arranged attributes back to original format and
        restore SH coefficients from flipping if needed.
        
        Args:
            splats_videos: Attribute dictionary arranged in 2D grid
            flip_info: SH coefficient flip information (optional)
            
        Returns:
            List of attribute dictionaries per frame
        """
        # Copy data to restore  
        restored_videos = {k: v.clone() for k, v in splats_videos.items()}  
        
        # Restore if 'shN' attribute exists and flip info is available  
        if "shN" in splats_videos and flip_info is not None and "shN" in flip_info:  
            print("Restoring flipped SH coefficients...")  
            
            # Restore SH coefficients  
            attr = "shN"  
            x = restored_videos[attr]  
            flip_mask = flip_info["shN"]  
            
            # Check input data format  
            if len(x.shape) != 5:
                print(f"Warning: Expected 5D tensor for grid format, got shape {x.shape}")
                # Continue but log warning
            
            T, H, W, K, C = x.shape  
            N = H * W  
            
            # Check and adjust flip mask format
            if len(flip_mask.shape) == 4:  # If in [T, H, W, K] format
                print("Converting flip mask from [T, H, W, K] to [T, H*W, K]")
                flip_mask = flip_mask.reshape(flip_mask.shape[0], -1, flip_mask.shape[-1])
            
            # Check size mismatch
            if flip_mask.shape[0] != T or flip_mask.shape[1] != N or flip_mask.shape[2] != K:
                print(f"Warning: Flip mask shape {flip_mask.shape} doesn't match data shape T={T}, N={N}, K={K}")
                
                # Crop to smallest size
                min_t = min(flip_mask.shape[0], T)
                min_n = min(flip_mask.shape[1], N)
                min_k = min(flip_mask.shape[2], K)
                
                flip_mask = flip_mask[:min_t, :min_n, :min_k]
                
                # Process subset of x with temporary variable
                x_temp = x[:min_t].reshape(min_t, N, K, C)[:, :min_n, :min_k]
                
                # Temporarily convert 2D grid to 1D  
                x_flat = x.reshape(T, N, K, C)
                
                # Generate sign tensor and apply restoration (subset)
                sign_temp = torch.where(
                    flip_mask.unsqueeze(-1) == 1,
                    torch.tensor(-1, device=x_flat.device, dtype=x_flat.dtype),
                    torch.tensor(1, device=x_flat.device, dtype=x_flat.dtype),
                )
                
                # Apply partial restoration
                x_flat[:min_t, :min_n, :min_k] = x_temp * sign_temp
                
                # Convert back to 2D grid format
                restored_videos[attr] = x_flat.reshape(T, H, W, K, C)
            else:
                # Standard processing when sizes match
                # Temporarily convert 2D grid to 1D  
                x_flat = x.reshape(T, N, K, C)  
                
                # Generate sign tensor and apply flip again (double flip returns to original)  
                sign = torch.where(  
                    flip_mask.unsqueeze(-1) == 1,  
                    torch.tensor(-1, device=x_flat.device, dtype=x_flat.dtype),  
                    torch.tensor(1, device=x_flat.device, dtype=x_flat.dtype),  
                )  
                
                # Apply flip to restore  
                x_restored = x_flat * sign  
                
                # Convert back to 2D grid format  
                restored_videos[attr] = x_restored.reshape(T, H, W, K, C)  
            
            print("SH coefficients restored successfully") 
            
        # Convert 2D grid format to 1D
        flattened_splats_videos = {}
        for attr_name, splats_video in restored_videos.items():
            ori_shape = list(splats_video.shape)
            new_shape = [ori_shape[0], ori_shape[1] * ori_shape[2]] + ori_shape[3:]
            flattened_splats_videos[attr_name] = splats_video.reshape(new_shape)

        # Separate data by frame
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
    n_clusters: int, 
    flip_info: Dict[str, torch.Tensor] = None,
    qp: Dict[str, int] = None, 
    debug: bool = False,
    use_all_intra: bool = False
) -> Dict[str, Any]:
    import imageio.v2 as imageio
    import json
    
    n_frames = int(params.size(0))
    H, W = params.shape[1], params.shape[2]
    device = params.device
    
    # Create debug directory
    debug_dir = os.path.join(compress_dir, "debug")
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Generate SH basis function name list
    shN_name_list = []
    for degree in range(1,4):
        for level in range(-degree, degree+1):
            shN_name_list.append(f"sh{degree}_{level}")
    
    # Cluster block labeling (using flip_info)
    if flip_info and "cluster_blocks_meta" in flip_info:
        print("Using cluster blocks metadata for labeling...")
        # Use cluster block metadata
        cluster_blocks_meta = flip_info["cluster_blocks_meta"]
        labeled_array = np.zeros((H, W), dtype=np.int32)
        
        # Label for each cluster block
        valid_blocks = 0
        for block_id_str, block_info in cluster_blocks_meta.items():
            y0, x0, y1, x1 = block_info["boundary"]
            
            # Convert string block_id to integer (safely)
            block_id = int(block_id_str)

            # Check and adjust block boundaries
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(H-1, y1), min(W-1, x1)
            
            # Check if block is valid
            if y1 >= y0 and x1 >= x0:
                valid_blocks += 1
                # Label block region
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        labeled_array[y, x] = block_id
    
    # Dictionary to save block metadata
    block_meta = {}
    
    # Initialize array to store normalized results
    shN_norm = np.zeros_like(params.cpu().numpy(), dtype=np.float32)
    print("Labeled array shape:", labeled_array.shape, "dtype:", labeled_array.dtype)

    # Process all blocks
    for block_id in range(1, n_clusters + 1):
        # Find pixels corresponding to this block
        block_mask_np = (labeled_array == block_id)
        block_pixel_count = np.sum(block_mask_np)
        
        # Skip empty blocks
        if block_pixel_count == 0:
            print(f"Block {block_id} is empty, skipping")
            continue
        
        # Find block boundaries
        y_indices, x_indices = np.where(block_mask_np)
        y0, x0 = np.min(y_indices), np.min(x_indices)
        y1, x1 = np.max(y_indices), np.max(x_indices)
        block_area = (y1-y0+1) * (x1-x0+1)
        
        print(f"Processing block {block_id}: boundary=({y0},{x0})-({y1},{x1}), pixels={block_pixel_count} (filling {block_pixel_count/block_area*100:.1f}% of area)")
        
        # Convert block mask to tensor
        block_mask = torch.from_numpy(block_mask_np).to(device)
        
        # Collect block data
        block_data = []
        for t in range(n_frames):
            # Extract all SH coefficients and RGB channels at corresponding block positions using mask
            frame_data = params[t].reshape(-1, params.shape[3], params.shape[4])[block_mask.reshape(-1)]  # [num_pixels, 15, 3]
            block_data.append(frame_data)
        
        # Combine all frame data
        block_data = torch.cat(block_data, dim=0)  # [n_frames*num_pixels, 15, 3]
        
        # Calculate min/max values for each SH basis function and RGB channel
        block_mins = torch.amin(block_data, dim=0)  # [15, 3]
        block_maxs = torch.amax(block_data, dim=0)  # [15, 3]
        
        # Calculate value range and add safety margin
        value_range = block_maxs - block_mins
        
        # Adjust if value range is too small (prevent division by zero + safety margin)
        epsilon = 1e-8
        adjusted_range = torch.where(value_range < epsilon, 
                                   torch.ones_like(value_range) * epsilon, 
                                   value_range)
        
        # Normalize each frame
        for t in range(n_frames):
            # Get indices of positions corresponding to this block's mask
            y_idx, x_idx = np.where(block_mask_np)
            
            for i, (y, x) in enumerate(zip(y_idx, x_idx)):
                # Normalize SH coefficients of current pixel
                pixel_data = params[t, y, x]  # [15, 3]
                
                # Perform normalization
                pixel_norm = (pixel_data - block_mins) / adjusted_range
                
                # Validate value range and clipping
                pixel_norm = torch.clamp(pixel_norm, 0.0, 1.0)
                
                # Save normalized value
                shN_norm[t, y, x] = pixel_norm.cpu().numpy()
        
        # Save block metadata
        block_meta[str(block_id)] = {
            "boundary": [int(y0), int(x0), int(y1), int(x1)],
            "mins": block_mins.tolist(),
            "maxs": block_maxs.tolist(),
            "block_id": int(block_id),
            "pixel_count": int(block_pixel_count)
        }
    
    # Quantization (8-bit)
    shN_uint8 = (shN_norm * (2**8 - 1)).round().astype(np.uint8)
    
    # Save PNG and compress video for each SH basis function
    file_extension = ".265" if debug else ".mp4"
    intra_params = ":keyint=1:min-keyint=1:scenecut=0" if use_all_intra else ""
    
    # Save PNG files by frame and SH basis
    for f_id in range(n_frames):
        for shN_id, shN_name in enumerate(shN_name_list):
            image = shN_uint8[f_id,:,:,shN_id,:]
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{shN_name}_frame{f_id:03d}.png"), image)
    
    # Compress video for each SH basis
    for shN_id, shN_name in enumerate(shN_name_list):
        sh_degree = shN_name[0:3]  # "sh1", "sh2", "sh3"
        sh_qp = qp[sh_degree] if sh_degree in qp else 24  # Default value 24
        
        print(f"QP value of {shN_name} is: {sh_qp}")
        video_file = os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}")
        cmd = f"ffmpeg -i {compress_dir}/{param_name}_{shN_name}_frame%03d.png -c:v libx265 -x265-params \"qp={sh_qp}{intra_params}\" {video_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Clean up temporary PNG files
    if not debug:
        png_files = sorted(glob.glob(os.path.join(compress_dir, f"{param_name}_*.png")))
        for png_file in png_files:
            os.remove(png_file)
    
    # Save block metadata
    block_meta_path = os.path.join(compress_dir, f"{param_name}_blocks_meta.json")
    with open(block_meta_path, 'w') as f:
        json.dump(block_meta, f, indent=2)
    
    # Overall metadata
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "file_extension": file_extension,
        "block_meta_file": f"{param_name}_blocks_meta.json",
        "shN_name_list": shN_name_list
    }
    
    return meta

def _decompress_shN_video_hevc(
    compress_dir: str, param_name: str, meta: Dict[str, Any], flip_info: Dict[str, torch.Tensor] = None,
):
    import imageio.v2 as imageio
    import json
    import numpy as np

    # Get SH basis function name list
    shN_name_list = meta.get("shN_name_list", [])
    if not shN_name_list:
        # Generate default values if not in metadata
        for degree in range(1,4):
            for level in range(-degree, degree+1):
                shN_name_list.append(f"sh{degree}_{level}")

    # Load required metadata
    file_extension = meta["file_extension"]
    output_shape = meta["shape"]
    n_frames = output_shape[0]
    H, W = output_shape[1], output_shape[2]

    # Load block metadata and labels
    # When cluster block metadata is passed
    block_meta_file = meta.get("block_meta_file", None)
    if block_meta_file:
        block_meta_path = os.path.join(compress_dir, block_meta_file)
        if not os.path.exists(block_meta_path):
            print(f"Warning: Block metadata file not found: {block_meta_path}")
            cluster_blocks_meta = {}
        else:
            with open(block_meta_path, 'r') as f:
                cluster_blocks_meta = json.load(f)

    if cluster_blocks_meta:
        # Use cluster block metadata
        labeled_array = np.zeros((H, W), dtype=np.int32)
        
        # Label for each cluster block
        valid_blocks = 0
        for block_id_str, block_info in cluster_blocks_meta.items():
            y0, x0, y1, x1 = block_info["boundary"]
            
            # Convert string block_id to integer (safely)
            block_id = int(block_id_str)
            
            # Check and adjust block boundaries
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(H-1, y1), min(W-1, x1)
            
            # Check if block is valid
            if y1 >= y0 and x1 >= x0:
                valid_blocks += 1
                # Label block region
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        labeled_array[y, x] = block_id
        
        # Construct block_meta
        block_meta = {}
        for block_id_str, block_info in cluster_blocks_meta.items():
            block_id = int(block_id_str)
            block_meta[block_id_str] = {
                "block_id": block_id,
                "mins": block_info["mins"],
                "maxs": block_info["maxs"],
                "boundary": block_info["boundary"]
            }

    # Check number of labeled pixels (debugging)
    labeled_pixel_count = np.sum(labeled_array > 0)
    total_pixels = H * W
    print(f"Decompress: Using {valid_blocks} valid blocks out of {len(cluster_blocks_meta)} metadata blocks")
    print(f"Labeled pixels: {labeled_pixel_count}/{total_pixels} ({labeled_pixel_count/total_pixels*100:.2f}%)")
    
    # If not all pixels are labeled, treat remaining pixels as separate blocks
    if labeled_pixel_count < total_pixels:
        print(f"Warning: {total_pixels-labeled_pixel_count} pixels are not assigned to any cluster block during decompression!")
    
    # Load videos
    shN_reader_list = []
    for shN_name in shN_name_list:
        video_path = os.path.join(compress_dir, f"{param_name}_{shN_name}.{file_extension[1:]}")
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            # Add empty reader
            shN_reader_list.append(None)
            continue
        
        try:
            reader = imageio.get_reader(video_path, format='FFMPEG')
            shN_reader_list.append(reader)
        except Exception as e:
            print(f"Error opening {shN_name}: {str(e)}")
            shN_reader_list.append(None)
    
    # Load frames for each SH basis function
    shN_frames_list = []
    for i, reader in enumerate(shN_reader_list):
        if reader is None:
            # Create empty frames if no reader
            frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
            shN_frames_list.append(frames)
            continue
        
        frames = []
        for frame in reader:
            frames.append(frame)
        
        if len(frames) < n_frames:
            # Fill missing frames
            for _ in range(n_frames - len(frames)):
                frames.append(np.zeros_like(frames[0]))
        
        shN_frames_list.append(np.stack(frames, axis=0))
        reader.close()
    
    # Stack all frames
    shN_videos = np.stack(shN_frames_list, axis=3)  # [T, H, W, 15, 3]
    
    # Convert to 8-bit normalized values
    shN_norm = shN_videos.astype(np.float32) / 255.0
    
    # Initialize result tensor
    result = torch.zeros(output_shape, dtype=getattr(torch, meta["dtype"]))
    
    # Apply denormalization for each block
    for block_id, block_info in block_meta.items():
        block_id_int = int(block_info.get("block_id", block_id))
        
        # Generate block mask
        block_mask = (labeled_array == block_id_int)
        if not np.any(block_mask):
            print(f"Warning: Block {block_id} has no pixels")
            continue
        
        # Check block boundaries (for efficiency)
        y_indices, x_indices = np.where(block_mask)
        y_min, y_max = min(y_indices), max(y_indices)
        x_min, x_max = min(x_indices), max(x_indices)
        
        # Get block mins/maxs
        block_mins = torch.tensor(block_info["mins"], dtype=getattr(torch, meta["dtype"]))
        block_maxs = torch.tensor(block_info["maxs"], dtype=getattr(torch, meta["dtype"]))
        block_range = block_maxs - block_mins
        
        # Denormalize for all pixels in block
        for t in range(n_frames):
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if not block_mask[y, x]:
                        continue
                    
                    # For each SH basis function
                    for sh_id in range(len(shN_name_list)):
                        # Get normalized value (including RGB channels)
                        normalized = torch.tensor(shN_norm[t, y, x, sh_id, :], 
                                                dtype=getattr(torch, meta["dtype"]))
                        
                        # Denormalize to original value range (calculated per RGB channel)
                        denormalized = normalized * block_range[sh_id] + block_mins[sh_id]
                        
                        # Save result
                        result[t, y, x, sh_id] = denormalized
    
    result = result.to(dtype=getattr(torch, meta["dtype"]))
    return result 

def _compress_npz(
    compress_dir: str, param_name: str, params, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    # Save as json if it's a dictionary
    if isinstance(params, dict):
        save_fp = os.path.join(compress_dir, f"{param_name}.json")
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        with open(save_fp, 'w') as f:
            json.dump(params, f)
        meta = {
            "is_dict": True,
            "shape": None,
            "dtype": "dict",
        }
    else:  # If it's a tensor
        npz_dict = {"arr": params.detach().cpu().numpy()}
        save_fp = os.path.join(compress_dir, f"{param_name}.npz")
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        np.savez_compressed(save_fp, **npz_dict)
        meta = {
            "is_dict": False,
            "shape": params.shape,
            "dtype": str(params.dtype).split(".")[1],
        }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any]):
    """Decompress parameters with numpy's NPZ compression or JSON."""
    # Load from json if it's a dictionary
    if meta.get("is_dict", False):
        with open(os.path.join(compress_dir, f"{param_name}.json"), 'r') as f:
            params = json.load(f)
        return params
    else:  # Load from npz if it was a tensor
        arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
        params = torch.tensor(arr)
        params = params.reshape(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
        return params