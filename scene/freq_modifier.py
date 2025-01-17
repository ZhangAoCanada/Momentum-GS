import os, sys
import torch.nn as nn
import torch.nn.functional as F 
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.signal import savgol_filter

class FrequencyModifier(nn.Module):
    def __init__(self, dataset, scene, opt, pipe, tb_writer):
        super().__init__()
        self.dataset = dataset
        self.scene = scene
        self.opt = opt
        self.pipe = pipe
        self.tb_writer = tb_writer
        self.freq_interval = self.dataset.voxel_size
    

    def anchor_freq_smooth(self, gs, freq_scale=10., mode="anchor_offset", smooth_win=31):
        assert mode in ["anchor_offset", "anchor_scale", "anchor_all"]
        if mode == "anchor_all":
            max_scale = gs.get_scaling.max(dim=-1).values
        elif mode == "anchor_scale":
            max_scale = gs.get_scaling[:, 3:].max(dim=-1).values # anchor scale
        elif mode == "anchor_offset":
            max_scale = gs.get_scaling[:, :3].max(dim=-1).values # anchor offset scale
        else:
            raise NotImplementedError
        scale_freq = max_scale // (self.freq_interval * freq_scale)
        freq_unique, freq_counts = torch.unique(scale_freq, sorted=True, return_counts=True)
        assert smooth_win % 2 == 1
        kernel_size = smooth_win
        stride = 1
        padding = kernel_size // 2
        freq_counts_new = freq_counts.float()
        # freq_counts_new = F.avg_pool1d(freq_counts_new.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride, padding=padding).squeeze(0).squeeze(0)
        freq_counts_new = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)(freq_counts_new.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return {"freq": scale_freq,
                "freq_unique": freq_unique,
                "freq_counts": freq_counts_new}


    def anchor_freq(self, gs, freq_scale=1., mode="anchor_offset"):
        assert mode in ["anchor_offset", "anchor_scale", "anchor_all"]
        if mode == "anchor_all":
            max_scale = gs.get_scaling.max(dim=-1).values
        elif mode == "anchor_scale":
            max_scale = gs.get_scaling[:, 3:].max(dim=-1).values # anchor scale
        elif mode == "anchor_offset":
            max_scale = gs.get_scaling[:, :3].max(dim=-1).values # anchor offset scale
        else:
            raise NotImplementedError
        scale_freq = max_scale // (self.freq_interval * freq_scale)
        freq_unique, freq_counts = torch.unique(scale_freq, sorted=True, return_counts=True)
        return {"freq": scale_freq,
                "freq_unique": freq_unique,
                "freq_counts": freq_counts}


    def filter_smooth(self, freq, freq_scale=10., threshold=0.8, min_freq=1000):
        frequency, freq_unique, freq_counts = freq["freq"], freq["freq_unique"], freq["freq_counts"]
        min_freq_index = int(min_freq // freq_scale)
        min_index = torch.argmin(torch.abs(freq_unique - min_freq_index))
        start_index = min_index
        chunk_size = min(start_index, 200)
        freq_counts_diff = freq_counts[start_index-chunk_size:-chunk_size*2] * threshold < freq_counts[start_index+chunk_size:]
        if freq_counts_diff.sum() == 0:
            index = len(freq_unique) - 1
        else:
            index = torch.argmax(freq_counts_diff.float()) + start_index
        freq_threshold = freq_unique[index]
        freq_mask = frequency < freq_threshold
        freq_scale_ratio = freq_threshold / frequency
        freq_unique_masked = freq_unique[freq_unique < freq_threshold] * freq_scale
        freq_counts_masked = freq_counts[freq_unique < freq_threshold]
        return {"freq_mask": freq_mask,
                "freq_scale_ratio": freq_scale_ratio,
                "freq_unique": freq_unique_masked,
                "freq_counts": freq_counts_masked}


    def filter_freq(self, freq, mode="unique", k=500, modifier=10):
        frequency, freq_unique, freq_counts = freq["freq"], freq["freq_unique"], freq["freq_counts"]
        counts_median = torch.median(freq_counts)
        if mode == "counts":
            ### NOTE: filter 1: counts median filter ###
            mask = freq_counts > counts_median * modifier
        elif mode == "unique":
            ### NOTE: filter 2: freq unique topk filter ###
            k = min(k, len(freq_unique))
            _, indices = torch.topk(freq_counts, k)
            mask = torch.zeros_like(freq_unique, dtype=torch.bool)
            mask[indices] = 1
        else:
            raise NotImplementedError
        freq_unique_masked = freq_unique[mask]
        freq_counts_masked = freq_counts[mask]
        freq_mask = torch.zeros_like(frequency)
        for scale in freq_unique_masked:
            freq_mask = torch.where(frequency == scale, 1, freq_mask)
        freq_mask = freq_mask.bool()
        return {"freq_mask": freq_mask,
                "freq_unique": freq_unique_masked,
                "freq_counts": freq_counts_masked}


    def filter_freq_direct(self, freq, k=1000):
        frequency, freq_unique, freq_counts = freq["freq"], freq["freq_unique"], freq["freq_counts"]
        k = min(k, freq_unique.max())
        freq_mask = frequency < k
        mask = freq_unique < k
        freq_unique = freq_unique[mask]
        freq_counts = freq_counts[mask]
        return {"freq_mask": freq_mask,
                "freq_unique": freq_unique,
                "freq_counts": freq_counts}
    


    def plot_freq(self, freq, save_name):
        freq_unique, freq_counts = freq["freq_unique"], freq["freq_counts"]
        freq_counts = torch.log(freq_counts.float() + 1)
        figure, ax = plt.subplots()
        ax.plot(freq_unique.cpu().numpy(), freq_counts.cpu().numpy())
        ax.set_title('Frequency of each scale')
        ax.set_xlabel('Scale')
        ax.set_ylabel('Frequency')
        plt.savefig(save_name)
        plt.close()


    def plot_freq_compare(self, freq, freq_filtered, save_name):
        freq_n, freq_counts = freq["freq_unique"], freq["freq_counts"]
        freq_filtered_n, freq_counts_filtered = freq_filtered["freq_unique"], freq_filtered["freq_counts"]
        freq_counts = torch.log(freq_counts.float() + 1)
        freq_counts_filtered = torch.log(freq_counts_filtered.float() + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].plot(freq_n.cpu().numpy(), freq_counts.cpu().numpy())
        axes[0].set_title('Frequency of before filtering')
        axes[0].set_xlabel('Scale')
        axes[0].set_ylabel('Frequency')
        axes[1].plot(freq_filtered_n.cpu().numpy(), freq_counts_filtered.cpu().numpy())
        axes[1].set_title('Frequency of after filtering')
        axes[1].set_xlabel('Scale')
        axes[1].set_ylabel('Frequency')
        plt.savefig(save_name)
        plt.close()
        

