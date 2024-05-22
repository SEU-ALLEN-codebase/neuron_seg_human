from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from nnunetv2.training.loss.fmm.fmm_process import get_fmm_from_img
import time
import os
import cc3d


class AddPredecessorImageTransform(AbstractTransform):
    def __init__(self, predecessor_key: str = "predecessor", target_key: str = "target", soma_key: str = "soma"):
        """
        Adds the predecessor image to the data dictionary. This transform assumes that the predecessor
        image is already computed and available in the data_dict under the key specified by predecessor_key.
        """
        self.predecessor_key = predecessor_key
        self.target_key = target_key
        self.soma_key = soma_key

    def largest_connected_component(self, input_image):
        # 使用cc3d计算所有连通块
        labels_out, N = cc3d.connected_components(input_image, connectivity=26, return_N=True)

        # 如果没有找到连通块，返回None
        if N == 0:
            return None

        # 计算每个连通块的大小
        component_sizes = np.bincount(labels_out.flat)

        # 获取最大连通块的标签
        largest_component_label = component_sizes[1:].argmax() + 1  # 忽略背景标签0

        # 返回最大连通块
        largest_component = labels_out == largest_component_label

        # 检查最大连通块的比例
        # if(np.sum(largest_component) / np.sum(input_image) < 0.5):
        #     print(f"ratio: {np.sum(largest_component) / np.sum(input_image)}")

        return largest_component

    def __call__(self, **data_dict):
        target_shape = data_dict[self.target_key].shape
        data_dict[self.predecessor_key] = np.zeros(target_shape, dtype=np.int32) # (2, 1, 48, 224, 224)
        data_dict[self.soma_key] = np.zeros(tuple([target_shape[0], target_shape[1], 3]), dtype=np.int32)
        target = data_dict[self.target_key]

        # print(target[0].shape, target.dtype, len(target))

        for batch_idx in range(target.shape[0]):
            for channel_idx in range(target.shape[1]):
                current_target = target[batch_idx, channel_idx]
                max_cc = self.largest_connected_component(current_target)
                current_predecessor, current_soma = get_fmm_from_img(max_cc)
                if((current_predecessor is None) or (current_soma is None)):
                    # print("... no predecessor found for batch_idx: ", batch_idx, " channel_idx: ", channel_idx)
                    data_dict[self.predecessor_key][batch_idx, channel_idx] = np.ones_like(current_target) * -1
                    data_dict[self.soma_key][batch_idx, channel_idx] = np.zeros(3)
                else:
                    data_dict[self.predecessor_key][batch_idx, channel_idx] = current_predecessor
                    data_dict[self.soma_key][batch_idx, channel_idx] = current_soma

        return data_dict