# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from inspect import formatargvalues
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils


# ----------------------------------------------------------------------------


class PCAWidget:
    def __init__(self, viz):
        self.viz = viz
        self.num_components = 100
        self.vals = np.zeros(self.num_components)
        self.pca = np.load("edit/w_pca_100_components_1000000_iterations.npz")

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            if imgui_utils.button('Reset All', width=-1, enabled=True):
                self.vals = np.zeros(self.num_components)
            for i in range(self.num_components):
                imgui.text(f'PCA {i}')
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(viz.font_size * 40):
                    _changed, self.vals[i] = imgui.slider_float(f'##PCA{i}', self.vals[i], -5, 5, format='%.2f')
                imgui.same_line()
                if imgui_utils.button(f'Reset {i}', enabled=True):
                    self.vals[i] = 0

        viz.w_offset = np.sum(self.vals[:, None] * self.pca["components"], axis=0)
# ----------------------------------------------------------------------------
