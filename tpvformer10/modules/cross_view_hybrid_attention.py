
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import math
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TPVCrossViewHybridAttention(BaseModule):
    
    def __init__(self, 
        tpv_h, tpv_w, tpv_z,
        embed_dims=256, 
        num_heads=8, 
        num_points=4,
        num_anchors=2,
        init_mode=0,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = 3             # 表示TPV的三个平面，而不是特征图的三个层级
        self.num_points = num_points    # 表示每个参考点的采样点数量
        self.num_anchors = num_anchors
        self.init_mode = init_mode
        self.dropout = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(3)
        ])
        self.output_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(3)
        ])
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * num_points * 2) for _ in range(3)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * 3 * (num_points + 1)) for _ in range(3)
        ])
        self.value_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(3)
        ])

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        # self plane
        theta_self = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_self = torch.stack([theta_self.cos(), theta_self.sin()], -1) # H, 2
        grid_self = grid_self.view(
            self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for j in range(self.num_points):
            grid_self[:, j, :] *= (j + 1) / 2

        if self.init_mode == 0:
            # num_phi = 4
            phi = torch.arange(4, dtype=torch.float32) * (2.0 * math.pi / 4) # 4
            assert self.num_heads % 4 == 0
            num_theta = int(self.num_heads / 4)
            theta = torch.arange(num_theta, dtype=torch.float32) * (math.pi / num_theta) + (math.pi / num_theta / 2) # 3
            x = torch.matmul(theta.sin().unsqueeze(-1), phi.cos().unsqueeze(0)).flatten()
            y = torch.matmul(theta.sin().unsqueeze(-1), phi.sin().unsqueeze(0)).flatten()
            z = theta.cos().unsqueeze(-1).repeat(1, 4).flatten()
            xyz = torch.stack([x, y, z], dim=-1) # H, 3

        elif self.init_mode == 1:
            
            xyz = [
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0]
            ]
            xyz = torch.tensor(xyz, dtype=torch.float32)

        grid_hw = xyz[:, [0, 1]] # H, 2
        grid_zh = xyz[:, [2, 0]]
        grid_wz = xyz[:, [1, 2]]

        for i in range(3):
            grid = torch.stack([grid_hw, grid_zh, grid_wz], dim=1) # H, 3, 2
            grid = grid.unsqueeze(2).repeat(1, 1, self.num_points, 1)
            
            grid = grid.reshape(self.num_heads, self.num_levels, self.num_anchors, -1, 2)
            for j in range(self.num_points // self.num_anchors):
                grid[:, :, :, j, :] *= 2 * (j + 1)
            grid = grid.flatten(2, 3)
            grid[:, i, :, :] = grid_self
            
            constant_init(self.sampling_offsets[i], 0.)
            self.sampling_offsets[i].bias.data = grid.view(-1)

            constant_init(self.attention_weights[i], val=0., bias=0.)
            attn_bias = torch.zeros(self.num_heads, 3, self.num_points + 1)
            attn_bias[:, i, -1] = 10
            self.attention_weights[i].bias.data = attn_bias.flatten()
            xavier_init(self.value_proj[i], distribution='uniform', bias=0.)
            xavier_init(self.output_proj[i], distribution='uniform', bias=0.)    
    
    # 以query为输入，通过全连接网络映射为采样偏移量和注意力权重
    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            # fc是采样偏移的全连接网络，以query为输入，输出的形状是[bs, l, num_heads, num_levels, num_points, 2]
            # l表示每个平面的查询数量，对于hw平面，l=tpv_h*tpv_w，对于zh平面，l=tpv_z*tpv_h，对于wz平面，l=tpv_w*tpv_z
            # num_levels=3表示TPV的三个平面，而不是特征图的三个层级
            # num_points表示每个参考点的总采样点数量
            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels, self.num_points, 2)
            offsets.append(offset)

            # attn是注意力权重的全连接网络，以query为输入，输出形状是[bs, l, H, 3, 4+1]
            attention = attn(query).reshape(bs, l, self.num_heads, 3, -1)
            level_attention = attention[:, :, :, :, -1:].softmax(-2) # bs, l, H, 3, 1
            attention = attention[:, :, :, :, :-1]
            attention = attention.softmax(-1) # bs, l, H, 3, p
            attention = attention * level_attention
            attns.append(attention)
        
        # 将三个平面的采样偏移量和注意力权重沿第1维度连接起来，形状是[bs, l, num_heads, num_levels, num_points, 2]和[bs, l, H, 3, p]
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(output, [lens[0], lens[1], lens[2]], dim=1)
        return outputs

    def forward(self,                
                query,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        identity = query if identity is None else identity
        if query_pos is not None:
            query = [q + p for q, p in zip(query, query_pos)]

        # 通过对query进行全连接映射，获得三个视图对应的value，连接成一个张量，并分配到多个检测头
        # value proj
        query_lens = [q.shape[1] for q in query]
        # 分别对三个查询进行全连接映射，获得三个视图对应的value，形状是[bs, num_query, embed_dims]
        value = [layer(q) for layer, q in zip(self.value_proj, query)]
        # 将三个视图的value沿第1维度连接起来，形状是[bs, num_query, embed_dims]
        value = torch.cat(value, dim=1)
        bs, num_value, _ = value.shape
        # 将value的embed_dims维度分配到多个检测头，形状变成[bs, num_query, num_heads, embed_dims/num_heads]
        value = value.view(bs, num_value, self.num_heads, -1)

        # 以query为输入，通过全连接网络映射为采样偏移量和注意力权重
        # 其中sampling_offsets的形状从上下文来看是[bs, num_query, num_heads, num_levels, num_all_points, xy]
        # 其中num_all_points == num_points * num_Z_anchors
        # FIXME: 但是从get_sampling_offsets_and_attention的实现来看，num_all_points似乎等于num_points
        # sampling offsets and weights
        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)

        # reference_points的形状是[bs, hw+zh+wz, 3, p, 2]
        # 其中hw+zh+wz就是num_query
        # 其中3就是num_levels，表示TPV的三个平面，而不是特征图的三个层级
        # 其中p=32就是num_Z_anchors
        # 其中2就是xy

        # 确认参考点的最后一个维度是2
        if reference_points.shape[-1] == 2:
            """
            For each tpv query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each tpv query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """

            # 对于每个 tpv 查询，它在 3D 空间中拥有 num_Z_anchors 个不同高度的锚点。
            # 在投影之后，每个 tpv 查询在每个 2D 图像中有 num_Z_anchors 个参考点。
            # 对于每个参考点，我们采样 num_points 个采样点。
            # 对于 num_Z_anchors 个参考点来说，总共有 num_points * num_Z_anchors 个采样点。

            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # reference_points的形状是[bs, hw+zh+wz, 3, p, 2]，其中p=32就是num_Z_anchors
            bs, num_query, _, num_Z_anchors, xy = reference_points.shape
            # 给参考点扩展两个维度，分别是num_heads和num_points，形状变成：
            # [bs, hw+zh+wz, num_heads, 3, num_Z_anchors, num_points, 2]
            reference_points = reference_points[:, :, None, :, :, None, :]
            # 对采样偏移量进行归一化
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            # 下面的num_all_points = num_points * num_Z_anchors
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors, num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            # 下面的代码似乎把num_points和num_Z_anchors弄反了，不过没关系，后续也不会再用到它们
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            # 把采样位置的num_points和num_Z_anchors维度合并，形状变成[bs, num_query, num_heads, num_levels, num_all_points, 2]
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2, but get {reference_points.shape[-1]} instead.')
        
        if torch.cuda.is_available():
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        outputs = self.reshape_output(output, query_lens)

        results = []
        for out, layer, drop, residual in zip(outputs, self.output_proj, self.dropout, identity):
            results.append(residual + drop(layer(out)))

        return results
