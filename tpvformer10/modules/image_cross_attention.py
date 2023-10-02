
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32
from mmcv.runner.base_module import BaseModule

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TPVImageCrossAttention(BaseModule):
    """An attention module used in TPVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=True,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                 **kwargs
                 ):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'reference_points_cams'))
    def forward(self,
                query,  # 将三个视图query沿num_query维度连接起来的合并query，形状是[bs, num_query, embed_dims]
                key,
                value,
                residual=None,
                spatial_shapes=None,
                reference_points_cams=None, # list，为三视图分别生成的三组3D参考点，像素坐标，每个元素的形状是[N,B,H*W,D,2]
                tpv_masks=None, # list类型，为三视图分别生成的三组mask，标记有效的参考点，形状是（N,B,H*W,D）
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_key, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_key, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        bs, num_query, _ = query.size()

        # 将TPVFormerLayer中连接起来的query又切分成三个视图的query
        queries = torch.split(query, [self.tpv_h*self.tpv_w, self.tpv_z*self.tpv_h, self.tpv_w*self.tpv_z], dim=1)
        if residual is None:
            slots = [torch.zeros_like(q) for q in queries]
        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []

        # 逐一遍历三个视图，根据mask提取有效的query和参考点，重新构造query和参考点张量，使query和参考点Pillar一一对应，并降低计算量。
        # 重新构造的query和参考点按照相机图像的顺序进行了分组，即属于同一个相机图像的query和参考点放在一起
        # 这样就便于将TPV三个视图的query和参考点合并到一起，即同一个相机的query和参考点放在一起，简化MSDA的采样过程
        # tpv_mask形状是（N,B,H*W,D），第0维度是num_cams
        for tpv_idx, tpv_mask in enumerate(tpv_masks):
            indexes = []
            # 取得每个相机图像对应的参考点的mask，形状是[B,H*W,D]
            # tpv_masks的形状和参考点的形状是一样的，表示每个参考点是否有效
            # 假设总共有6个相机图像，那么这里就是分别取得每个相机对应的mask，形状是[B,H*W,D]
            # 找到每个相机图像下，每个有效query的索引，即在BEV网格中的位置
            # 所谓有效query是指对应的柱体采样点中至少有一个点是有效的，是落在了当前图像上的
            for _, mask_per_img in enumerate(tpv_mask):
                # 首先取第0批次的mask，形状是[H*W,D]，同一个批次的情况都是一样的
                # 然后对最后一个维度求和，获得每个BEV网格（即每个query）对应的有效点数，形状是[H*W]
                # 然后获得非零元素的索引，形状是[N, 1]，N是非零元素的数量，1是非零元素的索引，
                # 如果某个query对应的有效点数是0，那么这里就没有它的索引，也就是被剔除了，保留下来的都是有效的query
                # 最后去掉最后一个维度，获得有效query对应的索引，即在每个BEV网格中的位置，形状是[N]
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            # 取得若干个相机图像中，有效query的最大数量，对于某一个TPV视图来说，这个数量是一样的
            max_len = max([len(each) for each in indexes])
            max_lens.append(max_len)
            indexeses.append(indexes)

            # 取得当前视图的reference points，形状是[N, B, H*W, D, 2]
            reference_points_cam = reference_points_cams[tpv_idx]
            D = reference_points_cam.size(3)

            # 根据当前视图的每图像最大有效query值，构造新的query张量，形状是[bs*num_cams, max_len, embed_dims]，初始化为全0
            queries_rebatch = queries[tpv_idx].new_zeros(
                [bs * self.num_cams, max_len, self.embed_dims])
            # 根据当前视图的每图像最大有效query值，构造新的参考点张量，形状是[bs*num_cams, max_len, D, 2]，初始化为全0
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs * self.num_cams, max_len, D, 2])

            # reference_points_cam的原始形状是[N, B, H*W, D, 2]，第0维度是图像索引，第1维度是bs

            # 将有效query和对应的参考点提取出来，重组成新的张量，剔除无效query和参考点，降低计算量；
            # 新的query形状是[bs*num_cams, max_len, embed_dims]，max_len是若干相机图像中有效query的最大数量，不同TPV视图的max_len可能不一样
            # 新的参考点形状是[bs*num_cams, max_len, D, 2]，max_len是若干相机图像中有效query的最大数量，不同TPV视图的max_len可能不一样
            # 新的张量是一个稀疏张量，有些元素是0
            for i, reference_points_per_img in enumerate(reference_points_cam):
                for j in range(bs):
                    index_query_per_img = indexes[i]
                    queries_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = queries[tpv_idx][j, index_query_per_img]
                    reference_points_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
            
            # 三个视图的查询和参考点以list的方式分别存储，并没有合并在一起
            queries_rebatches.append(queries_rebatch)
            reference_points_rebatches.append(reference_points_rebatch)

            # query的初始形状是[bs, num_query, embed_dims]，其中num_query是每个TPV视图的查询数量，即BEV网格的数量。
            # 以hw视图为例，num_query = tpv_h * tpv_w，与相机图像并无关系。但是在交叉注意力机制中query就是用来构造
            # 采样偏移和注意力权重的，为了给每个图像对应的每个参考点构造采样偏移和注意力权重，需要找到每个参考点对应的
            # query。

            # 而参考点是基于query构造的，每个query（BEV网格）对应一个pillar的D个参考点，然后再通过相机参数投射到每个
            # 图像，在剔除无效点之前，参考点的总数是num_query * D * N，其中N是相机图像数量，D是每个pillar的参考点数量。
            # 
            # 上面的重排列相当于使query扩充了N倍，然后再剔除无效的部分，使得query和参考点前两维度的形状一致：
            # query: [bs*num_cams, max_len, embed_dims]
            # 参考点: [bs*num_cams, max_len, D, 2]
            # 使得query和参考点在bs、num_cams、num_query（即BEV网格）上是一一对应的，其中有大量的query是重复的，
            # 每个query（即BEV网格）与参考点Pillar一一对应。

        # 特征图此时的形状是[N, H*W++, B, C]，其中N是相机图像数量，B是bs，H*W++是多尺度特征图的像素数量，C是嵌入维度
        num_cams, l, bs, embed_dims = key.shape

        # 将特征图重排列成[N*B, H*W++, C]的形状
        # 之所以这样重排列，是为了配合将三个视图的采样点合并到一起，简化MSDA的采样过程
        key = key.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)
        value = value.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)

        # 调用deformable_attention进行注意力计算
        queries = self.deformable_attention(
            query=queries_rebatches, key=key, value=value,
            reference_points=reference_points_rebatches, 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,)
        
        for tpv_idx, indexes in enumerate(indexeses):
            for i, index_query_per_img in enumerate(indexes):
                for j in range(bs):
                    slots[tpv_idx][j, index_query_per_img] += queries[tpv_idx][j * self.num_cams + i, :len(index_query_per_img)]

            count = tpv_masks[tpv_idx].sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots[tpv_idx] = slots[tpv_idx] / count[..., None]
        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class TPVMSDeformableAttention3D(BaseModule):
    """An attention module used in tpvFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=[8, 64, 64],
                 num_z_anchors=[4, 32, 32],
                 pc_range=None,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 floor_sampling_offset=True,
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [points // self.base_z_anchors for points in num_z_anchors]
        self.pc_range = pc_range
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2) for i in range(3)
        ])
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i]) for i in range(3)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for i in range(3):
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1,
                2).repeat(1, self.num_levels, self.num_points[i], 1)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels, self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1
        
            self.sampling_offsets[i].bias.data = grid_init.view(-1)
            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    # @info 对三个视图的query通过全连接网络进行映射，获得采样偏移和注意力权重，并分配到多个注意力头上
    #       将三个视图对应的采样偏移和注意力权重沿第一维度连接成一个张量
    # @param queries: list，每个元素的形状是[num_cams*bs, num_query, embed_dims]
    # @return offsets: 采样偏移，形状是[bs, num_query, num_heads, num_levels, num_points, 2]
    # @return attns: 注意力权重，形状是[bs, num_query, num_heads, num_levels, num_points]
    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []

        # 不论采样偏移，还是注意力权重，全连接网络映射都是针对每一个查询向量的（即embed_dims），与查询的数量num_query无关。
        # 以采样偏移为例，输入是长度为embed_dims的查询向量，输出是num_heads * num_levels * num_points[i] * 2的张量。
        # 也就是说全连接网络负责把每个查询向量分配到每个注意力头、每个层级、每个采样点。所有的查询向量共用一个全连接网络。
        # 并没有针对每个查询向量单独构造全连接网络，因此虽然是全连接网络，但总的参数量很少，且数量固定。

        # 遍历三个视图的query
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape

            # 对query进行线性变换，获得采样偏移，并分配到多个注意力头上，
            # 形状变成[num_cams*bs, num_query, num_heads, num_levels, num_points, 2]
            # 其中num_points=[8, 64, 64]，num_levels=4，num_heads=8
            # 然后将形状改成[num_cams*bs, num_query, num_heads, num_levels, pm, num_points, 2]，pm=[1, 8, 8]，num_points统一成了8
            offset = fc(query).reshape(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1, 2)
            # 然后将形状改成[num_cams*bs, num_query, pm, num_heads, num_levels, num_points, 2]，并将num_query和pm合并成一个维度
            # 变成[num_cams*bs, num_query*pm, num_heads, num_levels, num_points, 2]
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)

            # 对query进行线性变换，获得注意力权重，并分配到多个注意力头上，
            # 形状变成[num_cams*bs, num_query, num_heads, num_levels, num_points]
            # 其中num_points=[8, 64, 64]，num_levels=4，num_heads=8
            # 然后将num_levels和num_points两个维度合并，进行softmax归一化
            attention = attn(query).reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            # 再将形状转换成[num_cams*bs, num_query, num_heads, num_levels, pm, num_points]
            attention = attention.view(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1)
            # 再将形状转换成[num_cams*bs, num_query, pm, num_heads, num_levels, num_points]，并将num_query和pm合并成一个维度
            # 变成[num_cams*bs, num_query*pm, num_heads, num_levels, num_points]
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)

            # 由于三个TPV视图的参考点Pillar数量不一样，为了便于MSDA处理，需要使参考点Pillar长度一致，每参考点采样点数量一致。
            # 上面的代码在完成全连接映射后，将num_points统一成了8，多出来的参考点合并到了num_query维度上。
        
        # 将三个视图的采样偏移和注意力权重沿num_query维度连接成一个张量
        # 连接后三个视图的采样偏移和注意力权重自动按照相机图像的顺序进行了分组，即属于同一个相机图像的采样偏移和注意力权重放在一起
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    # @info 三个TPV视图的参考点Pillar数量不一样，为了便于MSDA处理，需要使参考点Pillar长度一致，每参考点采样点数量一致。
    #       将三视图的参考点张量形状重整为[N*B, H*W*PM, D, 2]，PM=[1, 8, 8]，D统一成了4，然后连接成一个张量。
    # @param reference_points: 是一个list，有三个张量，每个张量对应一个视图，每个张量的形状是[N*B, H*W, D, 2]
    #        hw、zh、wz三视图的D分别是[4, 32, 32]，将三视图的D统一成4，zh和wz分出来的8合并到H*W维度上
    # @return reference_points: 重整并连接后的参考点张量，形状是[bs, num_query, num_Z_anchors, xy]
    #         bs是num_cams*bs，num_query是三视图的H*W*PM之和，num_Z_anchors统一都是4，xy是2
    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        # 遍历每个视图对应的参考点
        for i, reference_point in enumerate(reference_points):
            # bs=N*B, l=H*W, z_anchors=D
            bs, l, z_anchors, _  = reference_point.shape
            # 增加一个维度，形状变成[N*B, H*W, 1, D, 2]
            # points_multiplier=[1, 8, 8]，hw、zh、wz三视图的D分别是[4, 32, 32]
            # 三视图的形状变成[N*B, H*W, PM, D, 2]，PM是points_multiplier，D统一成了4
            reference_point = reference_point.reshape(bs, l, self.points_multiplier[i], -1, 2)
            # 将HW和PM两个维度合并成一个维度，使参考点形状变成[N*B, H*W*PM, D, 2]
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
            # 将三视图的参考点沿第一维度连接成一个张量，形状是[N*B, H*W*PM, D, 2]
        return torch.cat(reference_point_list, dim=1)
    
    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(output, [lens[0]*self.points_multiplier[0], lens[1]*self.points_multiplier[1], lens[2]*self.points_multiplier[2]], dim=1)
        
        outputs = [o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2) for i, o in enumerate(outputs)]
        return outputs

    def forward(self,
                query,  # list，每个元素的形状是[num_cams*bs, num_query, embed_dims]
                key=None,
                value=None, # 2D特征图，形状是[num_cams*bs, H*W++, embed_dims]
                identity=None,
                reference_points=None,  # list，每个元素的形状是[N*B, H*W, D, 2]
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        # bs, num_query, _ = query.shape
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # 对value进行线性变换，并分配到多个注意力头上，num_value是多尺度特征图的像素数量
        # 形状变成[bs, num_value, num_heads, embed_dims//num_heads]
        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 对三视图的query进行线性变换，获得采样偏移和注意力权重，并将三视图的采样偏移和注意力权重沿第一维度连接成一个张量
        # 原本三个视图对应三组参考点，每一组都要单独进行采样，这里将三组参考点沿num_query维度连接成一个张量
        # num_query表示对每个批次图像，每个检测头，每个层级采样的数量，三个视图的num_query合并到一起后
        # 就可以在MSDA的架构下，一次性完成三视图三组参考点的采样
        # 采样偏移的形状是[bs, num_query, num_heads, num_levels, num_points, 2]，num_points=8
        # 注意力权重的形状是[bs, num_query, num_heads, num_levels, num_points]，num_points=8
        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)

        # 重整三视图的参考点张量形状，然后连接成一个张量，形状是[bs, num_query, num_Z_anchors, xy]
        # bs是num_cams*bs，num_query是三视图的H*W*PM之和，num_Z_anchors统一都是4，xy是2
        reference_points = self.reshape_reference_points(reference_points)
        
        if reference_points.shape[-1] == 2:
            """
            For each tpv query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each tpv query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            # 将采样偏移的num_points维度拆成num_Z_anchors和num_all_points//num_Z_anchors两个维度，分别是4和2
            # 形状变成[bs, num_query, num_heads, num_levels, num_Z_anchors, num_points, 2]，num_Z_anchors=4，num_points=2
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors, num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            # 将采样位置的num_points和num_Z_anchors两个维度合并成一个维度，
            # 形状变成[bs, num_query, num_heads, num_levels, num_all_points, 2]，num_all_points=8
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape:  bs, num_query, num_heads, num_levels, num_all_points
        #  value.shape:              bs, num_value, num_heads, embed_dims//num_heads

        #  num_query是三视图的H*W*PM之和，H*W是BEV特征图的尺寸，PM=[1, 8, 8]
        #  num_value是每一个相机图像对应的多尺度特征图的总像素数量

        #  联系起来，表示把每个批次-图像、每个检测头的num_value个像素划分成num_levels个层级，对每个层级进行num_query个采样点的采样
        #  每个层级的像素数量不同，因此需要由spatial_shapes和level_start_index两个参数指定层级划分方式
        #  以上就是MSDA要做的事

        #  与BEVFormer只有一个视图相比，TPVFormer将三个视图的参考点、采样偏移、注意力权重叠加后主要体现在num_query维度上，其他几个
        #  维度都是一致的，三个视图的参考点等都是沿num_query维度连接成一个张量。

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]

        return output
