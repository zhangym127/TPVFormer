from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TPVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross attention.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, tpv_h, tpv_w, tpv_z, pc_range=None, 
                 num_points_in_pillar=[4, 32, 32], 
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 return_intermediate=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.pc_range = pc_range
        self.fp16_enabled = False

        ref_3d_hw = self.get_reference_points(tpv_h, tpv_w, pc_range[5]-pc_range[2], num_points_in_pillar[0], '3d', device='cpu')

        ref_3d_zh = self.get_reference_points(tpv_z, tpv_h, pc_range[3]-pc_range[0], num_points_in_pillar[1], '3d', device='cpu')
        ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]
        ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)

        ref_3d_wz = self.get_reference_points(tpv_w, tpv_z, pc_range[4]-pc_range[1], num_points_in_pillar[2], '3d', device='cpu')
        ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)
        
        # 注意：构造交叉视图参考点的时候使用的是num_points_in_pillar_cross_view，即三个平面的柱体点数相同，都是32
        cross_view_ref_points = self.get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)
        self.num_points_cross_view = num_points_in_pillar_cross_view


    @staticmethod
    def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2

        # 为HW平面生成参考点，Z方向构造柱体，设置p个点，p=32
        # hw_hw是为hw查询去查询hw平面准备的参考点
        # hw_zh是为hw查询去查询zh平面准备的参考点
        # hw_wz是为hw查询去查询wz平面准备的参考点
        # generate points for hw and level 1
        #生成0.5到tpv_h-0.5的tpv_h个数，再除以tpv_h，得到0到1之间的数
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h        # 生成参考点在h方向的坐标序列且归一化  
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w        # 生成参考到在w方向的坐标序列且归一化
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()   # 将h方向的坐标序列扩展到w方向，得到hw个点的h坐标
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()    # 将w方向的坐标序列扩展到h方向，得到hw个点的w坐标
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2       # 将h方向和w方向的坐标序列拼接起来，得到hw个点的完整坐标
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2 # 在每个hw点上，构造pillar形式的若干点，得到hw*p个点的hw坐标
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p     # 生成参考点在z方向的坐标序列且归一化
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p                   # 将z方向的坐标序列扩展到hw方向，得到hw*p个点的z坐标
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h                            # 生成参考点在h方向的坐标序列且归一化
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)  # 将h方向的坐标序列扩展到w方向和z方向，得到hw*p个点的h坐标
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2                       # 将h方向和z方向的坐标序列拼接起来，得到hw*p个点的hz坐标
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p     # 同上
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2                       # 将w方向和z方向的坐标序列拼接起来，得到hw*p个点的wz坐标
        
        # 为ZH平面生成参考点，W方向构造柱体，设置p个点，p=32
        # zh_hw是为zh查询去查询hw平面准备的参考点
        # zh_zh是为zh查询去查询zh平面准备的参考点
        # zh_wz是为zh查询去查询wz平面准备的参考点
        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w  # 生成参考点在w方向的坐标序列且归一化
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)                    # 将w方向的坐标序列扩展到zh方向，得到zh*p个点的w坐标
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h                    # 生成参考点在h方向的坐标序列且归一化
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)  # 将h方向的坐标序列扩展到z方向和w方向，得到zh*p个点的h坐标
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)                           # 将w方向和h方向的坐标序列拼接起来，得到zh*p个点的hw坐标
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2               # 将h方向和z方向的坐标序列拼接起来，得到zh*p个点的hz坐标
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)                           # 将w方向和z方向的坐标序列拼接起来，得到zh*p个点的wz坐标

        # 为WZ平面生成参考点，H方向构造柱体，设置p个点，p=32
        # wz_hw是为wz查询去查询hw平面准备的参考点
        # wz_zh是为wz查询去查询zh平面准备的参考点
        # wz_wz是为wz查询去查询wz平面准备的参考点
        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)           # 将w方向和h方向的坐标序列拼接起来，得到wz*p个点的hw坐标
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)           # 将h方向和z方向的坐标序列拼接起来，得到wz*p个点的hz坐标
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)           # 将w方向和z方向的坐标序列拼接起来，得到wz*p个点的wz坐标

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),  # 将三个[hw, p, 2]张量叠加在一起，形状变成[hw, 3, p, 2]
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),  # 将三个[zh, p, 2]张量叠加在一起，形状变成[zh, 3, p, 2]
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)   # 将三个[wz, p, 2]张量叠加在一起，形状变成[wz, 3, p, 2]
        ], dim=0) # hw+zh+wz, 3, #p, 2  # 将三个张量沿第0维连接起来，形状变成[hw+zh+wz, 3, p, 2]

        # 这个函数是实现交叉视图混合注意力机制的关键，下面详细解释一下
        # 交叉视图混合注意力机制本质上就是BEVFormer的自注意力机制，不同的是：
        # TPVFormer有三个方向的BEV特征图（即视图），就需要三个查询：tpv_queries_hw, tpv_queries_zh, tpv_queries_wz
        # 每个查询都需要和三个视图进行交互，进行特征提取，这也是“交叉视图混合注意力”名字的由来。
        # 这样就需要3×3=9次查询，即三个查询对三个视图的九份查询，对应的，需要九组不同尺寸的参考点，相当繁琐，是否能够简化？
        #
        # 简化的思路就是利用MSDA的多尺度输入特征图机制，将三个查询对三个视图的九份查询，简化成对三个输入特征图的三层级查询
        # 在MSDA中，会根据对输入特征图的层级划分，将采样点和注意力权重也划分成对应的层级，也就是分成多份，每份对应一个层级。
        # 例如典型的输入特征图有四个层级，采样点和注意力权重也会分成四份，每份对应一个层级，这样就可以实现对多尺度输入特征
        # 图的特征采样。

        # 自注意力机制原本就只有一个输入特征图层级，采样点和注意力权重也只需要一份。但是，TPVFormer把三个视图看成三个层级，
        # 在自注意力机制中value源自查询query，value按照hw、zh、wz三个平面排列，形式上类似于特征图的三个层级；
        # 这样就有了三个输入特征图层级，num_levels=3，那么采样点和注意力权重也需要分成三份，每份对应一个层级。

        # 而参考点则是采样点的基础，于是九组参考点就按照视图分成三层级，就有了下面这个表：水平方向是查询，垂直方向是层级：
        #                   level_1  level_2  level_3
        #  tpv_queries_hw：  hw_hw,   hw_zh,   hw_wz
        #  tpv_queries_zh：  zh_hw,   zh_zh,   zh_wz
        #  tpv_queries_wz：  wz_hw,   wz_zh,   wz_wz
        #
        # level 1 是只对hw平面进行采样的参考点，其中包括hw、zh、wz三个查询对hw平面的三份采样参考点，三份的参考点的数量各不相同
        # level 2 是只对zh平面进行采样的参考点，其中包括hw、zh、wz三个查询对zh平面的三份采样参考点，三份的参考点的数量各不相同
        # level 3 是只对wz平面进行采样的参考点，其中包括hw、zh、wz三个查询对wz平面的三份采样参考点，三份的参考点的数量各不相同

        # 这样一来，参考点也被划分成level 1、2、3三个层级，对应于value的hw、zh、wz三个平面，实现了参考点和value的层级对应；
        # 对level 1的一次特征采样，就完成了hw、zh、wz三个查询对hw平面的三份采样，对level 2、3也是如此，大大简化了特征提取的过程。
        # 形式上与MSDA的多尺度输入特征图机制一致，但是本质上是自注意力机制，只是利用了MSDA的多尺度输入特征图机制来简化特征提取过程。
        # 完成特征采集之后，再将采集到的特征分配到各自的查询上，
        # 这样，就利用MSDA支持的多尺度（多个level）输入特征图机制，完成了交叉视图混合注意力机制；
        
        return reference_points

    # @info 获得空间交叉注意力机制（SCA）和时域自注意力机制（TSA）中要使用的参考点
    #       hw视图的柱子采样点数是4，zh和wz视图的柱子采样点数都是32
    # @param H, W: bev网格的高度和宽度
    # @param Z: 柱子的高度，源自3D点云的高度
    # @param num_points_in_pillar: 每个柱子中采样的点数, hw、zh、wz三视图的点数分别是[4, 32, 32]
    # @param dim: 3d或2d
    # @param bs: batch size
    # @param device: 设备，cpu或cuda
    # @param dtype: 数据类型
    # @return ref_3d: 3D参考点，形状是 (B,N,H*W,3)
    # @return ref_2d: 2D参考点，形状是 (B,H*W,1,2)
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of tpv.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H
            # 将Z、X、Y坐标拼接在一起，得到3D参考点，形状是（P,H,W,3），P是Z轴Pillar采样点个数，H和W是BEV网格的尺寸，3是参考点归一化坐标
            ref_3d = torch.stack((xs, ys, zs), -1)
            # 将参考点的形状变为（P,H*W,3），即将H、W两个维度合并
            # 先把H和W两个维度挪到最后，合并后再挪回到前面
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # 将参考点张量ref_3d在第0维上扩展一个维度，然后重复bs次，形状变为（B,P,H*W,3），B是batch size
            # [None]操作将ref_3d在最外围扩展一个维度，形状变为（1,P,H*W,3），再对第0维重复bs次，形状变为（B,P,H*W,3）
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D tpv plane, used in self attention in tpvformer04 
        # which is an older version. Now we use get_cross_view_ref_points instead.
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # @info: 将3D参考点投影到图像上，获得每个参考点在每个图像上的像素归一化坐标
    # @param reference_points: 3D参考点，形状是(B,N,H*W,3)
    # @param pc_range: 点云范围，形状是（6）
    # @param img_metas: 图像元数据，形状是（B）
    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        # 参考点的形状从（B,D,H*W,3）变成（D, B, H*W, 3），D也就是Pillar的采样点数
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # 为参考点增加一个num_cam维度，形状变成（D,B,1,H*W,3），并根据相机数目复制num_cam份
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        tpv_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        tpv_mask = (tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            tpv_mask = torch.nan_to_num(tpv_mask)
        else:
            tpv_mask = tpv_mask.new_tensor(
                np.nan_to_num(tpv_mask.cpu().numpy()))

        # 将参考点的维度顺序调整为（N,B,H*W,D,2），相机数放到第0维度，然后是bs
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        # 将mask的维度顺序调整为（N,B,H*W,D,1），然后去掉最后一个维度，形状变成（N,B,H*W,D）
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, tpv_mask

    @auto_fp16()
    def forward(self,
                tpv_query, # list
                key,
                value,
                *args,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                tpv_pos=None, # list
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            tpv_query (Tensor): Input tpv query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
        """
        output = tpv_query
        intermediate = []
        bs = tpv_query[0].shape[0]

        # 为三视图中的每个视图生成参考点，合并在一起，投影到图像，获取像素坐标和mask
        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            # 将3D参考点投影到图像上，获得每个参考点在每个图像上的像素归一化坐标，形状是（N,B,H*W,D,2）
            # 部分参考点可能落在图像外，因此需要一个mask来标记有效的参考点，形状是（N,B,H*W,D），为1表示有效，为0表示无效
            reference_points_cam, tpv_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            # 将三视图对应的归一化像素坐标和mask叠加在一起
            reference_points_cams.append(reference_points_cam) # list，每个元素的形状是（N,B,H*W,D,2）
            tpv_masks.append(tpv_mask)
        
        # 交叉视图参考点扩展到Batch维度，形状是[bs, hw+zh+wz, 3, p, 2]
        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(0).expand(bs, -1, -1, -1, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,  # list类型，包含三个视图的query
                key,        # 2D图像特征，格式是(num_cam, H*W++, bs, embed_dims)
                value,      # 2D图像特征，格式是(num_cam, H*W++, bs, embed_dims)
                *args,
                tpv_pos=tpv_pos,    # list类型，包含三个视图的位置编码
                ref_2d=ref_cross_view,  # 2D参考点，格式是[bs, hw+zh+wz, 3, p, 2]
                tpv_h=tpv_h,
                tpv_w=tpv_w,
                tpv_z=tpv_z,
                spatial_shapes=spatial_shapes,          # 特征图的尺寸，形状为[L,2]，分别记录每一层的H和W
                level_start_index=level_start_index,    # 每一层特征图的起始索引
                reference_points_cams=reference_points_cams,    # list类型，为三视图分别生成的三组3D参考点，投影到图像后的像素坐标
                tpv_masks=tpv_masks,                    # list类型，为三视图分别生成的三组mask，标记有效的参考点，形状是（N,B,H*W,D）
                **kwargs)
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output