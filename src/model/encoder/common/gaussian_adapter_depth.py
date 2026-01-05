from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance
from .me_fea import project_features_to_me
from typing import Tuple, Optional 
from ....geometry.projection import sample_voxel_grid


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter_depth(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1): 
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Tensor,
        intrinsics: Tensor | None,
        opacities: Tensor,
        raw_gaussians: Tensor, #[1, 1, N, 37]
        input_images: Tensor | None = None,
        depth : Tensor | None = None,
        coordinate: Optional[Tensor] = None,
        points: Optional[Tensor] = None,
        voxel_resolution: float = 0.01,
        eps: float = 1e-8,
    ) :
    #-> Gaussians

        batch_dims = extrinsics.shape[:-2]
     
        b, v = batch_dims
        
        offset_xyz,scales, rotations, sh = raw_gaussians.split((3,3, 4, 3 * self.d_sh), dim=-1) #[1, 1, N,1, 1,c]

        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)                          
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)    # [1, 1, 256000, 1, 1, 3, 9]
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None :
            voxel_color, aggregated_points, counts = project_features_to_me(
                intrinsics = intrinsics,
                extrinsics = extrinsics,
                out = input_images,
                depth =  depth,
                voxel_resolution = voxel_resolution,
                b=b,v=v
            )
            # if torch.equal(coordinate, voxel_color.C):
            if coordinate.shape == voxel_color.C.shape:
                colors = voxel_color.F  # [N_total, C]

                if sh.shape[0] > 1 and sh.shape[2] != colors.shape[0]:
                    batch_size = sh.shape[0]
                    max_voxels = sh.shape[2]
                    device = colors.device

                    batched_colors = torch.zeros(batch_size, max_voxels, 3, device=device)

                    for batch_idx in range(batch_size):
                        mask = voxel_color.C[:, 0] == batch_idx
                        batch_colors = colors[mask]  # [N_i, 3]
                        n_voxels = batch_colors.shape[0]
                        batched_colors[batch_idx, :n_voxels, :] = batch_colors

                    sh0 = RGB2SH(batched_colors)  # [b, N_max, 3]
                    sh0_expanded = sh0.view(batch_size, 1, max_voxels, 1, 1, 3)  # [b, 1, N_max, 1, 1, 3]
                else:
                    sh0 = RGB2SH(colors)  
                    sh0_expanded = sh0.view(1, 1, -1, 1, 1, 3)  # [1,1,N,1,1,3]

                sh[..., 0] = sh0_expanded  
        
        
        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)  #[1, 1, 256000, 1, 1, 3, 3]
        
        xyz = points
        if xyz.ndim == 2:
            xyz = rearrange(xyz, "n c -> 1 1 n () () c")
        elif xyz.ndim == 4:
            xyz = rearrange(xyz, "b v n c -> b v n () () c")

        offset_xyz = offset_xyz.sigmoid() 
        offset_world = (offset_xyz - 0.5) *voxel_resolution*3  # [1,1,N,1,1, 3]

        means = xyz + offset_world  # [1,1,N, 1,1,3]
        
        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
