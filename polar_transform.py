import numpy as np
import torch
from typing import Tuple, Union
from kornia.geometry.transform import remap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device('cuda')

class PolarTransformModule(torch.nn.Module):
    def __init__(self, tensor: torch.Tensor,
                 inner_radius: Union[float, int] = 0, outer_radius: Union[float, int, None] = None,
                 upscale_factor: Tuple[float, float] = (4., 4.)):
        super(PolarTransformModule, self).__init__()

        self.upscale_factor = upscale_factor

        # inner radius, in px or px fraction
        self.inner_radius = inner_radius

        # get max radius up to which there is non-zero data
        self.radius_data, self.radius_data_deprojected = self.compute_data_radius(tensor)
        self.radius_data_px = int(self.radius_data * np.ceil(tensor.shape[-2] / 2.))
        safety_margin = 1  # in px
        self.outer_radius_gap_px = int(
            np.clip(((1. - (self.radius_data_deprojected))
                     * np.ceil(tensor.shape[-2] / 2.)) - safety_margin,
                    0, np.inf
                    )
        )

        # outer radius, in px or px fraction
        if outer_radius is None:
            outer_radius = self.radius_data_px
        self.outer_radius = outer_radius

        # SHAPES
        # shape: (*, x, y)
        self.original_shape = tensor.shape

        # define radii
        self.radius_normal = 1.
        self.radius_diagonal = np.sqrt(2.)

        self.px_normal = int(np.ceil(tensor.shape[-2] / 2.))
        self.radius_normal_px = int(self.radius_normal * self.px_normal)
        self.radius_diagonal_px = int(self.radius_diagonal * self.px_normal)

        # shape: (*, x, y)
        self.cartesian_shape = torch.tensor([*tensor.shape])
        self.cartesian_center = torch.floor(torch.tensor([
            tensor.shape[-2] / 2.,
            tensor.shape[-1] / 2.
        ])).to(torch.int)

        # polar shape: (*, rho, phi)
        cross = torch.tensor([
            [self.cartesian_shape[-1] - 1, self.cartesian_center[1]],
            [0, self.cartesian_center[1]],
            [self.cartesian_center[0], self.cartesian_shape[-2] - 1],
            [self.cartesian_center[0], 0]
        ])

        self.radius_size = torch.ceil(self.upscale_factor[-2] * torch.abs(
            cross - self.cartesian_center
        ).max() * 2. * (self.outer_radius - self.inner_radius) / self.radius_diagonal_px).to(torch.int64).item()

        self.angle_size = int(self.upscale_factor[-1] * torch.max(self.cartesian_shape[-2:]))

        # shape: (*, rho, phi)
        self.polar_shape = torch.tensor([
            *tensor.shape[:-2],
            self.radius_size,
            self.angle_size,
        ]).to(torch.int)

        # PIXEL FLOW MAPS
        (self.map_cartesian_to_polar_x, self.map_cartesian_to_polar_y), self.phase_mask = self.build_cart2pol_maps()
        (self.map_polar_to_cartesian_x, self.map_polar_to_cartesian_y), self.radius_mask = self.build_pol2cart_maps()

        self._unsqueeze_maps()

    def cart2pol(self, cart_tensor: torch.Tensor) -> torch.Tensor:
        polar_tensor = remap(
            cart_tensor.view(1, -1, *cart_tensor.shape[-2:]),
            map_x=self.map_cartesian_to_polar_x, map_y=self.map_cartesian_to_polar_y,
            align_corners=True,
            normalized_coordinates=False,
            padding_mode="border",
            mode="bilinear",
        )
        polar_tensor = polar_tensor.view(*cart_tensor.shape[:-2], *polar_tensor.shape[-2:])
        return (polar_tensor * self.phase_mask).contiguous()

    def pol2cart(self, polar_tensor: torch.Tensor) -> torch.Tensor:
        # kornia expects the incoming tensor to be in (*, phi, rho) to be aligned with the x and y map,
        # but the tensor comes in as (*, rho, phi) with our convention, so we switch the map dimensions
        cart_tensor = remap(
            polar_tensor.view(1, -1, *polar_tensor.shape[-2:]),
            map_x=self.map_polar_to_cartesian_y, map_y=self.map_polar_to_cartesian_x,
            align_corners=True,
            normalized_coordinates=False,
            padding_mode="border",
            mode="bilinear",
        )

        cart_tensor = cart_tensor.view(*polar_tensor.shape[:-2], *cart_tensor.shape[-2:])

        return (cart_tensor * self.radius_mask).contiguous()

    def pol2cart_masked_values(self, polar_tensor: torch.Tensor) -> torch.Tensor:
        cart_tensor = self.pol2cart(polar_tensor=polar_tensor)
        return cart_tensor[..., self.radius_mask]

    def car2pol_masked_values(self, cart_tensor: torch.Tensor) -> torch.Tensor:
        polar_tensor = self.cart2pol(cart_tensor=cart_tensor)
        return polar_tensor[..., self.phase_mask]

    def build_cart2pol_maps(self):
        rho = self.linspace(self.inner_radius, self.outer_radius, self.radius_size)
        phi = self.linspace(0., 2. * torch.pi, self.angle_size)

        rr, pp = torch.meshgrid(rho, phi, indexing="ij")

        map_cartesian_to_polar_x = rr * torch.cos(pp) + self.cartesian_center[0]
        map_cartesian_to_polar_y = rr * torch.sin(pp) + self.cartesian_center[1]

        r_max = torch.sqrt(1. + 2 *
                           torch.minimum(torch.square(torch.sin(phi)),
                                         torch.square(torch.cos(phi)),
                                         )
                           ) * self.radius_normal_px

        phase_mask = torch.lt(rr, r_max)

        return (map_cartesian_to_polar_x, map_cartesian_to_polar_y), phase_mask

    def build_pol2cart_maps(self):
        scale_radius = self.polar_shape[-2] / (self.outer_radius - self.inner_radius)
        scale_angle = self.polar_shape[-1] / (2. * torch.pi)

        x = self.linspace(- self.cartesian_shape[-2] / 2., self.cartesian_shape[-2] / 2., self.cartesian_shape[-2])
        y = self.linspace(- self.cartesian_shape[-1] / 2., self.cartesian_shape[-1] / 2., self.cartesian_shape[-1])

        yy, xx = torch.meshgrid(x, y, indexing="ij")

        # get x and y pixel flow from polar to cartesian
        map_polar_to_cartesian_x = torch.sqrt(torch.square(xx) + torch.square(yy))
        radius_mask = torch.logical_and(
            torch.gt(map_polar_to_cartesian_x, self.inner_radius),
            torch.lt(map_polar_to_cartesian_x, self.outer_radius)
        )
        map_polar_to_cartesian_x = map_polar_to_cartesian_x - self.inner_radius
        map_polar_to_cartesian_x = map_polar_to_cartesian_x * scale_radius

        map_polar_to_cartesian_y = torch.atan2(yy, xx)
        map_polar_to_cartesian_y = (map_polar_to_cartesian_y + 2. * torch.pi) % (2. * torch.pi)
        map_polar_to_cartesian_y = map_polar_to_cartesian_y * scale_angle

        return (map_polar_to_cartesian_x, map_polar_to_cartesian_y), radius_mask

    def _unsqueeze_maps(self):
        # unsqueeze for kornia
        self.map_cartesian_to_polar_x = self.map_cartesian_to_polar_x.unsqueeze(0).contiguous()
        self.map_cartesian_to_polar_y = self.map_cartesian_to_polar_y.unsqueeze(0).contiguous()

        self.map_polar_to_cartesian_x = self.map_polar_to_cartesian_x.unsqueeze(0).contiguous()
        self.map_polar_to_cartesian_y = self.map_polar_to_cartesian_y.unsqueeze(0).contiguous()

    @staticmethod
    def linspace(start, end, steps):
        return torch.linspace(start, end, steps + 1)[:-1].contiguous()

    @staticmethod
    def compute_data_radius(tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Compute the radius up which to there is meaningful data in the image cube.
        The radius is computed as the maximum radius up to which there is non-zero data in the cube.
        Then the position of the element associated with the maximum radius is used to compute the deprojected radius.
        The deprojected radius can be used to trim the cube.
        """
        try:
            # data radius is in units of normal radii
            x = torch.linspace(-1., 1., tensor.shape[-2])
            y = torch.linspace(-1., 1., tensor.shape[-1])

            yy, xx = torch.meshgrid(x, y, indexing="ij")
            radius = torch.sqrt(torch.square(xx) + torch.square(yy))
            mask = torch.eq(torch.sum(torch.abs(tensor), dim=(0, 1)), 0.)
            masked_radius = ~mask * radius  # invert mask: elements=0: True -> elements!=0: True

            radius_data = masked_radius.max().item()

            max_radius_idx_flat = torch.argmax(masked_radius)  # freaking torch and flat argmax?!

            max_radius_idx = [(max_radius_idx_flat % tensor.shape[-2]),
                              max_radius_idx_flat // tensor.shape[-2]]
            theta_max = torch.atan2(max_radius_idx[1] - tensor.shape[-2] / 2,
                                    max_radius_idx[0] - tensor.shape[-1] / 2)

            data_radius_deprojected = torch.maximum(
                torch.abs(torch.cos(theta_max) * radius_data),
                torch.abs(torch.sin(theta_max) * radius_data),
            ).item()
        except RuntimeError:
            radius_data = np.sqrt(2.)
            data_radius_deprojected = 1.

        return radius_data, data_radius_deprojected

    def show_maps(self, rr, pp, xx, yy):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(24, 8), constrained_layout=True)

        ax = axes[0, 0]
        im = ax.imshow(rr.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("rr")

        ax = axes[1, 0]
        im = ax.imshow(pp.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("pp")

        ax = axes[0, 1]
        im = ax.imshow(self.map_cartesian_to_polar_x.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_cartesian_to_polar_x")

        ax = axes[1, 1]
        im = ax.imshow(self.map_cartesian_to_polar_y.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_cartesian_to_polar_y")

        ax = axes[0, 2]
        im = ax.imshow(xx.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("xx")

        ax = axes[1, 2]
        im = ax.imshow(yy.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("yy")

        ax = axes[0, 3]
        im = ax.imshow(self.map_polar_to_cartesian_x.squeeze().cpu().numpy(), origin="lower", cmap="seismic",
                       # vmin=-1, vmax=1.
                       )
        plt.colorbar(im, ax=ax)
        ax.set_title("map_polar_to_cartesian_x")

        ax = axes[1, 3]
        im = ax.imshow(self.map_polar_to_cartesian_y.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("map_polar_to_cartesian_y")

        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with torch.no_grad():
        torch.manual_seed(42)

        X, Y = 290, 290
        noise = 1e-2
        annulus_radius_inner = 50
        annulus_radius_outer = 75

        x = torch.linspace(-1., 1., X + 1)[:-1]
        y = torch.linspace(-1., 1., Y + 1)[:-1]

        xx, yy = torch.meshgrid(x, y, indexing="ij")

        radius = torch.sqrt(torch.square(xx) + torch.square(yy))
        phase = torch.atan2(yy, xx)

        radius_cube = torch.zeros(1, 1, *radius.shape)
        radius_cube[:, :] = radius

        phase_cube = torch.zeros_like(radius_cube)
        phase_cube[:, :] = phase

        # =============================================================================
        # NOISELESS CASE
        # expected input is always 4D with intended shape (channels, samples, x, y)
        # any shape like (any, any, x, y) is accepted, only the trailing two dimensions are used
        # =============================================================================

        # RADIUS
        polar_transform = PolarTransformModule(radius_cube)
        polar_radius_cube = polar_transform.cart2pol(cart_tensor=radius_cube)
        cart_radius_cube = polar_transform.pol2cart(polar_tensor=polar_radius_cube)
        err_radius_cube = cart_radius_cube - radius_cube

        # PHASE
        # we could just reuse the previous instance since the cubes are the same shape
        polar_transform = PolarTransformModule(phase_cube)
        polar_phase_cube = polar_transform.cart2pol(cart_tensor=phase_cube)
        cart_phase_cube = polar_transform.pol2cart(polar_tensor=polar_phase_cube)
        err_phase_cube = cart_phase_cube - phase_cube


        # ANNULUS (RADIUS)
        polar_transform = PolarTransformModule(
            radius_cube, inner_radius=annulus_radius_inner, outer_radius=annulus_radius_outer)  # demonstrate annulus
        annulus_polar_radius_cube = polar_transform.cart2pol(cart_tensor=radius_cube)
        annulus_cart_radius_cube = polar_transform.pol2cart(polar_tensor=annulus_polar_radius_cube)
        annulus_cart_radius_cube_vals = polar_transform.pol2cart_masked_values(polar_tensor=annulus_polar_radius_cube)

        # PLOTS
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

        ax = axes[0, 0]
        im = ax.imshow(radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("radius_cube")

        ax = axes[1, 0]
        im = ax.imshow(phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("phase_cube")

        ax = axes[2, 0]
        ax.axis("off")

        ax = axes[0, 1]
        im = ax.imshow(polar_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("polar_radius_cube")

        ax = axes[1, 1]
        im = ax.imshow(polar_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("polar_phase_cube")

        ax = axes[2, 1]
        im = ax.imshow(annulus_polar_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic", aspect=8.)
        plt.colorbar(im, ax=ax)
        ax.set_title("annulus_polar_radius_cube")

        ax = axes[0, 2]
        im = ax.imshow(cart_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("cart_radius_cube")

        ax = axes[1, 2]
        im = ax.imshow(cart_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("cart_phase_cube")

        ax = axes[2, 2]
        im = ax.imshow(annulus_cart_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic",
                       vmin=annulus_cart_radius_cube_vals.min(), vmax=annulus_cart_radius_cube_vals.max())
        plt.colorbar(im, ax=ax)
        ax.set_title("annulus_cart_radius_cube")

        ax = axes[0, 3]
        im = ax.imshow(err_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("err_radius_cube")

        ax = axes[1, 3]
        im = ax.imshow(err_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("err_phase_cube")

        ax = axes[2, 3]
        color = radius_cube[:, :, polar_transform.radius_mask].flatten().cpu().numpy()
        color = (color - color.min()) / (color.max() - color.min())
        ax.scatter(
            phase_cube[:, :, polar_transform.radius_mask].flatten().cpu().numpy(),
            annulus_cart_radius_cube_vals.flatten().cpu().numpy(),
            s=4.0, alpha=0.75, marker="x", c=color, cmap="gist_rainbow"
        )
        ax.set_xlabel("phase")
        ax.set_ylabel("reconstructed radius")
        ax.set_title("annulus_cart_radius_cube_vals")

        fig.suptitle("Polar Transform (NOISELESS CASE)")
        # fig.savefig("polar_transform_no_noise.png", dpi=300)
        plt.show()

        # =============================================================================
        # NOISY CASE
        # =============================================================================

        noise = torch.randn_like(radius_cube) * noise
        radius_cube += noise
        phase_cube += noise

        # RADIUS
        polar_transform = PolarTransformModule(radius_cube)
        polar_radius_cube = polar_transform.cart2pol(cart_tensor=radius_cube)
        cart_radius_cube = polar_transform.pol2cart(polar_tensor=polar_radius_cube)
        err_radius_cube = cart_radius_cube - radius_cube

        # PHASE
        # we could just reuse the previous instance since the cubes are the same shape
        polar_transform = PolarTransformModule(phase_cube)
        polar_phase_cube = polar_transform.cart2pol(cart_tensor=phase_cube)
        cart_phase_cube = polar_transform.pol2cart(polar_tensor=polar_phase_cube)
        err_phase_cube = cart_phase_cube - phase_cube

        # ANNULUS (RADIUS)
        polar_transform = PolarTransformModule(
            radius_cube, inner_radius=annulus_radius_inner, outer_radius=annulus_radius_outer)  # demonstrate annulus
        annulus_polar_radius_cube = polar_transform.cart2pol(cart_tensor=radius_cube)
        annulus_cart_radius_cube = polar_transform.pol2cart(polar_tensor=annulus_polar_radius_cube)
        annulus_cart_radius_cube_vals = polar_transform.pol2cart_masked_values(polar_tensor=annulus_polar_radius_cube)

        # PLOTS
        fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

        ax = axes[0, 0]
        im = ax.imshow(radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("radius_cube")

        ax = axes[1, 0]
        im = ax.imshow(phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("phase_cube")

        ax = axes[2, 0]
        im = ax.imshow(noise.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("noise")

        ax = axes[0, 1]
        im = ax.imshow(polar_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("polar_radius_cube")

        ax = axes[1, 1]
        im = ax.imshow(polar_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("polar_phase_cube")

        ax = axes[2, 1]
        im = ax.imshow(annulus_polar_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic", aspect=8.)
        plt.colorbar(im, ax=ax)
        ax.set_title("annulus_polar_radius_cube")

        ax = axes[0, 2]
        im = ax.imshow(cart_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("cart_radius_cube")

        ax = axes[1, 2]
        im = ax.imshow(cart_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="hsv")
        plt.colorbar(im, ax=ax)
        ax.set_title("cart_phase_cube")

        ax = axes[2, 2]
        im = ax.imshow(annulus_cart_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic",
                       vmin=annulus_cart_radius_cube_vals.min(), vmax=annulus_cart_radius_cube_vals.max())
        plt.colorbar(im, ax=ax)
        ax.set_title("annulus_cart_radius_cube")

        ax = axes[0, 3]
        im = ax.imshow(err_radius_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("err_radius_cube")

        ax = axes[1, 3]
        im = ax.imshow(err_phase_cube.squeeze().cpu().numpy(), origin="lower", cmap="seismic")
        plt.colorbar(im, ax=ax)
        ax.set_title("err_phase_cube")

        ax = axes[2, 3]
        color = radius_cube[:, :, polar_transform.radius_mask].flatten().cpu().numpy()
        color = (color - color.min()) / (color.max() - color.min())
        ax.scatter(
            phase_cube[:, :, polar_transform.radius_mask].flatten().cpu().numpy(),
            annulus_cart_radius_cube_vals.flatten().cpu().numpy(),
            s=4.0, alpha=0.75, marker="x", c=color, cmap="gist_rainbow"
        )
        ax.set_xlabel("phase")
        ax.set_ylabel("reconstructed radius")
        ax.set_title("annulus_cart_radius_cube_vals")

        fig.suptitle("Polar Transform (NOISY CASE)")
        # fig.savefig("polar_transform_with_noise.png", dpi=300)
        plt.show()



