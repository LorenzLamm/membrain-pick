"""
The diffusionnet architecture and code is based on the following Github repository:
https://github.com/nmwsharp/diffusion-net/
"""

import scipy.sparse.linalg as sla

# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import to_basis, from_basis


class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i t) f_in

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(
        self,
        C_inout,
        method="spectral",
        fixed_time=None,
        shared_time=False,
        leaky_relu=False,
        clamp_diffusion=False,
    ):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method  # one of ['spectral', 'implicit_dense']
        self.fixed_time = fixed_time
        self.shared_time = shared_time
        self.leaky_relu = leaky_relu
        self.clamp_diffusion = clamp_diffusion

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-12)
            if self.clamp_diffusion:
                self.diffusion_time.data = torch.clamp(self.diffusion_time, max=1e-3)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        if self.method == "spectral":
            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            if self.fixed_time is not None:
                time = torch.ones_like(self.diffusion_time) * self.fixed_time
            elif self.shared_time:
                time = self.diffusion_time.mean()
            else:
                time = self.diffusion_time

            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex
            x_diffuse = from_basis(x_diffuse_spec, evecs)
            if self.leaky_relu:
                x_diffuse = F.leaky_relu(
                    x_diffuse, negative_slope=0.01
                )  # TODO: This is questionable!!!

        elif self.method == "implicit_dense":
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=0.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(name + "_mlp_act_{:03d}".format(i), activation())


# class SeparableDiffusionNetBlock(nn.Module):
#     """
#     Inputs and outputs are defined at vertices
#     """

#     def __init__(self, C_in, C_width, conv_hidden_dims,
#                  dropout=True,
#                  diffusion_method='spectral',
#                  with_gradient_features=True,
#                  with_gradient_rotations=True,
#                  fixed_time=None,
#                  ):
#         super(SeparableDiffusionNetBlock, self).__init__()

#         # Specified dimensions
#         self.C_width = C_width
#         self.C_in = C_in
#         self.conv_hidden_dims = conv_hidden_dims

#         self.dropout = dropout
#         self.with_gradient_features = with_gradient_features
#         self.with_gradient_rotations = with_gradient_rotations

#         # Diffusion block
#         self.diffusion = LearnedTimeDiffusion(self.C_in, method=diffusion_method, fixed_time=fixed_time, shared_time=True)

#         self.Conv_C_in = (3 if self.with_gradient_features else 2)
#         self.Conv_C_width = self.C_width

#         if self.with_gradient_features:
#             self.gradient_features = SpatialGradientFeatures(self.C_in, with_gradient_rotations=self.with_gradient_rotations)

#         # Convolutional layers
#         self.conv_layer1 = nn.Conv1d(self.Conv_C_in, self.Conv_C_width, kernel_size=3, padding=1)
#         self.conv_layer2 = nn.Conv1d(self.Conv_C_width, 1, kernel_size=3, padding=1)


#     def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):
#         # Manage dimensions
#         B = x_in.shape[0]

#         # Diffusion block
#         x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)
#         # Compute gradient features, if using
#         if self.with_gradient_features:

#                 # Compute gradients
#                 x_grads = [] # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
#                 for b in range(B):
#                     # gradient after diffusion
#                     x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
#                     x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

#                     x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
#                 x_grad = torch.stack(x_grads, dim=0)

#                 # Evaluate gradient features
#                 x_grad_features = self.gradient_features(x_grad)

#                 # Stack inputs to mlp
#                 feature_combined = torch.stack((x_in, x_diffuse, x_grad_features), dim=-1)

#         else:
#             # Stack inputs to mlp

#             feature_combined = torch.stack((x_in, x_diffuse), dim=-1)

#         feature_combined = feature_combined.reshape(-1, self.C_in, self.Conv_C_in)
#         feature_combined = feature_combined.permute(0, 2, 1)


#         # Apply the conv layers
#         x0_out = self.conv_layer1(feature_combined)
#         x0_out = F.relu(x0_out)

#         x0_out = F.max_pool1d(x0_out, kernel_size=3, stride=1, padding=1)
#         x0_out = self.conv_layer2(x0_out)
#         x0_out = F.relu(x0_out)
#         x0_out = F.max_pool1d(x0_out, kernel_size=3, stride=1, padding=1)
#         x_out = x0_out.reshape(1, x_in.shape[1], -1)
#         return x_out


class SeparableDiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        C_in,
        C_width,
        C_in_channels,
        conv_hidden_dims,
        dropout=True,
        diffusion_method="spectral",
        with_gradient_features=True,
        with_gradient_rotations=True,
        fixed_time=None,
    ):
        super(SeparableDiffusionNetBlock, self).__init__()

        self.diffusion_layer = DiffusionNetBlock(
            C_width=C_in_channels,
            mlp_hidden_dims=conv_hidden_dims,
            dropout=dropout,
            diffusion_method=diffusion_method,
            with_gradient_features=with_gradient_features,
            with_gradient_rotations=with_gradient_rotations,
            fixed_time=fixed_time,
            visualize_diffusion=False,
            visualize_grad_rotations=False,
            visualize_grad_features=False,
            separate_channels=True,
        )
        # Specified dimensions
        self.C_width = C_width
        self.C_in = C_in
        self.conv_hidden_dims = conv_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        self.Conv_C_in = 1
        self.Conv_C_width = self.C_width

        # Convolutional layers
        self.conv_layer1 = nn.Conv1d(
            self.Conv_C_in, self.Conv_C_width, kernel_size=3, padding=1
        )
        self.conv_layer2 = nn.Conv1d(self.Conv_C_width, 1, kernel_size=3, padding=1)

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):
        x = self.diffusion_layer(
            x_in, mass, L, evals, evecs, gradX, gradY
        )  # shape (1, 2000, 10)

        x = x.permute(1, 0, 2)

        x = self.conv_layer1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = self.conv_layer2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)

        x_out = x.permute(1, 0, 2)

        # Skip connection
        x_out = x_out + x_in
        return x_out


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        C_width,
        mlp_hidden_dims,
        dropout=True,
        diffusion_method="spectral",
        with_gradient_features=True,
        with_gradient_rotations=True,
        fixed_time=None,
        clamp_diffusion=False,
        visualize_diffusion=False,
        visualize_grad_rotations=False,
        visualize_grad_features=False,
        separate_channels=False,
    ):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        self.visualize_diffusion = visualize_diffusion
        self.visualize_grad_rotations = visualize_grad_rotations
        self.visualize_grad_features = visualize_grad_features

        self.separate_channels = separate_channels

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(
            self.C_width,
            method=diffusion_method,
            fixed_time=fixed_time,
            clamp_diffusion=clamp_diffusion,
        )

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        if self.separate_channels:
            self.MLP_C = 3 if self.with_gradient_features else 2
            self.MLP_OUT = 1
        else:
            self.MLP_OUT = self.C_width

        self.mlp = MiniMLP(
            [self.MLP_C] + self.mlp_hidden_dims + [self.MLP_OUT], dropout=self.dropout
        )

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):
        # Manage dimensions
        B = x_in.shape[0]  # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width
                )
            )

        # Diffusion block
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = (
                []
            )  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            if self.visualize_diffusion:
                # print("visualizing diffusion")
                return x_diffuse
                x_grad_features = torch.zeros_like(x_grad_features)
                x_in = torch.zeros_like(x_in)

                # feature_combined = torch.cat((x_diffuse, x_diffuse, x_diffuse), dim=-1)
            elif self.visualize_grad_features:
                # print("visualizing diffusion")
                return x_grad_features
                x_diffuse = torch.zeros_like(x_diffuse)
                x_in = torch.zeros_like(x_in)
                x_grad_features[:, :, 1:] = 0
                # feature_combined = torch.cat((x_grad_features, x_grad_features, x_grad_features), dim=-1)
            elif self.visualize_grad_rotations:
                # print("visualizing diffusion")
                return x_grad_features
                x_diffuse = torch.zeros_like(x_diffuse)
                x_in = torch.zeros_like(x_in)
                # feature_combined = torch.cat((x_grad_features, x_grad_features, x_grad_features), dim=-1)
            # else:
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
            # feature_combined = torch.cat((x_grad_features, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            # feature_combined = torch.cat((x_diffuse, x_diffuse), dim=-1)
            if self.visualize_diffusion:
                print("visualizing diffusion")
                return x_diffuse
                feature_combined = torch.cat((x_diffuse, x_diffuse), dim=-1)
            else:
                feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        if self.separate_channels:
            feature_combined = feature_combined.reshape(
                1, -1, (3 if self.with_gradient_features else 2)
            )

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        if self.separate_channels:
            x0_out = x0_out.reshape(1, -1, self.C_width)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class AnchorAdjustmentLayer(nn.Module):
    def __init__(self, input_dim, num_anchors, num_classes=0):
        super().__init__()
        self.adjustments = nn.Linear(
            input_dim, num_anchors * 6
        )  # For dx, dy, dz, dwidth, dheight, ddepth
        # self.orientation = nn.Linear(input_dim, num_anchors * orientation_dim)  # If orientation is needed
        self.class_probs = (
            nn.Linear(input_dim, num_anchors * num_classes) if num_classes > 0 else None
        )

    def forward(self, x):
        # Assuming x is [batch_size, num_points, feature_dim]
        adjustments = self.adjustments(
            x
        )  # Shape: [batch_size, num_points, num_anchors * 6]
        orientation = self.orientation(x)  # Optional, based on your needs
        class_probs = (
            F.softmax(self.class_probs(x), dim=-1)
            if self.class_probs is not None
            else None
        )
        return adjustments, orientation, class_probs


class DiffusionNet(nn.Module):

    def __init__(
        self,
        C_in,
        C_out,
        C_width=128,
        N_block=4,
        conv_width=32,
        last_activation=None,
        outputs_at="vertices",
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        device="cuda:0",
        fixed_time=None,
        one_D_conv_first=False,
        clamp_diffusion=False,
        visualize_diffusion=False,
        visualize_grad_rotations=False,
        visualize_grad_features=False,
    ):
        """
        Construct a DiffusionNet.

        Parameters:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient. Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.conv_width = conv_width
        self.fixed_time = fixed_time
        self.one_D_conv_first = one_D_conv_first
        self.visualize_diffusion = visualize_diffusion
        self.visualize_grad_rotations = visualize_grad_rotations
        self.visualize_grad_features = visualize_grad_features

        self.device = device

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ["vertices", "edges", "faces", "global_mean"]:
            raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ["spectral", "implicit_dense"]:
            raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        ## Set up the network
        if one_D_conv_first:
            self.conv_block_out_dim = C_width * (C_in // 4)
            self.conv_block = SeparableDiffusionNetBlock(
                C_in=self.C_in,
                C_width=self.conv_width,
                C_in_channels=C_in,
                conv_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                fixed_time=fixed_time,
            )
            self.conv_block2 = SeparableDiffusionNetBlock(
                C_in=self.C_in,
                C_width=self.conv_width,
                C_in_channels=C_in,
                conv_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                fixed_time=fixed_time,
            )

            self.first_lin = nn.Linear(self.C_in, C_width)
        else:
            # First and last affine layers
            self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                fixed_time=fixed_time,
                clamp_diffusion=clamp_diffusion,
                visualize_diffusion=visualize_diffusion,
                visualize_grad_rotations=visualize_grad_rotations,
                visualize_grad_features=visualize_grad_features,
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def get_config(self):
        return {
            "C_in": self.C_in,
            "C_out": self.C_out,
            "C_width": self.C_width,
            "N_block": self.N_block,
            "last_activation": self.last_activation,
            "outputs_at": self.outputs_at,
            "mlp_hidden_dims": self.mlp_hidden_dims,
            "dropout": self.dropout,
            "diffusion_method": self.diffusion_method,
            "with_gradient_features": self.with_gradient_features,
            "with_gradient_rotations": self.with_gradient_rotations,
            "device": self.device,
            "fixed_time": self.fixed_time,
            "one_D_conv_first": self.one_D_conv_first,
        }

    def forward(
        self,
        x_in,
        mass,
        L=None,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
        edges=None,
        faces=None,
    ):
        """
        A forward pass on the DiffusionNet.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].

        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet, not all are strictly necessary.

        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(
                    self.C_in, x_in.shape[-1]
                )
            )
        N = x_in.shape[-2]
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L != None:
                L = L.unsqueeze(0)
            if evals != None:
                evals = evals.unsqueeze(0)
            if evecs != None:
                evecs = evecs.unsqueeze(0)
            if gradX != None:
                gradX = gradX.unsqueeze(0)
            if gradY != None:
                gradY = gradY.unsqueeze(0)
            if edges != None:
                edges = edges.unsqueeze(0)
            if faces != None:
                faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        # Apply the first linear layer
        if self.one_D_conv_first:
            x = self.conv_block(x_in, mass, L, evals, evecs, gradX, gradY)
            # x = self.conv_block2(x, mass, L, evals, evecs, gradX, gradY)
            x = self.first_lin(x)
        else:
            x = self.first_lin(x_in)

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        if (
            self.visualize_diffusion
            or self.visualize_grad_rotations
            or self.visualize_grad_features
        ):
            return x[:, :, 7]

        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == "vertices":
            x_out = x

        elif self.outputs_at == "edges":
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)

        elif self.outputs_at == "faces":
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        elif self.outputs_at == "global_mean":
            # Produce a single global mean ouput.
            # Using a weighted mean according to the point mass/area is discretization-invariant.
            # (A naive mean is not discretization-invariant; it could be affected by sampling a region more densely)
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(
                mass, dim=-1, keepdim=True
            )

        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out
