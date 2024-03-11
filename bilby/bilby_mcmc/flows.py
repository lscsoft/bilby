import torch
from glasflow.nflows.distributions.normal import StandardNormal
from glasflow.nflows.flows.base import Flow
from glasflow.nflows.nn import nets as nets
from glasflow.nflows.transforms import (
    CompositeTransform,
    MaskedAffineAutoregressiveTransform,
    RandomPermutation,
)
from glasflow.nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
)
from glasflow.nflows.transforms.normalization import BatchNorm
from torch.nn import functional as F

# Turn off parallelism
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class NVPFlow(Flow):
    """A simplified version of Real NVP for 1-dim inputs.

    This implementation uses 1-dim checkerboard masking but doesn't use
    multi-scaling.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.

    This class has been modified from the example found at:
    https://github.com/bayesiains/nflows/blob/master/nflows/flows/realnvp.py
    """

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        use_volume_preserving=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=False,
        random_permutation=True,
    ):

        if use_volume_preserving:
            coupling_constructor = AdditiveCouplingTransform
        else:
            coupling_constructor = AffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            layers.append(transform)
            mask *= -1
            if batch_norm_between_layers:
                layers.append(BatchNorm(features=features))

        if random_permutation:
            layers.append(RandomPermutation(features=features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )


class BasicFlow(Flow):
    def __init__(self, features):
        transform = CompositeTransform(
            [
                MaskedAffineAutoregressiveTransform(
                    features=features, hidden_features=2 * features
                ),
                RandomPermutation(features=features),
            ]
        )
        distribution = StandardNormal(shape=[features])
        super().__init__(
            transform=transform,
            distribution=distribution,
        )
