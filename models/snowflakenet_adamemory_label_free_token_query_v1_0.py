"""Native AdaMemory SnowflakeNet with label-free memory clustering.

PCN is referenced only for the nearest-prototype pseudo-label objective. The
class-token encoder, memory addressing, semantic fusion, and Snowflake decoder
are inherited from the repository's AdaMemory SnowflakeNet main model.
"""

from .AdaMemorySnowFlakeNet_TokenBasedClassTokenizer import (
    AdaMemorySnowFlakeNet_TokenBasedClassTokenizer,
)
from .build import MODELS
from .label_free_adamemory_v1_0 import LabelFreeAdaMemoryMixin


@MODELS.register_module()
class SnowFlakeNetAdaMemoryLabelFreeTokenQueryV1(
    LabelFreeAdaMemoryMixin,
    AdaMemorySnowFlakeNet_TokenBasedClassTokenizer,
):
    """AdaMemory SnowflakeNet main architecture without category supervision."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.memory_key_dim = int(self.encoder_embed_dim)
        self.memory_value_dim = int(self.decoder_embed_dim)

        if self.memory_key_dim != int(config.dim_feat):
            raise ValueError(
                "Native token-based SnowflakeNet requires "
                "encoder_config.embed_dim == dim_feat"
            )
        if self.memory_value_dim != int(config.dim_feat):
            raise ValueError(
                "Native AdaMemory SnowflakeNet requires "
                "decoder_config.embed_dim == dim_feat"
            )

    @property
    def query_readout(self):
        """Expose the native two-stage token encoder to inspection tools."""
        return self.feat_extractor_cls_token

    def forward(self, input_dict, return_P0=False):
        # The native forward reads adj for historical reasons, but never uses
        # it in token extraction, memory retrieval, fusion, or reconstruction.
        input_dict.setdefault("adj", None)
        input_dict = super().forward(input_dict, return_P0=return_P0)

        if self.training:
            self._record_memory_training_features(
                input_dict,
                input_dict["encoder_cls_token"],
                input_dict["gt_points_token"],
            )
        return input_dict
