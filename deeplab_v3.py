"""Adapted from HistoL framework"""

"""Custom implementation of DeepLabV3 model using torchvision implementation.

For more information on the DeepLabV3 architecture,
[refer to the paper](https://arxiv.org/abs/1706.05587).
"""
from typing import Optional, List, Dict, Any, Tuple

import torch
from torchvision.models import WeightsEnum, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3
from torchvision.transforms import functional as F

_DEEPLABV3_MODEL_WEIGHTS: dict[str, WeightsEnum] = {
    'resnet50': deeplabv3.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    'resnet101': deeplabv3.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    'mobilenet_v3_large': deeplabv3.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
}


class DeepLabV3():
    name = 'DeepLabV3'
    SUPPORTED_BACKBONES = ['resnet50', 'resnet101', 'mobilenet_v3_large']

    def __init__(
        self,
        num_classes: int = 21,
        use_data_parallel: bool = False,
        device_ids: Optional[List[int]] = None,
        *,
        backbone: str = 'resnet50',
        pretrained: bool = False,
        disable_auxiliary_classifier: bool = True,
        train_output_crop_margin: Optional[int] = None,
        eval_output_crop_margin: Optional[int] = None,
        pretrain_model_file: Optional[str] = None,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        """Creates a DeepLabV3 model using torchvision

        Args:
            num_classes: The desired number of output classes for prediction. If not the
                default (21), the DeepLabHead will be replaced
            backbone: The backbone feature extractor to use. Supported backbones:
                'resnet50', 'resnet101', 'mobilenet_v3_large'
            pretrained: Whether the pretrained weights should be loaded (pretrained weights
                loaded before replacing DeepLabHead)
            disable_auxiliary_classifier: Whether the auxiliary classifier should be
                disabled. When setting pretrained=True, the auxiliary classifier is enabled by
                default, so this will disable it
            train_output_crop_margin: Crop margin used to center crop the output during
                training to match the required target shape.
            eval_output_crop_margin:  Crop margin used to center crop the output during
                evaluation to match the required target shape.
            pretrain_model_file: File with absolute path used to load the pre-specified
                weights. If no path specified the torchvision pretrained weights will be used.
                When path is specified pretrained flag should be set to true.
            pretrained_backbone: If this is True and pretrained is False, only the backbone will use
                pretrained weights.
            freeze_backbone: Whether the backbone weights should be frozen.
        """
        # Validate the backbone is supported
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(f'Error. Backbone {backbone} not supported. Supported backbones: '
                             f'{self.SUPPORTED_BACKBONES}')

        load_custom_weights = pretrain_model_file is not None
        if pretrained:
            aux_loss = True
            pretrained_backbone = False
        else:
            aux_loss = False
            if load_custom_weights:
                raise ValueError('Pretrained flag is False but pretrain_model_file is specified')

        if pretrained_backbone:
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights_backbone = None
        if pretrained and not load_custom_weights:
            weights = _DEEPLABV3_MODEL_WEIGHTS[backbone]
        else:
            weights = None

        # Create the model
        model_fn = getattr(deeplabv3, f'deeplabv3_{backbone}')
        model = model_fn(weights=weights, progress=False, aux_loss=aux_loss,
                         weights_backbone=weights_backbone)

        # super().__init__(model, use_data_parallel, device_ids)
        self.model = model

        # Instead of using torchvision's pretrained weights, load custom pretrained weights from
        # the specified file.
        if load_custom_weights:
            self.load_state_dict(load_partial_state_dict(pretrain_model_file, ['classifier']),
                                 strict=False)

        # Replace the head (if num_classes is different from default)
        if num_classes != 21:
            in_channels = self.model.classifier[0].convs[0][0].in_channels
            self.model.classifier = deeplabv3.DeepLabHead(in_channels, num_classes)

        # Disable the aux classifier
        if disable_auxiliary_classifier:
            self.model.aux_classifier = None

        # Freeze the backbone weights
        if freeze_backbone:
            self.model.backbone.requires_grad_(False)

        self.train_output_crop_margin = train_output_crop_margin
        self.eval_output_crop_margin = eval_output_crop_margin

    @classmethod
    def create_instance(cls, num_classes, use_data_parallel=False, device_ids=None,
                        train_output_crop_margin=None, eval_output_crop_margin=None,
                        model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        return cls(num_classes, use_data_parallel, device_ids,
                   train_output_crop_margin=train_output_crop_margin,
                   eval_output_crop_margin=eval_output_crop_margin, **model_kwargs)

    def forward(self, inputs):
        input_images = inputs
        output = self.forward_model(input_images)['out']
        assert output.shape[-2:] == input_images.shape[-2:]

        # Crop the output if a crop margin is specified
        margin = self.train_output_crop_margin if self.training else self.eval_output_crop_margin
        if margin is not None:
            height, width = calculate_cropped_size(output.shape[-2:], margin)
            output = F.crop(output, top=margin, left=margin, height=height, width=width)

        return output
    

def load_partial_state_dict(model_file: str, keys_to_exclude: List[str],
                            extract_dict_branch: Optional[str] = None) -> Dict[str, Any]:
    """Loads a partial model

    Args:
        model_file: The file containing the weights used to specify where the model should be
            loaded from. The string should include the absolute path for the file.
        keys_to_exclude: Any layers in the loaded dict that includes any of these keywords will be
            excluded from the loaded weights.
        extract_dict_branch: A dictionary sub-branch to extract the state dict from. For example in
            hovernet we are just using the ['desc'] sub-branch.

    Returns:
        Returns a dict of weights with the specified layers excluded.
    """
    loaded_model_weights = torch.load(model_file)
    if 'state_dict' in loaded_model_weights:
        loaded_model_weights = loaded_model_weights['state_dict']
    if extract_dict_branch is not None and extract_dict_branch in loaded_model_weights:
        loaded_model_weights = loaded_model_weights[extract_dict_branch]
    partial_dict = {k: v for k, v in loaded_model_weights.items() if not any(exclude in k for exclude in keys_to_exclude)}
    return partial_dict


def calculate_cropped_size(
    input_size: Tuple[int, int],
    crop_margin: Optional[int],
) -> Tuple[int, int]:
    """Calculate the output size given an input size and crop margin.

    Args:
        input_size: The size of the box before cropping.
        crop_margin: The amount to crop from each side of the box.

    Returns:
        The size of the box after cropping.
    """
    if crop_margin is None:
        return input_size
    output_size = (input_size[0] - crop_margin * 2, input_size[1] - crop_margin * 2)
    if any(x < 0 for x in output_size):
        raise ValueError(f'Crop margin {crop_margin} is too large for the input size {input_size}')
    return output_size
