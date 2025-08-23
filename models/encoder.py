import torch
import torch.nn as nn
import math
import torchvision.models as models

from typing import Union, Optional

class BaseEncoder(nn.Module):
    """
    A wrapper around a given neural network that extracts
    activations from a specified layer during forward pass
    - using forward hooks
    """
    def __init__(self, net: nn.Module, layer: Union[str, int] = -2):
        """
        net: a neural network
        layer: the layer from which to extract the hidden representation
        """
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self._register_hook()
 
    def _find_layer(self) -> Optional[nn.Module]:
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None) # returns None if layer not found
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None
    
    def _register_hook(self) -> None:
        """
        General guidelines to register a forward hook:
        1. Define a hook function that takes three arguments: module, input, output
            1.a don't pass "self" as pytorch expects these three arguments
        2. Get the layer from the network
        3. Register the hook and use the output as intended
        4. Input can be used/modified with pre_forward_hook
        """
        def hook(_, __, output: torch.Tensor) -> None:
            self.hidden = output
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        self.hook_handle = layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        # remove the hook when done     
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

  
    def forward(self, x) -> torch.Tensor:
        if self.layer == -1:
            return self.net(x)       
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer ({self.layer}) not found'
        return hidden

# Below we define classes for different encoder backbones

class ResNetEncoder(BaseEncoder):
    """
    A wrapper around a ResNet model that extracts
    activations from a specified layer during forward pass
    """
    def __init__(self, model: nn.Module, layer: Union[str, int] = -2, dataset: str = 'imagenet',
                 width_multiplier: int = 1, pretrained: bool = False, **kwargs):
        """
        layer: the layer from which to extract the hidden representation
        dataset: the dataset on which the model will be pretrained, either 'imagenet' or 'cifar'
        width_multiplier: the width multiplier for the ResNet model (1, 2, 4, 8, ...)
        pretrained: whether to load pretrained weights (does not work for pretrained weights if width_multiplier > 1)
        """
        
        super().__init__(model, layer)
        self.width_multiplier = int(width_multiplier)
        self.pretrained = pretrained
        if self.width_multiplier != 1:
            assert not pretrained, 'pretrained weights not available for wider ResNet'
        self.create_wider_resnet()
        if dataset == 'cifar' or 'cifar' in dataset or dataset=='svhn':
            self.modify_for_cifar()
        # for SSL, we do not need the final fc layer
        self.net.fc = nn.Identity()

    def create_wider_resnet(self) -> None:
        if self.width_multiplier == 1:
            return   
        # modify the first conv layer
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.net.bn1 = nn.BatchNorm2d(64 * self.width_multiplier)
        
        if not self.pretrained:
            # ✅ Reinitialize weights
            self._initialize_weights(self.net.conv1)
            self._initialize_weights(self.net.bn1)
        
        # modify the subsequent layers
        for layer in [self.net.layer1, self.net.layer2, self.net.layer3, self.net.layer4]:
            for bottleneck in layer:
                bottleneck.conv1 = self._wider_bottleneck(bottleneck.conv1)
                bottleneck.bn1 = nn.BatchNorm2d(bottleneck.conv1.out_channels)
                bottleneck.conv2 = self._wider_bottleneck(bottleneck.conv2)
                bottleneck.bn2 = nn.BatchNorm2d(bottleneck.conv2.out_channels)
                bottleneck.conv3 = self._wider_bottleneck(bottleneck.conv3)
                bottleneck.bn3 = nn.BatchNorm2d(bottleneck.conv3.out_channels)

                # modify the shortcut connection
                if bottleneck.downsample is not None:
                    bottleneck.downsample[0] = self._wider_bottleneck(bottleneck.downsample[0])
                    bottleneck.downsample[1] = nn.BatchNorm2d(bottleneck.downsample[0].out_channels, )

    def _wider_bottleneck(self, conv) -> nn.Conv2d:
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        bias = conv.bias is not None
        widened_conv = nn.Conv2d(in_channels * self.width_multiplier,
                                 out_channels * self.width_multiplier,
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding = padding,
                                 bias = bias)
        
        if not self.pretrained:
            # ✅ Reinitialize weights
            self._initialize_weights(widened_conv)
        
        # delete conv to save memory
        del conv
        return widened_conv
        

    def modify_for_cifar(self) -> None:
        # replace the first conv layer to adapt to CIFAR
        self.net.conv1 = nn.Conv2d(3, 64 * self.width_multiplier, kernel_size = 3, stride = 1, padding = 1, bias = False)
        # remove the first max pooling operation
        self.net.maxpool = nn.Identity()

    def _initialize_weights(self, layer: Union[nn.Conv2d, nn.BatchNorm2d]) -> None:

        # ✅ Reinitialize weights
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)

class ViTEncoder(BaseEncoder):
    """
    A wrapper around a Vision Transformer (ViT) model that inherits from BaseEncoder.
    """
    def __init__(self, vit_model: nn.Module,
                 image_size: int = 32,
                 patch_size: int = 4, # Smaller patch size for CIFAR
                 layer: str = 'encoder', # Hook into the output of the entire encoder block
                 **kwargs):
        """
        Args:
            vit_model: The torchvision ViT model instance.
            image_size: The input image size (e.g., 32 for CIFAR).
            patch_size: The desired patch size.
            layer: The name of the module to hook for feature extraction.
                   'encoder' is a good choice to get the final hidden states.
        """
        # 1. Initialize the BaseEncoder first
        super().__init__(net=vit_model, layer=layer)

        # Store parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = vit_model.hidden_dim

        # 2. Overwrite the original patch projection with our custom one
        # This is the key step to adapt the ViT for different input/patch sizes.
        self.net.conv_proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self._initialize_weights(self.net.conv_proj) # Initialize the new conv layer

        # 3. Re-initialize positional embeddings for the new sequence length
        self._initialize_vit_encoder()

        # 4. For SSL, we don't need the final classification head
        self.net.heads = nn.Identity()


    def _initialize_vit_encoder(self):
        """Recalculates and reinitializes the positional embeddings."""
        num_patches = (self.image_size // self.patch_size) ** 2
        seq_length = num_patches + 1  # Add 1 for the [CLS] token

        # Re-create the positional embedding parameter with the new size
        self.net.encoder.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, self.hidden_dim).normal_(std=0.02)
        )

    def _initialize_weights(self, layer: nn.Conv2d):
        """Initializes the weights of the patch projection layer."""
        fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        nn.init.trunc_normal_(layer.weight, std=math.sqrt(1 / fan_in))
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass and extracts the [CLS] token embedding.
        """
        # 1. Let the BaseEncoder run the full forward pass on the modified ViT.
        #    The hook will capture the output of the `self.net.encoder` block.
        #    This output contains all token embeddings (CLS + patches).
        hidden_representation = super().forward(x)

        # 2. From the output of the encoder, we only need the [CLS] token,
        #    which is always the first token in the sequence.
        cls_embedding = hidden_representation[:, 0]

        return cls_embedding

    def interpolate_pos_encoding(self, x: torch.Tensor):
        """
        Needed while using pretrained weights.
        (Implementation can be added here if you need to load ImageNet weights)
        """
        # TODO: Implement positional embedding interpolation if fine-tuning
        # from a model with different resolution.
        pass