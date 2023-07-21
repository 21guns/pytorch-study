from pathlib import Path
from torchvision import datasets, transforms
from helper_functions import *
import data_setup
import torch

data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

IMG_SIZE = 224

manual_transform = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,  test_dir=test_dir,
    transforms=manual_transform, batch_size=BATCH_SIZE
)

image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]

height, width = 224, 224
color_channels = 3
patch_size = 16

number_of_patches = int((height * width) / patch_size ** 2)

embedding_layer_input_shpe = (height, width, color_channels)
embedding_layer_output_shpe = (number_of_patches, patch_size **2 * color_channels)

image_permuted = image.permute(1, 2, 0)

# # Setup hyperparameters and make sure img_size and patch_size are compatible
# img_size = 224
# patch_size = 16
# num_patches = img_size/patch_size 
# assert img_size % patch_size == 0, "Image size must be divisible by patch size" 
# print(f"Number of patches per row: {num_patches}\
#         \nNumber of patches per column: {num_patches}\
#         \nTotal patches: {num_patches*num_patches}\
#         \nPatch size: {patch_size} pixels x {patch_size} pixels")

# # Create a series of subplots
# fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
#                         ncols=img_size // patch_size, 
#                         figsize=(num_patches, num_patches),
#                         sharex=True,
#                         sharey=True)

# # Loop through height and width of image
# for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
#     for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width
        
#         # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
#         axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height 
#                                         patch_width:patch_width+patch_size, # iterate through width
#                                         :]) # get all color channels
        
#         # Set up label information, remove the ticks for clarity and set labels to outside
#         axs[i, j].set_ylabel(i+1, 
#                              rotation="horizontal", 
#                              horizontalalignment="right", 
#                              verticalalignment="center") 
#         axs[i, j].set_xlabel(j+1) 
#         axs[i, j].set_xticks([])
#         axs[i, j].set_yticks([])
#         axs[i, j].label_outer()

# # Set a super title
# fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
# plt.show()

conv2d = torch.nn.Conv2d(
    in_channels=3,
    out_channels=768,
    kernel_size=patch_size,
    stride=patch_size,
    padding=0
)

image_out_of_conv = conv2d(image.unsqueeze(0))

flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
image_out_of_conv_flattened = flatten(image_out_of_conv)

class PatchEmbdding(torch.nn.Module):
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768):
        super().__init__()
        self.patcher = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x):
        image_resolution = x.shape[-1]
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)
        
patchify = PatchEmbdding(
    in_channels=3,
    patch_size=16,
    embedding_dim=768
)
patch_embedded_image = patchify(image.unsqueeze(0))

batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

class_token =  torch.nn.Parameter(torch.ones(batch_size, 1, embedding_dimension), requires_grad=True)
patch_embedded_image_with_class_embedding = torch.cat([class_token, patch_embedded_image], dim=1)

embedding_dimension = patch_embedded_image_with_class_embedding.shape[-1]
position_embedding = torch.nn.Parameter(torch.ones(1, number_of_patches + 1, embedding_dimension), requires_grad=True)

class MultiheadSelfAttentionBlock(torch.nn.Module):
    def __init__(self, num_heads:int=12, embedding_dim:int=768, attn_dropout:float=0):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.multihead_attn = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.multihead_attn(x, x, x, need_weights=False)
        return x
    
class MLPBlock(torch.nn.Module):
    def __init__(self, embedding_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_size, embedding_dim),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embedding_dim:int=768, num_heads:int=12, mlp_size:int=3072, mlp_dropout:float=0.1, attn_dropout:float=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(num_heads=num_heads, embedding_dim=embedding_dim, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)
    
    def forward(self, x):
        x = self.attention_block(x) + x
        x = self.mlp_block(x) + x
        return x
    
torch_transformer_encode_layer = torch.nn.TransformerEncoderLayer(
    embedding_dim=768,
    num_heads=12,
    dim_feedfoward=3072,
    dropout=0.1,
    activation="gelu",
    batch_first=True,
    norm_first = True
)

class ViT(torch.nn.Module):
    def __init__(self, img_size:int=224, in_channels:int=3, patch_size:int=16, num_transformer_layers:int=12, 
                 embedding_dim:int=768, mlp_size:int=3072, num_heads:int=12, attn_dropout:float=0, mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1, num_class:int=1000):
        super().__init__()
        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.class_embedding = torch.nn.Parameter(torch.randn(1, 1,  embedding_dim), requires_grad=True)
        self.position_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)

        self.patch_embedding = PatchEmbdding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )

        self.transformer_encoder = torch.nn.Sequential(
            *[TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                mlp_dropout=mlp_dropout
            ) for _ in range(num_transformer_layers)]
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, num_class)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]

        calss_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)

        x = torch.cat([calss_token, x], dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x