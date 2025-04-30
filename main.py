import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# Utility functions
# --------------------
def load_image(path):
    return Image.open(path).convert('RGB')

def center_crop(image, size):
    w, h = image.size
    tw, th = size
    left = (w - tw) // 2
    top  = (h - th) // 2
    return image.crop((left, top, left + tw, top + th))

def preprocess_images(content_path, style_path):
    content = load_image(content_path)
    style   = load_image(style_path)
    target_size = (
        min(content.size[0], style.size[0]),
        min(content.size[1], style.size[1])
    )
    content = center_crop(content, target_size)
    style   = center_crop(style, target_size)
    to_tensor = transforms.ToTensor()
    content_tensor = to_tensor(content).unsqueeze(0).to(device)
    style_tensor   = to_tensor(style).unsqueeze(0).to(device)
    return content_tensor, style_tensor

def imshow(tensor, title=None):
    img = tensor.cpu().clone().detach().squeeze(0)
    img = transforms.ToPILImage()(img)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# --------------------
# Loss modules
# --------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0.0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0.0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std  = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# --------------------
# Model + losses builder
# --------------------
def get_model_and_losses(cnn, mean, std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    # Freeze VGG parameters
    for p in cnn.parameters():
        p.requires_grad_(False)

    normalization = Normalization(mean, std).to(device)
    model = nn.Sequential(normalization)

    style_layers   = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
    content_layers = ['conv_4_2']

    style_losses, content_losses = [], []
    block, conv = 1, 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv += 1
            name = f'conv_{block}_{conv}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{block}_{conv}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{block}'
            block += 1
            conv = 0
        else:
            continue

        model.add_module(name, layer.to(device))

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target).to(device)
            model.add_module(f"content_loss_{block}_{conv}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target_feature = model(style_img).detach()
            sl = StyleLoss(target_feature).to(device)
            model.add_module(f"style_loss_{block}_{conv}", sl)
            style_losses.append(sl)

    # Trim off layers after last loss module
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            model = model[: i + 1]
            break

    return model, style_losses, content_losses

# --------------------
# Style transfer loop
# --------------------
def run_style_transfer(cnn, mean, std, content_img, style_img,
                       num_steps=300, style_weight=1e6, content_weight=1):
    # Initialize input as noise + content
    input_img = content_img.clone()
    input_img = torch.randn_like(content_img) * 0.1 + input_img
    input_img = input_img.to(device).requires_grad_()

    optimizer = optim.LBFGS([input_img])

    model, style_losses, content_losses = get_model_and_losses(
        cnn, mean, std, style_img, content_img
    )

    print("Starting optimization...")
    run = [0]
    while run[0] < num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            s_loss = sum(sl.loss for sl in style_losses)
            c_loss = sum(cl.loss for cl in content_losses)
            loss = style_weight * s_loss + content_weight * c_loss
            loss.backward()
            return loss

        optimizer.step(closure)
        run[0] += 1

        if run[0] % 50 == 0:
            s = sum(sl.loss.item() for sl in style_losses)
            c = sum(cl.loss.item() for cl in content_losses)
            print(f"Step {run[0]:03d}: Style Loss {s:.2f}, Content Loss {c:.2f}")

    # Final clamp
    input_img.data.clamp_(0, 1)
    return input_img.detach()


def tile_style_transfer(cnn, mean, std, content, style,
                        tile_size=512, overlap=64,
                        num_steps=200, style_weight=1e6, content_weight=1):
    """
    Split content/style into overlapping tiles of size <= tile_size,
    run style transfer on each, blend back with a cropped Hann window.
    """
    _, C, H, W = content.shape
    stride = tile_size - overlap

    # precompute a full-size 2D Hann window
    win1d = torch.hann_window(tile_size, periodic=False,
                              dtype=content.dtype, device=content.device)
    win2d_full = (win1d.unsqueeze(1) * win1d.unsqueeze(0)) \
                  .unsqueeze(0).unsqueeze(0)  # 1×1×tile×tile

    # accumulators
    output_sum = torch.zeros_like(content)
    weight_sum = torch.zeros_like(content)

    # compute tile start positions
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    # ensure last tile covers the border
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    for y in ys:
        for x in xs:
            y0, x0 = y, x
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            c_tile = content[:, :, y0:y1, x0:x1]
            s_tile = style[:,   :, y0:y1, x0:x1]

            # stylize this patch
            stylized = run_style_transfer(
                cnn, mean, std,
                c_tile, s_tile,
                num_steps=num_steps,
                style_weight=style_weight,
                content_weight=content_weight
            )

            # get actual tile size
            h, w_ = stylized.shape[-2], stylized.shape[-1]

            # crop the window to match this tile
            w_crop = win2d_full[:, :, :h, :w_]

            # blend
            output_sum[:, :, y0:y1, x0:x1] += stylized * w_crop
            weight_sum[:, :, y0:y1, x0:x1] += w_crop

    # normalize to finalize
    return output_sum / weight_sum


# --------------------
# Main execution
# --------------------
def main():
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    # load full-res content & style
    content, style = preprocess_images(
        "Testin/content_4.jpg",
        "Testin/style_4.jpg"
    )

    # run tile-based stylization
    output = tile_style_transfer(
        cnn, norm_mean, norm_std,
        content, style,
        tile_size=512, overlap=64,
        num_steps=200,
        style_weight=1e6,
        content_weight=1
    )

    imshow(output, title="High-Res Stylized (Tiles)")

if __name__ == "__main__":
    main()