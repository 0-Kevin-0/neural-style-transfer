import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import copy
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载图像
def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def center_crop(image, size):
    w, h = image.size
    target_w, target_h = size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return image.crop((left, top, right, bottom))

def preprocess_images(content_path, style_path):
    content = load_image(content_path)
    style = load_image(style_path)

    # 取两个图中最小的尺寸作为裁剪目标
    target_size = (min(content.size[0], style.size[0]),
                   min(content.size[1], style.size[1]))

    content = center_crop(content, target_size)
    style = center_crop(style, target_size)

    transform = transforms.ToTensor()
    content_tensor = transform(content).unsqueeze(0)
    style_tensor = transform(style).unsqueeze(0)
    return content_tensor, style_tensor


# 显示图像
def imshow(tensor, title=None):
    image = tensor.clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 计算内容损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Gram矩阵
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# 风格损失
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# 正则化模块
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# 构建模型
def get_model_and_losses(cnn, normalization_mean, normalization_std,
                         style_img, content_img):
    cnn = copy.deepcopy(cnn)

    # ✅ 确保所有模块放入 GPU
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            continue

        model.add_module(name, layer.to(device))  # ✅ 将每一层也移动到 device

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target).to(device)  # ✅
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target).to(device)  # ✅
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# 执行风格迁移
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                       num_steps=200, style_weight=1000000, content_weight=1):
    print("开始迁移...")
    model, style_losses, content_losses = get_model_and_losses(cnn, norm_mean, norm_std,
                                                                style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    for step in range(num_steps):
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            return loss

        optimizer.step(closure)

        if step % 50 == 0:
            print(f"Step {step}")

    input_img.data.clamp_(0, 1)
    return input_img


print("CUDA 是否可用：", torch.cuda.is_available())
print("CUDA 版本：", torch.version.cuda)
print("PyTorch 是否用 GPU：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")
# Main 流程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
cnn = models.vgg19(pretrained=True).features.to(device).eval()
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

content_img, style_img = preprocess_images("Testin/content_4.jpg", "Testin/style_4.jpg")
content_img = content_img.to(device)
style_img = style_img.to(device)
input_img = content_img.clone()

output = run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img)
imshow(output, title="Stylized Image")
