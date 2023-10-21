import torch
from torchvision.models import vgg19, vgg16
from torchvision import transforms

import dnnlib


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2))
    return x / (norm_factor + eps)


class CustomVGG(torch.nn.Module):
    def __init__(self, vgg_model: str):
        super().__init__()
        if vgg_model == "vgg19":
            self.layers = torch.nn.Sequential(*list(vgg19(pretrained=True).eval().features.children()))
        elif vgg_model == "vgg16":
            self.layers = torch.nn.Sequential(*list(vgg16(pretrained=True).eval().features.children()))
        else:
            raise NotImplementedError("Select from vgg16 and vgg19")
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

    def forward(self, x):
        # from -1,1 to 0,1
        x = (x + 1.0) / 2.0
        x = normalize(x)

        intermediate_results = []
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], torch.nn.MaxPool2d):
                x = self.avg_pool(x)
            else:
                x = self.layers[i](x)

                if isinstance(self.layers[i], torch.nn.ReLU) and isinstance(self.layers[i + 1], torch.nn.MaxPool2d):
                    intermediate_results.append(x)

        flat_features = torch.cat([res.reshape(-1) for res in intermediate_results], dim=0)
        gram_features = torch.cat([self.compute_gram_matrix(res).reshape(-1) for res in intermediate_results], dim=0)
        flat_features = normalize_activation(flat_features)
        return flat_features# , gram_features

    @staticmethod
    def compute_gram_matrix(feature_map):
        batch, channels, height, width = feature_map.size()
        features = feature_map.view(batch * channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch * channels * height * width)


class NvidiaVGG16:
    def __init__(self, device='cuda'):
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.model = torch.jit.load(f).eval().to(device)

    def __call__(self, img):
        img = (img + 1) / 2 * 255
        return self.model(img, resize_images=False, return_lpips=True)


if __name__ == "__main__":
    nv_vgg = NvidiaVGG16()
    vgg = CustomVGG("vgg16").to("cuda")
    x = torch.ones([1, 3, 256, 256]).to("cuda")
    flat_features = vgg(x)
    flat_features_ = nv_vgg(x)
    print()
