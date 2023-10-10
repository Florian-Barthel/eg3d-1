import torch
from torchvision.models import vgg19, vgg16


class CustomVGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(*list(vgg19(pretrained=True).features.children())[:-2])
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

    def forward(self, x):
        intermediate_results = []
        for layer in self.layers:
            if isinstance(layer, torch.nn.MaxPool2d):
                x = self.avg_pool(x)
            else:
                x = layer(x)
                if isinstance(layer, torch.nn.ReLU):
                    intermediate_results.append(x)

        intermediate_vector = torch.cat([res.reshape(-1) for res in intermediate_results], dim=0)
        return x, intermediate_vector


class CustomVGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(*list(vgg16(pretrained=True).features.children())[:-2])
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

    def forward(self, x):
        intermediate_results = []
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], torch.nn.MaxPool2d):
                x = self.avg_pool(x)
            else:
                x = self.layers[i](x)
                # if i == len(self.layers) - 1:
                #     intermediate_results.append(x)

                if isinstance(self.layers[i], torch.nn.ReLU) and isinstance(self.layers[i + 1], torch.nn.MaxPool2d):
                    intermediate_results.append(x)

        intermediate_vector = torch.cat([res.reshape(-1) for res in intermediate_results], dim=0)
        return x, intermediate_vector


if __name__ == "__main__":

    vgg = CustomVGG16()
    x = torch.zeros([1, 3, 256, 256])
    vgg(x)
    print()