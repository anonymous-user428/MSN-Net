import torch, torch.nn as nn, torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x 
        x = self.avg_pool(x.transpose(1,3))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).transpose(1,3)
        return module_input * x, x

class SEModule2(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x.transpose(1,2))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).transpose(1,2)
        return module_input * x, x

class SEModule_naive(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule_naive, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x, x
    

class SEModule_combine_v2(nn.Module):
    def __init__(self, freq_ch, time_ch, fea_ch) -> None:
        super().__init__()
        self.freq_se = SEModule(freq_ch, 1)
        self.time_se = SEModule2(time_ch, 1)
        self.total_se = SEModule_naive(fea_ch, 1)
        self.maps = {}
    
    def forward(self, x):
        freq_output, freq_map = self.freq_se(x)
        time_output, time_map = self.time_se(x)
        channel_output, channel_map = self.total_se(x)
        # save the maps
        self.maps.update({
            "freq_map": freq_map,
            "time_map": time_map,
            "channel_map": channel_map,
        })
        return freq_output + time_output + channel_output + x
    

if __name__ == "__main__":
    input = torch.randn((16, 128, 16, 18))
    net = SEModule(16, 1)
    output = net(input)
    print(output.shape)

