import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EfficientUNet(nn.Module):
    def __init__(self, n_classes=32):
        super().__init__()
        
        self.encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.encoder_layers = self.encoder.features
        
        self.adjust_conv4 = nn.Conv2d(1280, 640, kernel_size=1)
        self.adjust_conv3 = nn.Conv2d(160, 320, kernel_size=1)
        self.adjust_conv2 = nn.Conv2d(64, 160, kernel_size=1)
        
        self.decoder1 = self.decoder_block(1280, 640)
        self.decoder2 = self.decoder_block(640, 320)
        self.decoder3 = self.decoder_block(320, 160)
        self.decoder4 = self.decoder_block(160, 64)
            
        self.conv_last = nn.Conv2d(64, n_classes, 1)


    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
        )


    def forward(self, x):
        # Encoder
        conv1 = self.encoder_layers[0:2](x)
        conv2 = self.encoder_layers[2:4](conv1)
        conv3 = self.encoder_layers[4:6](conv2)
        conv4 = self.encoder_layers[6:8](conv3)
        conv5 = self.encoder_layers[8:](conv4)
        # Decoder
        up6 = self.decoder1(conv5)
        conv4 = self.adjust_conv4(conv4)
        conv4 = F.interpolate(conv4, size=up6.shape[2:], mode='bilinear', align_corners=True)
        up6= self.attention1(up6, conv4)
        up7 = self.decoder2(up6 + conv4)
        
        conv3 = self.adjust_conv3(conv3)
        conv3 = F.interpolate(conv3, size=up7.shape[2:], mode='bilinear', align_corners=True)
        up7= self.attention2(up7, conv4)
        up8 = self.decoder3(up7 + conv3)
        
        conv2 = self.adjust_conv2(conv2)
        conv2 = F.interpolate(conv2, size=up8.shape[2:], mode='bilinear', align_corners=True)
        up9 = self.decoder4(up8 + conv2)


        out = self.conv_last(up9)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        return out
    
class TabularNet(nn.Module):
    def __init__(self, n_cont_features, n_bin_features):
        super().__init__()
        self.embed = nn.Embedding(n_bin_features, 64)
        self.proj= nn.Linear(n_cont_features, 64)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.bn1= nn.BatchNorm1d(128)
        self.bn2= nn.BatchNorm1d(64)


    def forward(self, cont, cat):
        x = torch.cat([self.proj(cont), self.embed(cat).sum(1)], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x= self.dropout(x)
        return F.relu(self.fc3(x))


class EfficientUNetWithTabular(nn.Module):
    def __init__(self, n_cont_features, n_bin_features):
        super().__init__()
        self.unet = EfficientUNet()
        self.tabnet = TabularNet(n_cont_features, n_bin_features)
        self.fc = nn.Linear(64, 1)


    def forward(self, img, cont, cat):
        img_features = self.unet(img)        
        tab_features = self.tabnet(cont, cat)

        combined = torch.cat([img_features, tab_features], dim=1)
        return self.fc(combined)

if '__main__' == __name__:
    img = torch.randn(32, 3, 224, 224)
    cont = torch.randn(32, 10)
    cat = torch.randint(0, 5, (32, 1))
    
    model = EfficientUNetWithTabular(10, 5)
    output = model(img, cont, cat)
    print(output.squeeze(dim=1))
    print(model, f'\nTrainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e}')