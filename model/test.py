import torch 
from torchsummary import summary
from architectures import EfficientNetEVAModel

if '__main__' == __name__:
    image= torch.randn(16, 3, 336, 336)
    cont_tabular= torch.randn(16, 31)
    bin_tabular=  torch.randint(0, 5, (16, 1))


    model= EfficientNetEVAModel()
    output= model(image, cont_tabular, bin_tabular)
    summary(model, input_size= [(3, 336, 336), (31,), (5,)],  batch_dim=32, depth=3, verbose=1)