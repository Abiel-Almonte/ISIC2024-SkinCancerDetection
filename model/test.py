import torch 
from torchsummary import summary
from architectures import LightMutiModalNN

if '__main__' == __name__:
    image= torch.randn(16, 3, 336, 336)
    cont_tabular= torch.randn(16, 36)
    bin_tabular=  torch.randint(0, 6, (16, 1))


    model= LightMutiModalNN()
    output= model(image, cont_tabular, bin_tabular)
    
    summary(model, input_size= [(3, 336, 336), (31,), (5,)])