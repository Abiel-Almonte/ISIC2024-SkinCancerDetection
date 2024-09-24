import torch 
from torchsummary import summary
from architectures import LightMultiModalNN

if '__main__' == __name__:
    image= torch.randn(16, 3, 256, 256, device='cuda')
    cont_tabular= torch.randn(16, 36, device='cuda')
    bin_tabular=  torch.randint(0, 6, (16, 1), device='cuda')


    model= LightMultiModalNN()
    model.to('cuda')
    output= model(image, cont_tabular, bin_tabular)
    #summary(model, input_size= [(3, 336, 336), (31,), (6,)])