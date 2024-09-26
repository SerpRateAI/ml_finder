import torch
import torch.nn as nn
from torchview import draw_graph
from PIL import Image

# Define the ConvAutoencoder2 architecture (not necessary for loading the full model, but included for clarity)
class ConvAutoencoder2(nn.Module):
    def __init__(self, bottle, inputs=1):
        super(ConvAutoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 4, kernel_size=5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

# Load the entire model, which includes both the architecture and the weights
model = torch.load('model.pth', map_location=torch.device('cpu'))

# Generate model visualization with torchview
input_size = (1, 1, 200, 72)  # Adjust the input size if necessary, this is for a single 128x128 grayscale image
model_graph = draw_graph(model, input_size=input_size, device='meta')
model_graph.visual_graph.graph_attr['dpi'] = '600' 
model_graph.visual_graph.render('model_vis', format='png')
# This will save the file as 'model_vis.png'
# img = Image.open('model_vis.png')
# img.save('model_vis_high_dpi.png', dpi=(600, 600))


