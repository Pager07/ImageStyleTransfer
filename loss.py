import torch 
import torch.nn as nn
from torchvision import transforms
#https://github.com/enomotokenji/pytorch-Neural-Style-Transfer/blob/master/loss.py
# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

class ContentLoss(nn.Module):
    def __init__(self,target,weight):
        super(ContentLoss, self).__init__()

        # we 'detach' the target content from the tree used
        self.target = target.detach() *weight

        #to dynamically compute the gradient: this a state value
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()
    
    def forward(self, input):

        self.loss = self.criterion(input* self.weight,self.target)
        self.output = input 
        return self.output
    
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    

class GramMatrix(nn.Module):
    def forward(self, input):
        """
        Given feature maps of a layer, returns Gram matrix
        Args:
        =====
        input: feature maps if a layer
        """
        a,b,c,d = input.size() #batch(1),features,w,h

        features = input.view(a*b, c*d) #resize F_XL into hat F_XL, so a row is 1 feature map

        G = torch.mm(features, features.t()) #compute gram product 

        #we 'normalize' the values of the gram matrix
        #by dividing by the number of element in each feature maps 
        return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self,target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight #get the feautre maps
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()
    
    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight) #?
        self.loss = self.criterion(self.G,self.target)
        return self.output