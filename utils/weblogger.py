import numpy as np
from torch.autograd import Variable
from visdom import Visdom
from PIL import Image

# Mask
background = [200, 222, 250]
c1 = [0,0,205] #grasp red
c2 = [34,139,34] #cut green
c3 = [0,255,255] #scoop bluegreen
c4 = [165,42,42] #contain dark blue
c5 = [128,64,128] #pound purple
c6 = [51,153,255] #support orange
c7 = [184,134,11] #wrap-grasp light blue
c8 = [0,153,153]
c9 = [0,134,141]
c10 = [184,0,141]
c11 = [184,134,0]
c12 = [184,134,223]
label_colours = np.array([background, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

class Dashboard:

    def __init__(self, port, envname):
        self.vis = Visdom(port=port)
        self.logPlot = None
        self.dataCount = 0
        self.envname = envname

    def appendlog(self, value, logname, addcount=True):
        if addcount:
            self.dataCount+=1
        if self.logPlot:
            self.vis.updateTrace(
                X=np.array([self.dataCount]),
                Y=np.array([value]),
                win=self.logPlot,
                name=logname,
                env=self.envname
            )
        else:
            self.logPlot = self.vis.line(np.array([value]), np.array([self.dataCount]), env=self.envname, opts=dict(title=self.envname,legend=[logname]))

    def colorize_mask(self, mask):
        # mask: numpy array of the mask
        color_curr_mask = label_colours.take(mask, axis=0).astype('uint8')

        return color_curr_mask


    def affordanceMapping(self, image):
        image = image.transpose(1,2,0)
        image = np.asarray(np.argmax(image, axis=2), dtype=np.uint8)
        image = self.colorize_mask(image)
        image = image.transpose(2, 0, 1)

        return image



    def image(self, image, title, mode='img',denorm=True,caption=''): #denorm: de-normalization
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        if len(list(image.shape))==4:
            image = image[0]
        if denorm:
            #image[0] = image[0] * .229 + .485
            #image[1] = image[1] * .224 + .456
            #image[2] = image[2] * .225 + .406
            image[0] = image[0] * .153398 + .358388
            image[1] = image[1] * .137741 + .348858
            image[2] = image[2] * .230031 + .294015
            image = image.sub_(image.min())
            image = image.div_(image.max())
        image = image.numpy()
        if image.shape[0] > 3:
            image = self.affordanceMapping(image)
        self.vis.image(image, env=self.envname+'-'+mode, opts=dict(title=title,caption=caption))
        
    def text(self, text, mode):
        self.vis.text(text, env=self.envname+'-'+mode)

