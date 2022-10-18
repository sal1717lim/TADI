import matplotlib.pyplot as plt
from skimage import io as skio
import os
def gris_depuis_couleur(im):
    """ Transforme une image couleur en image a niveaux de gris"""
    return im[:,:,:3].sum(axis=2)/3
img=skio.imread("Images/retina2.gif")
lst=os.listdir("opening")
if not os.path.exists("tophat2"):
    os.mkdir("tophat2")
plt.figure(figsize=(20, 20), dpi=80)
cpt=1
for i in lst:
    if i.startswith("retina2"):
        x=gris_depuis_couleur(skio.imread("opening/"+i))
        tp=img-x
        plt.subplot(4,4,cpt)
        plt.title(i.replace("retina2","")[:-4])
        plt.imshow(tp,vmin=0,vmax=255,cmap="gray")
        cpt+=1

plt.savefig("tophat.png")