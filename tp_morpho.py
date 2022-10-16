#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:23:50 2018
Modified Oct 2020, Oct 2021

@author: Said Ladjal,Isabelle Bloch
"""


#%% SECTION 1 -- inclusion of packages


import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import io as skio


import skimage.morphology as morpho  
import skimage.feature as skf
from scipy import ndimage as ndi

#%% SECTION 2 -- Useful functions

def viewimage(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI et MAXI seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase=' ' 
    elif platform.system()=='Linux': #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'
    elif platform.system()=='Windows':
        prephrase='start /B "D:/GIMP/bin/gimp-2.10.exe" -a '#Remplacer D:/... par le chemin de votre GIMP
        endphrase= ''
    else:
        print('Systeme non pris en charge par l affichage GIMP')
        return 'erreur d afficahge'
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M

    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    
    nomfichier=tempfile.mktemp('TPIMA.png')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)


def viewimage_color(im,normalise=True,MINI=0.0, MAXI=255.0):
    """ Cette fonction fait afficher l'image EN NIVEAUX DE GRIS 
        dans gimp. Si un gimp est deja ouvert il est utilise.
        Par defaut normalise=True. Et dans ce cas l'image est normalisee 
        entre 0 et 255 avant d'être sauvegardee.
        Si normalise=False MINI(defaut 0) et MAXI (defaut 255) seront mis a 0 et 255 dans l'image resultat
        
    """
    imt=np.float32(im.copy())
    if platform.system()=='Darwin': #on est sous mac
        prephrase='open -a GIMP '
        endphrase=' ' 
    elif platform.system()=='Linux': #SINON ON SUPPOSE LINUX (si vous avez un windows je ne sais comment faire. Si vous savez dites-moi.)
        prephrase='gimp '
        endphrase= ' &'
    elif platform.system()=='Windows':
        prephrase='start /B "D:/GIMP/bin/gimp-2.10.exe" -a '#Remplacer D:/... par le chemin de votre GIMP
        endphrase= ''
    else:
        print('Systeme non pris en charge par l affichage GIMP')
        return 'erreur d afficahge'
    
    if normalise:
        m=imt.min()
        imt=imt-m
        M=imt.max()
        if M>0:
            imt=imt/M
    else:
        imt=(imt-MINI)/(MAXI-MINI)
        imt[imt<0]=0
        imt[imt>1]=1
    
    nomfichier=tempfile.mktemp('TPIMA.pgm')
    commande=prephrase +nomfichier+endphrase
    skio.imsave(nomfichier,imt)
    os.system(commande)


def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')

            

def couleurs_alea(im):
    """ 
    Donne des couleurs aleatoires a une image en niveau de gris.
    Cette fonction est utile lorsque le niveua de gris d'interprete comme un numero
      de region. Ou encore pour voir les leger degrades d'une teinte de gris.
      """
    sh=im.shape
    out=np.zeros((sh[0],sh[1],3),dtype=np.uint8)
    nbcoul=np.int32(im.max())
    tabcoul=np.random.randint(0,256,size=(nbcoul+1,3))
    tabcoul[0,:]=0
    for k in range(sh[0]):
        for l in range(sh[1]):
            out[k,l,:]=tabcoul[im[k,l]]
    return out

def gris_depuis_couleur(im):
    """ Transforme une image couleur en image a niveaux de gris"""
    return im[:,:,:3].sum(axis=2)/3
    
#%% SECTION 3 -- Examples of functions for this work

# Binary images 
#im=skio.imread('cellbin.bmp')
#im=skio.imread('cafe.bmp')

# Gray-scale images
im=skio.imread('Images/retina2.gif')
#im=skio.imread('Images/bat200.bmp')
#im=skio.imread('Images/bulles.bmp')
#im=gris_depuis_couleur (skio.imread('Images/cailloux.png'))
#im=gris_depuis_couleur(skio.imread('Images/cailloux2.png'))
#im=skio.imread('Images/laiton.bmp')
import os
plt.imshow(im,cmap="gray")

# viewimage(im) - Utilisable à la place de plt.imshow si Gimp est installé.
#dilatation
elmnts=['disk','diamond','square','line']
if not os.path.exists("dilatation"):
    os.mkdir("dilatation")
if not os.path.exists("erosion"):
    os.mkdir("erosion")
if not os.path.exists("opening"):
    os.mkdir("opening")
if not os.path.exists("closing"):
    os.mkdir("closing")
images=os.listdir("Images")

if not os.path.exists("question1"):
    os.mkdir("question1")

for img in images:
    plt.figure(figsize=(40, 20), dpi=120)
    cpt = 1
    for elt in elmnts:


      print(elt,img)
      for t in range(3,7,2):
        if img[-3:]=="png":
            im = gris_depuis_couleur( skio.imread('Images/'+img))
        else:
            im=skio.imread('Images/'+img)

        se = strel(elt, t)
        dil = morpho.dilation(im, se)
        plt.subplot(4, 8, cpt)
        plt.title("dilatation , element " + str(elt) + ',size:' + str(t))
        plt.imshow(dil,cmap='gray',vmin=0,vmax=255)

        er=morpho.erosion(im, se)
        plt.subplot(4, 8, cpt+8)
        plt.imshow(er, cmap='gray', vmin=0, vmax=255)
        plt.title(f"erosion , element " + str(elt) + ',size:' + str(t))
        op = morpho.opening(im, se)
        plt.subplot(4, 8, cpt+16)
        plt.imshow(op, cmap='gray', vmin=0, vmax=255)
        plt.title(f"opening , element " + str(elt) + ',size:' + str(t))
        cl = morpho.closing(im, se)
        plt.subplot(4, 8, cpt+24)
        plt.imshow(cl, cmap='gray', vmin=0, vmax=255)
        plt.title(f"closing , element " + str(elt) + ',size:' + str(t))
        cpt+=1
    plt.savefig("question1/"+img[:-4]+".png")



imt=im.copy()
N=5
for k in range(N):
    se=strel('disk',k)
    imt=morpho.closing(morpho.opening(imt,se),se)
plt.imshow(imt,cmap="gray")
plt.show()


#%% Watersheds
im=skio.imread('Images/bat200.bmp')
se=morpho.selem.disk(1)

grad=morpho.dilation(im,se)-morpho.erosion(im,se)
grad=np.int32(grad>40)*grad
plt.imshow(grad,cmap="gray")

local_mini = skf.peak_local_max(255-grad, #il n'y a pas de fonction local_min...
                            indices=False)
markers = ndi.label(local_mini)[0]
plt.imshow(local_mini,cmap="gray")

labels = morpho.watershed(grad, markers,watershed_line=True)
plt.imshow(couleurs_alea(labels))
# viewimage_color(couleurs_alea(labels)) - Utilisable si gimp est installé

# visualization of the result
segm=labels.copy()
for i in range(segm.shape[0]):
    for j in range(segm.shape[1]):
        if segm[i,j] == 0: 
            segm[i,j]=255
        else:
            segm[i,j]=0
#superimposition of segmentation contours on the original image
contourSup=np.maximum(segm,im)
plt.imshow(contourSup,cmap="gray") 


#%% reconstruction
im=skio.imread('Images/retina2.gif')
se4=strel('disk',4)
open4=morpho.opening(im,se4)
reco=morpho.reconstruction(open4,im)
plt.imshow(reco,cmap="gray")
#%% FIN  exemples TP MORPHO
