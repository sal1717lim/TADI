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
"""
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
"""
#preuve des propriete
#Dilatation
#1
img=skio.imread('Images/cellbin.bmp')
dilatation=gris_depuis_couleur(skio.imread('dilatation/cellbin_disk_5.png'))
x=np.zeros((dilatation.shape[0],dilatation.shape[1],3))
for i in range(dilatation.shape[0]):
    for j in range(dilatation.shape[1]):
        if dilatation[i,j]==255:
            if img[i,j]==255:
                x[i,j]=(255,255,255)
            else:
                x[i,j]=(255,0,0)

plt.subplot(1,3,1)
plt.imshow(img,cmap="gray")
plt.title("original")
plt.subplot(1,3,2)
plt.imshow(dilatation,cmap="gray")
plt.title("dilatation")
plt.subplot(1,3,3)
plt.imshow(x)
plt.title("inclusion")
plt.savefig("extensive.png")

x=dilatation-img
print("valeur de dilatation-image positive:",(x>=0).any())

#2
X=skio.imread("Question2/cellbin.bmp")
Y=skio.imread("Question2/cellbin2.bmp")
se = strel("disk", 3)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(X,cmap='gray')
plt.title("X")
plt.subplot(2,2,2)
plt.imshow(morpho.dilation(X,se),cmap='gray')
plt.title("dilatation(X,B)")
plt.subplot(2,2,3)
plt.imshow(Y,cmap='gray')
plt.title("Y")
plt.subplot(2,2,4)
plt.imshow(morpho.dilation(Y,se),cmap='gray')
plt.title("dilatation(Y,B)")
plt.savefig("question2.png")

x=morpho.dilation(Y,se)-morpho.dilation(X,se)
print("valeur de morpho.dilation(Y,se)-morpho.dilation(X,se) positive:",(x>=0).any())

#3
se = strel("disk", 3)
se2 = strel("disk", 5)

X1=morpho.dilation(Y,se)
X2=morpho.dilation(Y,se2)
res=np.zeros((X1.shape[0],X1.shape[1],3))
for i in range(X1.shape[0]):
    for j in range(X.shape[1]):
        if X2[i,j]==255:
            if X1[i,j]==255:
                res[i,j]=(255,255,255)
            else:
                res[i,j]=(255,0,0)
plt.subplot(1,4,1)
plt.imshow(Y,cmap='gray')
plt.title("X")
plt.subplot(1,4,2)
plt.imshow(X1,cmap='gray')
plt.title("D(X,B)")
plt.subplot(1,4,3)
plt.imshow(X2,cmap='gray')
plt.title("D(X,B')")
plt.subplot(1,4,4)
plt.imshow(res,cmap='gray')
plt.title("inclusion")
plt.savefig("question3.png")
print("\n\n\n")

x=X2-X1
print("valeur de morpho.dilation(X,se)-morpho.dilation(X,se2) positive:",(x>=0).any())

print("\n\n\n")


#4
X=skio.imread("Question2/cellbin.bmp")
U=skio.imread("Question2/cellbin2.bmp")
Y=U-X
plt.figure()
plt.subplot(1,3,1)
plt.imshow(X,cmap="gray")
plt.title("X")
plt.subplot(1,3,2)
plt.imshow(Y,cmap="gray")
plt.title("Y")
plt.subplot(1,3,3)
plt.imshow(U,cmap="gray")
plt.title("X U Y")
plt.savefig("input_question4.png")
imt=im.copy()
plt.figure()
plt.subplot(1,4,1)
plt.imshow(morpho.dilation(X,se),cmap="gray")
plt.title("D(X,B)")
plt.subplot(1,4,2)
plt.imshow(morpho.dilation(Y,se),cmap="gray")
plt.title("D(Y,B)")
plt.subplot(1,4,3)
plt.imshow(morpho.dilation(X,se)+morpho.dilation(Y,se),cmap="gray")
plt.title("D(X,B)UD(Y,B)")
plt.subplot(1,4,4)
plt.imshow(morpho.dilation(U,se),cmap="gray")
plt.title("D(XUY,B)")
plt.savefig("question4.png")

print("test d'egalité entre les deux images:",np.equal(morpho.dilation(U,se),morpho.dilation(X,se)+morpho.dilation(Y,se)).any())

X=skio.imread("Question2/cellbin.bmp")
Y=skio.imread("Question2/cellbin3.bmp")
inter=np.zeros(Y.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
        if X[i,j]==Y[i,j]:
          if X[i,j]!=0:
            inter[i,j]=255
z1=morpho.dilation(X,se)
z2=morpho.dilation(Y,se)
inter2=np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
        if z1[i,j]==z2[i,j]:
          if z1[i,j]!=0:
            inter2[i,j]=255
plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(z1,cmap="gray")
plt.title("D(X,B)")
plt.subplot(1,4,2)
plt.imshow(z2,cmap="gray")
plt.title("D(Y,B)")
plt.subplot(1,4,3)
plt.imshow(morpho.dilation(inter,se),cmap="gray")
plt.title("D(X inter Y,B)")
plt.subplot(1,4,4)
plt.imshow(inter2,cmap="gray")
plt.title("D(X,B) inter D(Y,B)")
plt.savefig("input_question42res.png")

print("\n\n\n\ntest inclusion:",((inter2-morpho.dilation(inter,se)>=0)).any())

X=skio.imread("Images/cellbin.bmp")
se=strel("square",3)
se2=strel("square",5)
se3=strel("square",7)
plt.subplot(1,2,1)
plt.imshow(morpho.dilation(morpho.dilation(X,se),se2),cmap="gray")
plt.title("D(D(X,B'),B)")
plt.subplot(1,2,2)
plt.imshow(morpho.dilation(X,se3),cmap="gray")
plt.title("D(X,B+B')")

print("D(X,B+B')=D(D(X,B'),B):",(morpho.dilation(morpho.dilation(X,se),se2)==morpho.dilation(X,se3)).any())

plt.savefig("question5.png")
#erosion

X=skio.imread("Images/laiton.bmp")
comp=255-X
plt.subplot(2,4,1)
plt.imshow(X,cmap="gray",vmin=0,vmax=255)
plt.title("X")
plt.subplot(2,4,2)
plt.imshow(comp,cmap="gray",vmin=0,vmax=255)
plt.title("255-X")
plt.subplot(2,4,3)
plt.imshow(morpho.dilation(comp,se),cmap="gray",vmin=0,vmax=255)
plt.title("D(255-X,B)")
plt.subplot(2,4,4)
plt.imshow(255-morpho.dilation(comp,se),cmap="gray",vmin=0,vmax=255)
plt.title("255-D(255-X,B)")
plt.subplot(2,4,5)
plt.imshow(X,cmap="gray")
plt.title("X")
plt.subplot(2,4,6)
plt.imshow(morpho.erosion(X,se),cmap="gray",vmin=0,vmax=255)
plt.title("E(X,B)")
plt.savefig("erosionquestion1.png")

##2
Y=morpho.erosion(X,se)
plt.subplot(1,2,1)
plt.imshow(Y,cmap="gray",vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.imshow(X,cmap="gray",vmin=0,vmax=255)
plt.savefig("question2ero.png")

print("\n\nanti-extensivité: ",((X-Y)>=0).any())
#3
X=skio.imread("Question2/cellbin.bmp")
Y=skio.imread("Question2/cellbin2.bmp")

se = strel("disk", 3)
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(X,cmap='gray')
plt.title("X")
plt.subplot(2,2,2)
plt.imshow(morpho.erosion(X,se),cmap='gray')
plt.title("E(X,B)")
plt.subplot(2,2,3)
plt.imshow(Y,cmap='gray')
plt.title("Y")
plt.subplot(2,2,4)
plt.imshow(morpho.erosion(Y,se),cmap='gray')
plt.title("E(Y,B)")
plt.savefig("question2Ero.png")


x=morpho.erosion(Y,se)-morpho.erosion(X,se)
print("valeur de E(Y,se)-E(X,se) positive:",(x>=0).any())

#3
se = strel("disk", 3)
se2 = strel("disk", 5)

X1=morpho.erosion(Y,se)
X2=morpho.erosion(Y,se2)
res=np.zeros((X1.shape[0],X1.shape[1],3))
for i in range(X1.shape[0]):
    for j in range(X.shape[1]):
        if X1[i,j]==255:
            if X2[i,j]==255:
                res[i,j]=(255,255,255)
            else:
                res[i,j]=(255,0,0)
plt.subplot(1,4,1)
plt.imshow(Y,cmap='gray')
plt.title("X")
plt.subplot(1,4,2)
plt.imshow(X1,cmap='gray')
plt.title("E(X,B)")
plt.subplot(1,4,3)
plt.imshow(X2,cmap='gray')
plt.title("E(X,B')")
plt.subplot(1,4,4)
plt.imshow(res,cmap='gray')
plt.title("inclusion")
plt.savefig("question3ero.png")
print("\n\n\n")

x=X1-X2
print("valeur de morpho.erosion(X,B)-morpho.erosion(X,B') positive:",(x>=0).any())

#5
X=skio.imread("Question2/cellbin.bmp")
Y=skio.imread("Question2/cellbin3.bmp")
inter=np.zeros(Y.shape)
for i in range(X.shape[0]):
    for j in range(Y.shape[1]):
        if X[i,j]==Y[i,j]:
          if X[i,j]!=0:
            inter[i,j]=255
z1=morpho.erosion(X,se)
z2=morpho.erosion(Y,se)
un=X+Y
UN2=z1+z2
plt.figure(figsize=(10,5))
plt.subplot(1,4,1)
plt.imshow(z1,cmap="gray")
plt.title("E(X,B)")
plt.subplot(1,4,2)
plt.imshow(z2,cmap="gray")
plt.title("E(Y,B)")
plt.subplot(1,4,3)
plt.imshow(morpho.erosion(un,se),cmap="gray")
plt.title("E(X U Y,B)")
plt.subplot(1,4,4)
plt.imshow(UN2,cmap="gray")
plt.title("E(X,B) U E(Y,B)")
plt.savefig("input_question4UNIOres.png")


print("\n\n\n\ntest d'inclusion:",((morpho.erosion(un,se)-UN2)>=0).any())

X=skio.imread("Images/cellbin.bmp")
se=strel("square",3)
se2=strel("square",5)
se3=strel("square",7)
plt.subplot(1,2,1)
plt.imshow(morpho.erosion(morpho.erosion(X,se),se2),cmap="gray")
plt.title("E(E(X,B'),B)")
plt.subplot(1,2,2)
plt.imshow(morpho.erosion(X,se3),cmap="gray")
plt.title("E(X,B+B')")

print("E(X,B+B')=E(E(X,B'),B):",(morpho.erosion(morpho.erosion(X,se),se2)==morpho.erosion(X,se3)).any())

plt.savefig("question5ero.png")
exit(0)
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
