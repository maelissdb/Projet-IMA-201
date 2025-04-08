#%% bibliothèques
import cv2
import numpy as np
import skimage
from skimage import color
from matplotlib import pyplot as plt

# Parameters
N = 256  # Number of gray levels

#%% Load the image
img = cv2.imread("ISIC_0000042.jpg")

# Transform the image to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display 2 images with the same number of canals for comparison
def view2images(img1, img2):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img_conc = np.hstack((img1, img2))
    cv2.imshow('Comparison of images', img_conc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display a single image
def viewimage(img):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_norm_hist(img):
    hist = cv2.calcHist([img], [0], None, [N], [0, N])
    plt.plot(hist, color='gray')
    plt.xlabel('intensity')
    plt.ylabel('number of pixels')
    plt.show()

def prob_gray_lvl(img):
    data_level = np.zeros(N)  # Array with number of pixel corresponding to each gray level
    num_pixels = img.shape[0] * img.shape[1]  # Total number of pixels
    hist = cv2.calcHist([img], [0], None, [N], [0, N])  # Histogram
    for i in range(N):
        data_level[i] = hist[i][0]
    proba_level = data_level / num_pixels
    return proba_level

# Compute the probability of occurrence C0, going to the gray level of value t, and of C1
def proba_class_thresh(img, t):
    proba_level = prob_gray_lvl(img)
    # Threshold is the gray level that separates the two classes
    p = np.sum(proba_level[:t])
    q = np.sum(proba_level[t:])
    return p, q

def mean_class_thresh(img, t):
    mean = 0
    proba_level = prob_gray_lvl(img)
    for i in range(1, t + 1):
        mean += i * proba_level[i - 1]
    return mean

def mean_C0(img, t):
    w = proba_class_thresh(img, t)[0]
    mu = mean_class_thresh(img, t)
    return mu / w if w != 0 else 0

def mean_C1(img, t):
    mu = mean_class_thresh(img, N) - mean_class_thresh(img, t)
    w = proba_class_thresh(img, t)[1]
    return mu / w if w != 0 else 0

def var_between_class(img, t):
    w0, w1 = proba_class_thresh(img, t)
    mu0 = mean_C0(img, t)
    mu1 = mean_C1(img, t)
    return w0 * w1 * (mu0 - mu1) ** 2

# The final goal of the algorithm is to find the threshold that maximizes the between class variance
def otsu(img):
    var_max = 0
    tresh = 0
    for i in range(N):
        var = var_between_class(img, i)
        if var > var_max:
            var_max = var
            tresh = i
    return tresh

#%% Debut Zortea 
eta = 1/4
img = cv2.imread("ISIC_0000042.jpg")
L = max(img.shape[:2]) # Maximum dimension of the image
s = 0.02 * L #taille étape
if img.dtype == 'float64':
    img = img.astype('float32')

# Convertir en LAB
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# Convert the image to the CIELAB color space
#img_lab = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#idée : on va se déplacer sur l'image avec des rectanges de taille s, calculer ecart type et moyenne 
#jusqu'à eta de la distance du plus petit côté de l'image
# Select the skin region avec somme ratio ecart type/ moyenne minimal
def select_skin_region_init(img_lab):
    return img_lab[50:150, 50:150]


#img_lab = skimage.color.rgb2lab(img)
img_bgr = cv2.cvtColor(select_skin_region_init(img_lab), cv2.COLOR_Lab2BGR)  # Convertir en BGR
region_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
viewimage(region_rgb)

def choix_region(region, min_ratio, best_region):
    mean_0 = np.mean(region[:,:,0])
    mean_1 = np.mean(region[:,:,1])
    mean_2 = np.mean(region[:,:,2])
    if mean_0 == 0 or mean_1 == 0 or mean_2 == 0:
        return best_region, min_ratio  # Ne pas mettre à jour si division impossible
    
    ecart_type_0 = np.std(region[:,:,0])
    ecart_type_1 = np.std(region[:,:,1])
    ecart_type_2 = np.std(region[:,:,2])
    ratio = sum([ecart_type_0/mean_0, ecart_type_1/mean_1, ecart_type_2/mean_2])
    if ratio < min_ratio:
        min_ratio = ratio
        best_region = region
    return(best_region, min_ratio)

def select_skin_region(img_lab, s=0.02, eta=1/4):
    h,w = img_lab.shape[:2]
    s_size = int(s * max(h, w))
    center_img_x, center_img_y = w//2, h//2
    min_ratio = float('inf')
    best_region = None
#on itère sur les rectangles de taille s contenus dans les bordures de l'images à eta du plus petit côté
    for i in range(0, h - s_size + 1, s_size):
        for j in range(0, w - s_size + 1, s_size):
            start_w = i
            end_w = i + s_size
            start_h = j
            end_h = j + s_size    
    
            if end_w > h or end_h > w:
                j+=s_size
                continue
            region = img_lab[start_w:end_w, start_h:end_h]
            #region_sym = img_lab[start_h:end_h, -start_w:-end_w]
            best_region, min_ratio = choix_region(region, min_ratio, best_region)
            #best_region, min_ratio = choix_region(region_sym, min_ratio, best_region)
            j += s_size
        i += s_size
    return best_region
             
"""
def select_skin_region_2(img, s=0.02, eta=1/4):
    h, w = img.shape[:2]
    s_size = int(s * max(h, w))
    center_x, center_y = w // 2, h // 2
    min_ratio = float('inf')
    best_region = None
    
    for i in range(0, min(center_x, center_y), 5):
        x1, y1 = center_x - i, center_y - i
        x2, y2 = center_x + i, center_y + i
        region = img[y1:y2, x1:x2]
        mean = np.mean(region)
        ecart_type = np.std(region)
        ratio = ecart_type / mean
        if ratio < min_ratio:
            min_ratio = ratio
            best_region = region
    
    return best_region"""

#%% 1. Automatic selection of candidate skin pixels
print(int(0.02 * max(img.shape[:2])))

def select_skin_region_bis(img_lab, s=0.02, eta=1/4):
    h, w = img_lab.shape[:2]
    s_size = int(s * max(h, w))
    center_img_x, center_img_y = w//2, h//2
    min_ratio = float('inf')
    best_region = None
#on itère sur les rectangles de taille s contenus dans les bordures de l'images à eta du plus petit côté
    for i in range(0, h - s_size + 1, s_size):
        for j in range(0, w - s_size + 1, s_size):
            start_x = i
            end_x = i+s_size
            start_y = j
            end_y = j+s_size
            if end_x > h or end_y > w: #vérifier que la région est bien contenue dans l'image
                continue

            region = img_lab[start_x:end_x, start_y:end_y]
            mean_0 = np.mean(region[:,:,0])
            mean_1 = np.mean(region[:,:,1])
            mean_2 = np.mean(region[:,:,2])
            ecart_type_0 = np.std(region[:,:,0])
            ecart_type_1 = np.std(region[:,:,1])
            ecart_type_2 = np.std(region[:,:,2])
            ratio = sum([ecart_type_0/mean_0, ecart_type_1/mean_1, ecart_type_2/mean_2])
            print(ratio)
            if ratio < min_ratio:
                min_ratio = ratio
                best_region = region
    return best_region

# %% 2. Computation of an intensity image for thresholding
# on cherche les composentes médians des pixels de la région de peau sélectionnée 
''' We choose the CIELAB because of its relative perceptual uniformity. 
Large (small) differences between any two colors correspond approximately 
to long (short) Euclidian distances between the colors in the three-dimensional CIELAB space.'''
#img_lab = skimage.color.rgb2lab(img)
#Rs = select_skin_region_init(img_lab)
Rs = region_rgb
if Rs is not None and Rs.size > 0:
    R_median = np.median(Rs, axis=(0, 1))
    print("TRs_median", R_median)
if Rs is None:
    print("None")
else:
    print("No skin region found")
#R_median = np.median(Rs, axis=(0,1))
#### MODIFIER CA RS VIDE
plt.imshow(Rs)
plt.show()

print(R_median)
#%%calul de l'image d'intensité
def image_intensity(img_lab, R_median):
    l, a, b = cv2.split(img_lab) #img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    l_s, a_s, b_s = R_median
    intensity_image = np.sqrt((l - l_s) ** 2 + (a - a_s) ** 2 + (b - b_s) ** 2)
    return intensity_image

intensity_image = image_intensity(img_lab, R_median)
ws = 0.01 * max(img.shape[:2])
ws = int(ws)
intensity_image_med = skimage.filters.median(intensity_image, footprint=np.ones((ws, ws)))

#%% afficher la région sélectionnée
viewimage(Rs)

#%% 3.  Threshold estimation
#selectionne les pixels cross_diagonal de taille de région ws
def selection_2(image_lab):
    h, w = image_lab.shape[:2]
    ws = 0.01 * w
    ws = int(ws)
    center_x, center_y = w // 2, h // 2
    cross_diagonal = []
    i = 0
    for i in range(-h,h): #on parcourt lespixels verticaux
        x, y = center_x, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w,w): #on parcourt les pixels horizontaux
        x, y = center_x + i*ws, center_y
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w, w): #on parcourt les pixels de la diagonale haut-gauche à bas-droite
        x, y = center_x + i*ws, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    for i in range(-w, w): #on parcourt les pixels de la diagonale haut-droite à bas-gauche
        x, y = center_x - i*ws, center_y + i*ws
        if 0 <= x < w and 0 <= y < h:
            cross_diagonal.append(image_lab[y, x])
    return cross_diagonal

def sigma(image,t):
    return var_between_class(image, t)

def phi(L_im, t):
    phi = []
    for i in range(len(L_im)):
        phi.append(sigma(L_im[i], t))
    #calcul of the scaled Euclidian norm:
    phi = len(L_im)*np.linalg.norm(phi)
    return phi

# calcul threshold_h 1er seuil
def threshold_h(image, R_median):
    th = sigma(intensity_image,0)
    for i in range(256):
        s1 = sigma(intensity_image, i)/phi([intensity_image, selection_2(img_lab)], i)
        s2 = sigma(selection_2(img_lab), i)/phi([intensity_image, selection_2(img_lab)], i)
        if s1 + s2 > th:    
            th = i
    return th

def threshold_s(img, beta):
    Rs = select_skin_region_bis(img, s, eta)
    nu1 = 0.05
    nu2 = 0.5
    im_i = image_intensity(Rs, R_median)
    gamma_1 = np.percentile(im_i, nu1)
    gamma_2 = np.percentile(im_i, nu2)
    return gamma_2 + beta*(gamma_2 - gamma_1)

#final threshold
def threshold_z(img, alpha, beta):
    if threshold_s(img, R_median) > threshold_h(img, beta):
        alpha = 1
    th_z = alpha*threshold_h(img, R_median) + (1-alpha)*threshold_s(img, beta)
    return th_z


# %% 4. Image segmentation
# Apply the final threshold to the intensity image
img = cv2.imread("ISIC_0000042.jpg")


img_lab = color.rgb2lab(img)
final_threshold = threshold_z(img, 0.5, 0.5)
img_seg = otsu(intensity_image)

# Display the segmented image
viewimage(img_seg)

# %%
