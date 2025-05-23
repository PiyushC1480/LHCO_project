import pickle
import gzip
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import matplotlib as mpl
import numpy as np
import random
import pandas as pd
import scipy 


random.seed(42)
np.random.seed(42)

bias=2e-02
npoints = 65 #npoint=(Number of pixels+1) of the image
DR= 0.4 #Sets the size of the image as (2xDR,2xDR)
treshold= 1.0 #We ask some treshold for the total pT fraction to keep the image when some constituents fall outside
ptjmin=0 #Cut on the minimum pT of the jet                  # undecided
sample_name='pflow'
std_label='bg_std'
batch_size = 32

Images_dir= "../../dataset/all_imgs_order/" #Output dir to save the image plots
signal = "../../dataset/eta_phi/tau_lj_constituents"
background = "../../dataset/eta_phi/hardqcd_lj_constituents"
lhco = "../../dataset/events_anomalydetection.h5"

N_pixels=np.power(npoints-1,2)



def loadfiles(folder_path):
    subjets = [[], [], []]  # List of lists for pT, eta, phi
    
    # Loop through all files in the folder
    all_files = 0
    for file_name in os.listdir(folder_path):
        if file_name.startswith("event") and file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            # Load data from the file
            data = pd.read_csv(file_path, sep="\t", header=0).iloc[:, 1:].values
            
            # Append data to respective lists
            subjets[0].append(np.array(data[:, 2]))  # pT
            subjets[1].append(np.array(data[:, 0]))  # eta
            subjets[2].append(np.array(data[:, 1]))  # phi
            all_files+=1
            # if(all_files >400):
            #     break
    
    return [np.array(subjets[0], dtype=object), np.array(subjets[1], dtype=object), np.array(subjets[2], dtype=object)]


def data_generator(filename, chunksize=512,total_size=1100000):

    m = 0
    while True:
        yield pd.read_hdf(filename,start=m*chunksize, stop=(m+1)*chunksize)
        m+=1
        if (m+1)*chunksize > total_size:
            m=0


def loadLHCOfiles(file_path):
    sig_subjects = [[], [], []] 
    bg_subjects = [[], [], []] 

    df= data_generator(file_path)

    num_signal = 0
    b =0
    for batch in df:
        for j in range(batch.shape[0]):
            pseudojet_0 =[]
            pseudojet_1= []
            pseudojet_2 = []
            # pseudojet = np.zeros((700,), dtype=[('pT', np.float32), ('eta', np.float32), ('phi', np.float32)])
            if(batch.iloc[j][2100]==1):
                for k in range(700):
                    if(batch.iloc[j][k*3] != 0.0):
                        pseudojet_0.append(batch.iloc[j][k*3])  # pT
                        pseudojet_1.append(batch.iloc[j][k*3+1])  # eta
                        pseudojet_2.append(batch.iloc[j][k*3+2])  # phi
                num_signal+=1
                # print(num_signal)
                

                pseudojet_0 = np.array(pseudojet_0)
                pseudojet_1 = np.array(pseudojet_1) 
                pseudojet_2 = np.array(pseudojet_2)

                sig_subjects[0].append(pseudojet_0)
                sig_subjects[1].append(pseudojet_1)
                sig_subjects[2].append(pseudojet_2)
            else:
                for k in range(700):
                    if(batch.iloc[j][k*3] != 0.0):
                        pseudojet_0.append(abs(batch.iloc[j][k*3]))  # pT
                        pseudojet_1.append(batch.iloc[j][k*3+1])  # eta
                        pseudojet_2.append(batch.iloc[j][k*3+2])  # phi
                pseudojet_0 = np.array(pseudojet_0)
                pseudojet_1 = np.array(pseudojet_1)
                pseudojet_2 = np.array(pseudojet_2)

                bg_subjects[0].append(pseudojet_0)
                bg_subjects[1].append(pseudojet_1)
                bg_subjects[2].append(pseudojet_2)

            #     pseudojet[k] = (batch.iloc[j][k*3], batch.iloc[j][k*3+1], batch.iloc[j][k*3+2])
            # weight = pseudojet[:]['pT']/np.max(pseudojet[:]['pT'])
            # generate_jet_image(pseudojet, weight, j)
            # if(batch.iloc[j][2100] == 0):
            #     for k in range(700):
                    # bg_pseudojet_wt_list.append( (batch.iloc[j][k*3], batch.iloc[j][k*3+1]-np.pi, batch.iloc[j][k*3+2])) 
            # else:
            #     for k in range(700):
            #     pseudojet.append( (batch.iloc[j][k*3], batch.iloc[j][k*3+1]-np.pi, batch.iloc[j][k*3+2])) 
        # generate_combined_jet_image([pseudojet for pseudojet, _ in pseudojet_wt_list], [weight for _, weight in pseudojet_wt_list])
        # bg_pseudojet_wt_list = np.vstack(bg_pseudojet_wt_list)
        # preprocess_and_plot(bg_pseudojet_wt_list)
        # generate_combined_jet_image([pseudojet for pseudojet, _ in sig_pseudojet_wt_list], [weight for _, weight in sig_pseudojet_wt_list], name = "sig")
            # if(num_signal > 2) : 
            #     break
        b+=1
        if(b==10):
            break

    return [np.array(sig_subjects[0], dtype=object), np.array(sig_subjects[1], dtype=object), np.array(sig_subjects[2], dtype=object)], [np.array(bg_subjects[0], dtype=object), np.array(bg_subjects[1], dtype=object), np.array(bg_subjects[2], dtype=object)]
    # return sig_subjects, bg_subjects



def deltaphi(phi1,phi2):
   deltaphilist=[phi1-phi2,phi1-phi2+np.pi*2.,phi1-phi2-np.pi*2.]
   sortind=np.argsort(np.abs(deltaphilist))
   return deltaphilist[sortind[0]]

def center(Subjets):
    # print('Calculating the image center for the total pT weighted centroid pixel is at (eta,phi)=(0,0) ...')
    print('-----------'*10)
    #print('subjet type {}'.format(type(subjets[0][0])))

    Njets=len(Subjets[0])
    pTj=[]
    for ijet in range(0,Njets):  
        pTj.append(np.sum(Subjets[0][ijet]))
    #print('Sum of pTj for subjets = {}'.format(pTj))
    #print('pTj ={}'.format(jets[0][0])) #This is different for Sum of pTj for subjets, as for the jets, we first sum the 4-momentum vectors of the subjets and then get the pT
    #print('subjet 1 size {}'.format(subjets[1][0]))

    eta_c=[]
    phi_c=[]
    weigh_eta=[]
    weigh_phi=[]
    for ijet in range(0,Njets):
        weigh_eta.append([ ])
        weigh_phi.append([ ])
        for isubjet in range(0,len(Subjets[0][ijet])):
            weigh_eta[ijet].append(Subjets[0][ijet][isubjet]*Subjets[1][ijet][isubjet]/pTj[ijet]) #We multiply pT by eta of each subjet
            # print('weighted eta ={}'.format(weigh_eta))  
            weigh_phi[ijet].append(Subjets[0][ijet][isubjet]*deltaphi(Subjets[2][ijet][isubjet],Subjets[2][ijet][0])/pTj[ijet]) #We multiply pT by phi of each subjet
        eta_c.append(np.sum(weigh_eta[ijet])) #Centroid value for eta
        phi_c.append(np.sum(weigh_phi[ijet])+Subjets[2][ijet][0]) #Centroid value for phi
    #print('weighted eta ={}'.format(weigh_eta))
    #print('Position of pT weighted centroid pixel in eta for [jet1,jet2,...] ={}'.format(eta_c))
    #print('Position of pT weighted centroid pixel in phi for [jet1,jet2,...]  ={}'.format(phi_c))
    #print('-----------'*10)
    return pTj, eta_c, phi_c

def shift(Subjets,Eta_c,Phi_c):
    # print('Shifting the coordinates of each particle so that the jet is centered at the origin in (eta,phi) in the new coordinates ...')
    # print('-----------'*10)

    Njets=len(Subjets[1])
    for ijet in range(0,Njets):
        # if ijet == 0:
            # print("center",Eta_c[ijet],Phi_c[ijet])
        Subjets[1][ijet]=(Subjets[1][ijet]-Eta_c[ijet])
        Subjets[2][ijet]=(Subjets[2][ijet]-Phi_c[ijet])
        Subjets[2][ijet]=np.unwrap(Subjets[2][ijet])#We fix the angle phi to be between (-Pi,Pi]
    #print('Shifted eta = {}'.format(Subjets[1]))
    #print('Shifted phi = {}'.format(Subjets[2]))
    #print('-----------'*10)
    return Subjets

def principal_axis(Subjets):
    # print('Getting DeltaR for each subjet in the shifted coordinates and the angle theta of the principal axis ...')
    # print('-----------'*10) 
    tan_theta=[]#List of the tan(theta) angle to rotate to the principal axis in each jet image
    Njets=len(Subjets[1])
    for ijet in range(0,Njets):
        M11=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[2][ijet])
        M20=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[1][ijet])
        M02=np.sum(Subjets[0][ijet]*Subjets[2][ijet]*Subjets[2][ijet])
        tan_theta_use=2*M11/(M20-M02+np.sqrt(4*M11*M11+(M20-M02)*(M20-M02)))
        tan_theta.append(tan_theta_use)

        # if ijet == 0:
        #     print("principal axis",tan_theta)
    #   print('tan(theta)= {}'.format(tan_theta))
    #   print('-----------'*10)
    return tan_theta



def rotate(Subjets,tan_theta):
    # print('Rotating the coordinate system so that the principal axis is the same direction (+ eta) for all jets ...')
    # print('-----------'*10)
    #   print(Subjets[2][0])
    #   print('Shifted eta for jet 1= {}'.format(Subjets[1][0]))
    #   print('Shifted phi for jet 1 = {}'.format(Subjets[2][0]))
    #   print('-----------'*10)
    rot_subjet=[[],[],[]]
    Njets=len(Subjets[1])
    for ijet in range(0,Njets):
        rot_subjet[0].append(Subjets[0][ijet]) 
        rot_subjet[1].append(Subjets[1][ijet]*np.cos(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.sin(np.arctan(tan_theta[ijet])))
        rot_subjet[2].append(-Subjets[1][ijet]*np.sin(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.cos(np.arctan(tan_theta[ijet])))
        #print('Rotated phi for jet 1 before fixing -pi<theta<pi = {}'.format(Subjets[2][0]))    
        rot_subjet[2][ijet]=np.unwrap(rot_subjet[2][ijet]) #We fix the angle phi to be between (-Pi,Pi]
    #   print('Subjets pT (before rotation) = {}'.format(Subjets[0]))
    #   print('-----------'*10)
    #   print('Subjets pT (after rotation) = {}'.format(rot_subjet[0]))
    #   print('-----------'*10)
    #   print('eta = {}'.format(Subjets[1]))
    #   print('-----------'*10)
    #   print('Rotated eta = {}'.format(rot_subjet[1]))
    #   print('-----------'*10)
    #   print('Rotated phi = {}'.format(Subjets[2]))
    #   print('-----------'*10)
    #   print('Rotated phi = {}'.format(rot_subjet[2]))
    #   print('-----------'*10)
    #   print('-----------'*10)
    return rot_subjet


def normalize(Subjets,pTj):
    # print('Scaling the pixel intensities such that sum_{i,j} I_{i,j}=1 ...')
    # print('-----------'*10)
    Njets=len(Subjets[0])
    #   print('pT jet 2= {}'.format(Subjets[0][1])) 
    for ijet in range(0,Njets):
        Subjets[0][ijet]=Subjets[0][ijet]/pTj[ijet]
        # Subjets[0][ijet]=Subjets[0][ijet]
    #   print('Normalizes pT jet 2= {}'.format(Subjets[0][1]))  
    #   print('Sum of normalized pT for jet 2 = {}'.format(np.sum(Subjets[0][1])))
    # print('-----------'*10)
    return Subjets


#10)Reflect the image with respect to the vertical axis to ensure the 3rd maximum is on the right half-plane
def flip(Image,Nimages): 
    count=0
    #   print('Number of rows = ',np.shape(Image[0][0])[0])
    half_img=np.int_((npoints-1)/2)
    flip_image=[]
    for i_image in range(len(Image)):
        left_img=[] 
        right_img=[]
        for i_row in range(np.shape(Image[i_image])[0]):
            left_img.append(Image[i_image][i_row][0:half_img])
            right_img.append(Image[i_image][i_row][-half_img:])
    #       print('-half_img = ',-half_img)
    #     print('Left half of image (we suppose the number of pixels is odd and we do not include the central pixel)\n',np.array(left_img))
    #     print('Right half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(right_img))
        
        left_sum=np.sum(left_img)
        right_sum=np.sum(right_img)
    #     print('Left sum = ',left_sum)
    #     print('Right sum = ',right_sum)
        
        if left_sum>right_sum:
            flip_image.append(np.fliplr(Image[i_image]))     
        else:
            flip_image.append(Image[i_image])
    #       print('Image not flipped')
    #       print('Left sum = ',left_sum)
    #       print('Right sum = ',right_sum)
            count+=1
    #     print('Array of images before flipping =\n {}'.format(Image[i_image])) 
    #     print('Array of images after flipping =\n {}'.format(flip_image[i_image])) 
    return flip_image  
 
 
##---------------------------------------------------------------------------------------------
#11)Reflect the image with respect to the horizontal axis to ensure the 3rd maximum is on the top half-plane
def hor_flip(Image,Nimages): 
    count=0
    half_img=np.int_((npoints-1)/2)
    hor_flip_image=[]
    for i_image in range(len(Image)):
        top_img=[] 
        bottom_img=[]
    #     print('image',Image[i_image])
    #     print('image',Image[i_image][0])
        for i_row in range(half_img):
    #       for i_col in range(np.shape(Image[i_image][0])[1]):
            top_img.append(Image[i_image][i_row])
            bottom_img.append(Image[i_image][-i_row-1])
    #         print('-i_row-1 = ',-i_row-1)
    #     print('Top half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(top_img))
    #     print('Bottom half of image (we suppose the number of pixels is odd and we do not include the central pixel) \n',np.array(bottom_img))
        top_sum=np.sum(top_img)
        bottom_sum=np.sum(bottom_img)
    #     print('Top sum = ',top_sum)
    #     print('Bottom sum = ',bottom_sum)
    #     
        if bottom_sum>top_sum:
            hor_flip_image.append(np.flip(Image[i_image],axis=0))     
        else:
            hor_flip_image.append(Image[i_image])
    #       print('Image not flipped')

            count+=1
    #     print('Array of images before flipping =\n {}'.format(Image[i_image])) 
    #     print('Array of images after flipping =\n {}'.format(hor_flip_image[i_image])) 
    return hor_flip_image  

def get_std(Image): 
    Image_row=[]
    #   N_pixels=np.power(npoints-1,2)
    #   Image[0].reshape((N_pixels))
    for i_image in range(len(Image)):
    #     Image_row.append([])
    #     print('i_image ={}'.format(i_image))
        Image_row.append(Image[i_image].reshape((N_pixels)))
    #   print('Image arrays as rows (1st 2 images)=\n {}'.format(Image_row[0:2]))

    Image_row=np.array(Image_row,dtype=np.float64)
    Image_row.reshape((len(Image),N_pixels))
    #   print('All image arrays as rows of samples and columns of features (pixels) (for the 1st 2 images) =\n {}'.format(Image_row[0:2]))

    #   standard_img=preprocessing.scale(Image_row)

    #     kurtosis=scipy.stats.kurtosis(Image_row,axis=0, fisher=False)
    n_moment=scipy.stats.moment(Image_row, moment=4, axis=0)
    final_bias=np.power(n_moment,1/4)+bias
    #     final_bias=n_moment/np.power(standard_dev,2)+bias
    #     print('N moment/std with bias for =\n {}'.format(final_bias[0:40]))
    #    standard_img=Image_row/final_bias
    #    print('-----------'*10)
    #    print('N moment images with bias (1st 2 image arrays as rows)=\n {}'.format(standard_img[0:2]))
    
    # elif method=='std':  
    final_bias=final_bias.reshape((npoints-1,npoints-1))
    #   print('Standard deviation with bias for =\n {}'.format(final_bias))
    
    return final_bias

def standardize_bias_std_other_set(Image, input_std_bias): 
  std_im_list=[]
  for i_image in range(len(Image)):
    std_im_list.append(Image[i_image]/input_std_bias)
    std_im_list[i_image]=std_im_list[i_image].reshape((npoints-1,npoints-1))
  return std_im_list

def standardize_images(images,reference_images):

    # print('CALCULATING STANDARD DEVIATIONS OF REFERENCE SET')
    out_std_bias=get_std(reference_images)

    # CALCULATE AVERAGE IMAGE OF REFERENCE SET
    # print('CALCULATING AVERAGE IMAGE OF REFERENCE SET')
    out_avg_image=add_images(reference_images)

    # ZERO CENTER
    # print('ZERO CENTERING IMAGES')
    #  image_zero=zero_center(images,out_avg_image)
    image_zero=images

    # DIVIDE BY STANDARD DEVIATION
    # print('STANDARDIZING IMAGES')
    standard_image=standardize_bias_std_other_set(image_zero,out_std_bias)

    return standard_image


def create_image(Subjets):
    # print('Generating images of the jet pT ...')
    # print('-----------'*10)
    etamin, etamax = -DR, DR # Eta range for the image
    phimin, phimax = -DR, DR # Phi range for the image
    eta_i = np.linspace(etamin, etamax, npoints) #create an array with npoints elements between min and max
    phi_i = np.linspace(phimin, phimax, npoints)
    image=[]
    Njets=len(Subjets[0])
    print(Njets)
    for ijet in range(0,Njets):
        grid=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image 
    #     print('Grid= {}'.format(grid))
    #     print('eta_i= {}'.format(eta_i))
        
        eta_idx = np.searchsorted(eta_i,Subjets[1][ijet]) # np.searchsorted finds the index where each value in my data (Subjets[1] for the eta values) would fit into the sorted array eta_i (the x value of the grid).
        phi_idx = np.searchsorted(phi_i,Subjets[2][ijet])# np.searchsorted finds the index where each value in my data (Subjets[2] for the phi values) would fit into the sorted array phi_i (the y value of the grid).
    
    #     print('Index eta_idx for jet {} where each eta value of the jet constituents in the data fits into the sorted array eta_i = \n {}'.format(ijet,eta_idx))
    #     print('Index phi_idx for jet {} where each phi value of the jet constituents in the data fits into the sorted array phi_i = \n {}'.format(ijet,phi_idx))
    #     print('-----------'*10)
        
    #     print('Grid for jet {} before adding the jet constituents pT \n {}'.format(ijet,grid))
        for pos in range(0,len(eta_idx)):
            if eta_idx[pos]!=0 and phi_idx[pos]!=0 and eta_idx[pos]<npoints and phi_idx[pos]<npoints: #If any of these conditions are not true, then that jet constituent is not included in the image. 
                grid[eta_idx[pos]-1,phi_idx[pos]-1]=grid[eta_idx[pos]-1,phi_idx[pos]-1]+Subjets[0][ijet][pos] #We add each subjet pT value to the right entry in the grid to create the image. As the values of (eta,phi) should be within the interval (eta_i,phi_i) of the image, the minimum eta_idx,phi_idx=(1,1) to be within the image. However, this value should be added to the pixel (0,0) in the grid. That's why we subtract 1.         
    #     print('Grid for jet {} after adding the jet constituents pT \n {}'.format(ijet,grid))  
    #     print('-----------'*10)
        
        sum=np.sum(grid)
        # print('Sum of all elements of the grid for jet {} = {} '.format(ijet,sum))
    #     print('-----------'*10)
    #     print('-----------'*10)
        
        #We ask some treshold for the total pT fraction to keep the image when some constituents fall outside of the range for (eta,phi)
        
        # if sum>=treshold:
    # and ptjmin<Jets[0][ijet]<ptjmax and jetMass_min<Jets[3][ijet]<jetMass_max:
    #       print('Jet Mass={}'.format(Jets[3][ijet]))
        image.append(grid)
        if len(grid)==0:
            print(ijet)
        if ijet%10000==0:
            print('Already generated jet images for {} jets'.format(ijet))
    #   print('Array of images before deleting empty lists = \n {}'.format(image)) 
    #   print('-----------'*10)
    #  image=[array for array in image if array!=[]] #We delete the empty arrays that come from images that don't satisfy the treshold
    
    #   print('Array of images = \n {}'.format(image[0:2])) 
    #   print('-----------'*10)
    print('Number of images= {}'.format(len(image)))
    # print('-----------'*10)
    N_images=len(image)
    
    Number_jets=N_images   #np.min([N_images, myN_jets])
    final_image=image
    #  final_image=image[0:Number_jets]
    print('N_images = ',N_images)
    print('Final images = ',len(final_image))
    
    return final_image, Number_jets


def add_images(Image):
    # print('Adding the images to get the average jet image for all the events ...')
    # print('-----------'*10)
    N_images=len(Image)
    #   print('Number of images= {}'.format(N_images))
    #   print('-----------'*10)
    avg_im=np.zeros((npoints-1,npoints-1)) #create an array of zeros for the image
    for ijet in range(0,len(Image)):
        avg_im=avg_im+Image[ijet]
        #avg_im2=np.sum(Image[ijet])
    #   print('Average image = \n {}'.format(avg_im))
    # print('-----------'*10)
    #  print('Average image 2 = \n {}'.format(avg_im2))
    #We normalize the image
    Total_int=np.absolute(np.sum(avg_im))
    # print('Total intensity of average image = \n {}'.format(Total_int))
    # print('-----------'*10)
    #  norm_im=avg_im/Total_int
    norm_im=avg_im/N_images
    #   print('Normalized average image (by number of images) = \n {}'.format(norm_im))
    #   print('Normalized average image = \n {}'.format(norm_im))
    # print('-----------'*10)
    norm_int=np.sum(norm_im)
    # print('Total intensity of average image after normalizing (should be 1) = \n {}'.format(norm_int))
    return norm_im


def plot_avg_image(Image, type,name,Nimages):
    # print('Plotting the averaged image ...')
    # print('-----------'*10)
    #   imgplot = plt.imshow(Image[0], 'viridis')# , origin='upper', interpolation='none', vmin=0, vmax=0.5)  
    imgplot = plt.imshow(Image, 'gnuplot',extent=[-DR, DR,-DR, DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)
    #   imgplot = plt.imshow(Image[0])
    #   plt.show()
    # plt.xlabel('$\eta^{\prime\prime}$')
    # plt.ylabel('$\phi^{\prime\prime}$')
    fig = plt.gcf()
    image_name=str(name)+'_avg_im_'+str(Nimages)+'_'+str(npoints-1)+'_'+type+'_'+sample_name+'.png'
    plt.savefig(Images_dir+image_name)
    print('Average image filename = {}'.format(Images_dir+image_name))

def plot_my_image(images,std_name,type):

    Nimages=len(images)

    average_im =add_images(images)
    plot_avg_image(average_im,type,std_name,Nimages)  
    # plot_avg_image(average_im,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax)+'_'+myMethod,std_name,Nimages)

def plot_all_images(Image, name):
  

#   random.shuffle(Image)

  train_qty = 0.6*len(Image)
  val_qty = 0.2*len(Image) + train_qty
  
  for ijet in range(0,len(Image)):

    imgplot = plt.imshow(Image[ijet], 'gnuplot', extent=[-DR, DR,-DR, DR])# , origin='upper', interpolation='none', vmin=0, vmax=0.5)

    # plt.xlabel('$\eta^{\prime\prime}$')
    # plt.ylabel('$\phi^{\prime\prime}$')
    plt.axis('off')
  #plt.show()
    fig = plt.gcf()

    if(ijet < train_qty):
        plt.savefig(Images_dir + "train/" +name +'/'+str(ijet)+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    elif(ijet >= train_qty and ijet < val_qty):
        plt.savefig(Images_dir + "val/" + name +'/'+str(int(ijet-train_qty))+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    else:
        plt.savefig(Images_dir + "test/" + name +'/'+str(int(ijet -(val_qty)))+'.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
#   print(len(Image))
#   print(type(Image[0]))


def preprocess(subjets,std_name):
    pTj, eta_c, phi_c=center(subjets)  
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    shift_subjets=shift(subjets,eta_c,phi_c)
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    # #print(shift_subjets)
    tan_theta=principal_axis(shift_subjets) 
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    rot_subjets=rotate(shift_subjets,tan_theta)
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    norm_subjets=normalize(rot_subjets,pTj)
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    # print('Generating raw images.. .')
    raw_image, Nimages=create_image(norm_subjets)  
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    # ver_flipped_img=flip(raw_image,Nimages)  
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    # hor_flipped_img=hor_flip(ver_flipped_img,Nimages)  
    # elapsed=time.time()-start_time
    # print('elapsed time',elapsed)

    # plot_my_image(raw_image,std_name,'_rot'+'_'+str(ptjmin))
    plot_all_images(raw_image, std_name)
    # # plot_my_image(hor_flipped_img,std_name,'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax))

    # #   hor_flipped_img=raw_image
    return raw_image


def output(images,std_name):

  Nimages=len(images)

  average_im =add_images(images)  
#   output_image_array_data_true_value(images,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+str(ptjmax)+'_'+myMethod,std_name)   
  plot_avg_image(average_im,str(std_label)+'_bias'+str(bias)+'_vflip_hflip_rot'+'_'+str(ptjmin)+'_'+'nmoment',std_name,Nimages)

if __name__=='__main__':
    print('LOADING FILES...')

    sig_subjets= loadfiles(signal) 
    sig_subjets = np.array(sig_subjets)
    print((sig_subjets.shape))
    sig_images=preprocess(sig_subjets,'signal')

    bg_subjets = loadfiles(background)
    bg_subjets = np.array(bg_subjets)
    print((bg_subjets.shape))
    bg_images=preprocess(bg_subjets,'bg') 

    # print(sig_subjets.nbytes / (1024*1024))
    # print(bg_subjets.nbytes / (1024*1024))


    # sig_subjets, bg_subjets = loadLHCOfiles(lhco)


    # sig_image_norm = standardize_images(sig_images,bg_images)
    # bg_image_norm = standardize_images(bg_images,bg_images)
    # output(sig_image_norm,'tau')
    # output(bg_image_norm,'QCD')