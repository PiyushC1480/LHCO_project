
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
import random 

random.seed(42)
np.random.seed(42)

mpl.interactive(False)



#each row in the dataset corresponds to a single event. For each event, all reconstructed particles are assumed to be massless and are recorded in detector coordinates (pT, eta, phi). Events are zero padded to constant size arrays of 700 particles. 


# i want you to make a eta, phi plot for each event.
# for each event you have atmost 700 particles with their eta and phi values.
#plot the pt values of the particles in the eta, phi plane for each event.
#last column is the event label. 1 for signal and 0 for background.


def data_generator(filename, chunksize=512,total_size=1100000):

    m = 0
    while True:
        yield pd.read_hdf(filename,start=m*chunksize, stop=(m+1)*chunksize)
        m+=1
        if (m+1)*chunksize > total_size:
            m=0


def generate_jet_image(pseudojet, weight, idx, pixel_size=200, image_size=200, eta_min=-5., eta_max=5.):
    """
    Function to generate and save pixelated jet images from pseudojets.

    Args:
        pseudojet: Unclustered pseudojet with 'eta' and 'phi' attributes.
        weight: Weights for the histogram to colorize pixels.
        idx: Event index (used for naming the saved image).
        pixel_size: Number of pixels in each dimension of the image.
        image_size: Dimension of the generated image.
        eta_min: Minimum pseudo-rapidity.
        eta_max: Maximum pseudo-rapidity.
        path: Path to save the image.
    """

    # Define edges and bins for pixelation
    eta_edges = np.linspace(eta_min, eta_max, pixel_size + 1)
    phi_edges = np.linspace(-np.pi, np.pi, pixel_size + 1)

    phi_min, phi_max = -np.pi, np.pi
    extent = (eta_min, eta_max, phi_min, phi_max)

    # Create 2D histogram
    hadrons, _, _ = np.histogram2d(pseudojet[:]['eta'], pseudojet[:]['phi'], 
                                   bins=(eta_edges, phi_edges), weights=weight)

    # Plot the image
    fig, ax = plt.subplots(subplot_kw=dict(xlim=(eta_min, eta_max),
                                           ylim=(phi_min, phi_max)),
                           figsize=(image_size/10, image_size/10), dpi=10)
    ax.axis('off')

    ax.imshow(np.ma.masked_where(hadrons == 0, hadrons).T,
              extent=extent, aspect=(eta_max - eta_min) / (2*np.pi),
              interpolation='none', origin='lower')

    # Save the image
    plt.tight_layout()
    plt.savefig(f"../output/event_{idx}.png", dpi=10)
    plt.clf()
    plt.close()


def generate_combined_jet_image(pseudojet_list, weight_list, pixel_size=100, image_size=400, eta_min=-5., eta_max=5., name = "bg"):
    """
    Function to generate a combined pixelated jet image from multiple pseudojets.

    Args:
        pseudojet_list: List of pseudojets, where each pseudojet has 'eta' and 'phi' attributes.
        weight_list: List of corresponding weights for each pseudojet.
        pixel_size: Number of pixels in each dimension of the image.
        image_size: Dimension of the generated image.
        eta_min: Minimum pseudo-rapidity.
        eta_max: Maximum pseudo-rapidity.
    """

    # Define edges and bins for pixelation
    eta_edges = np.linspace(eta_min, eta_max, pixel_size + 1)
    phi_edges = np.linspace(-np.pi, np.pi, pixel_size + 1)

    phi_min, phi_max = -np.pi, np.pi
    extent = (eta_min, eta_max, phi_min, phi_max)

    # Initialize empty histogram
    combined_hadrons = np.zeros((pixel_size, pixel_size))

    # Accumulate histograms for all pseudojets
    for pseudojet, weight in zip(pseudojet_list, weight_list):
        hadrons, _, _ = np.histogram2d(pseudojet[:]['eta'], pseudojet[:]['phi'], 
                                       bins=(eta_edges, phi_edges), weights=weight)
        combined_hadrons += hadrons  # Sum over all events

    # Plot the combined image
    fig, ax = plt.subplots(subplot_kw=dict(xlim=(eta_min, eta_max),
                                           ylim=(phi_min, phi_max)),
                           figsize=(image_size/10, image_size/10), dpi=10)
    ax.axis('off')

    ax.imshow(np.ma.masked_where(combined_hadrons == 0, combined_hadrons).T,
              extent=extent, aspect=(eta_max - eta_min) / (2*np.pi),
              interpolation='none', origin='lower')

    # Save the image
    plt.tight_layout()
    plt.savefig(f"../output/combined_jet_image_{name}.png", dpi=10)
    plt.clf()
    plt.close()


jet_radius =  2.5
def preprocess_and_plot(jet_array, grid_size=127):
    # Step 1: Translation
    # Sort particles by pT to find the leading subjet
    # jet_array[:, 2] = np.log(jet_array[:, 2])
    jet_array[:,1] = jet_array[:,1] - np.pi
    jet_array = jet_array[jet_array[:, 2].argsort()[::-1]]  # Sort by pT in descending order
    leading_subjet = jet_array[0]  # (eta, phi, pT)
    
    # Translate the jet so that the leading subjet is at (0,0)
    translated_eta = jet_array[:, 0] - leading_subjet[0]
    translated_phi = jet_array[:, 1] - leading_subjet[1]
    translated_pT = jet_array[:, 2]   # pT -> transverse energy

    #write this to a file
    # np.savetxt("../output/translated_data.txt", np.array([translated_eta, translated_phi, translated_pT]).T, delimiter="\t", header="eta\tphi\tpT", comments="")

    # # Step 2: Rotation
    # if len(jet_array) > 1:
    #     second_subjet = jet_array[1]  # Second highest pT subjet
    #     angle = -np.arctan2(second_subjet[1] - leading_subjet[1], second_subjet[0] - leading_subjet[0]) - np.pi/2
    
    # # Apply rotation
    # cos_angle = np.cos(angle)
    # sin_angle = np.sin(angle)
    # rotated_eta = cos_angle * translated_eta - sin_angle * translated_phi
    # rotated_phi = sin_angle * translated_eta + cos_angle * translated_phi
    
    # # Step 3: Re-Pixelation (Interpolation)
    grid_eta = np.linspace(-jet_radius, jet_radius, grid_size)
    grid_phi = np.linspace(-np.pi, np.pi, grid_size)
    grid_eta, grid_phi = np.meshgrid(grid_eta, grid_phi)

    
    jet_image = griddata((translated_eta, translated_phi), translated_pT, (grid_eta, grid_phi), method='cubic', fill_value=0)
    # jet_image = griddata((rotated_eta, rotated_phi), translated_pT, (grid_eta, grid_phi), method='cubic', fill_value=0)
    
    # # Step 4: Parity Flip
    # if np.sum(jet_image[:, grid_size//2:]) < np.sum(jet_image[:, :grid_size//2]):
    #     jet_image = np.fliplr(jet_image)  # Flip horizontally
    
    # Plot the jet image
    plt.figure(figsize=(6, 6))
    plt.imshow(jet_image, extent=[-jet_radius, jet_radius, -jet_radius, jet_radius], origin='lower', cmap='inferno',aspect='auto')
    plt.colorbar(label='(pT)')
    plt.xlabel('η')
    plt.ylabel('φ')
    plt.title('Jet Image in η-φ Plane')
    plt.savefig("../output/newDeepTau.png")


# def plot_event(j):
#     event = df.iloc[j]
#     for i in range(0, 700, 3):
#         # i%3 is the total momentum of the particle, i%3+1 is the eta and i%3+2 is the phi
#         #plot the value of pt at the eta, phi plane, for tow different particle with same pT, take sum of their pt values.
#         plt.scatter(event[i+1], event[i+2], s = event[i])
#     plt.xlabel("Eta")
#     plt.ylabel("Phi")
#     plt.title("Event , label = {}".format(event[2100]))
#     plt.savefig(f"../output/event_{j}.png")



if __name__  == "__main__":

    # test_file = "/Users/piyushchauhan/Desktop/btp2/local/sample_files/parsingscripts-master/events_anomalydetection_tiny.h5"
    # rnd_file = "/Users/piyushchauhan/Desktop/btp2/local/dataset/LHCO/events_anomalydetection.h5"

    rnd_file = "../dataset/events_anomalydetection.h5"

    # df = pd.read_hdf(rnd_file)
    # df = data_generator(test_file)
    df= data_generator(rnd_file)
    for batch in df:
        print(type(batch))
        print(batch.keys())
        print(batch.shape)
        bg_pseudojet_wt_list = []
        sig_pseudojet_wt_list = []
        for j in range(batch.shape[0]):
            pseudojet =[]
            # pseudojet = np.zeros((700,), dtype=[('pT', np.float32), ('eta', np.float32), ('phi', np.float32)])
            # for k in range(700):
            #     pseudojet[k] = (batch.iloc[j][k*3], batch.iloc[j][k*3+1], batch.iloc[j][k*3+2])
            # weight = pseudojet[:]['pT']/np.max(pseudojet[:]['pT'])
            # generate_jet_image(pseudojet, weight, j)
            if(batch.iloc[j][2100] == 0):
                for k in range(700):
                    bg_pseudojet_wt_list.append( (batch.iloc[j][k*3], batch.iloc[j][k*3+1]-np.pi, batch.iloc[j][k*3+2])) 
            # else:
            #     for k in range(700):
            #     pseudojet.append( (batch.iloc[j][k*3], batch.iloc[j][k*3+1]-np.pi, batch.iloc[j][k*3+2])) 
        # generate_combined_jet_image([pseudojet for pseudojet, _ in pseudojet_wt_list], [weight for _, weight in pseudojet_wt_list])
        bg_pseudojet_wt_list = np.vstack(bg_pseudojet_wt_list)
        preprocess_and_plot(bg_pseudojet_wt_list)
        # generate_combined_jet_image([pseudojet for pseudojet, _ in sig_pseudojet_wt_list], [weight for _, weight in sig_pseudojet_wt_list], name = "sig")
        break

    # for i in range(1):
    #     #create a pseudojet data sturcture of size 700*3 with 0 values.
    #     pseudojet = np.zeros((700,), dtype=[('pT', np.float32), ('eta', np.float32), ('phi', np.float32)])
    #     #fill the pseudojet data structure with the values of the particles in the event.
    #     for j in range(700):
    #         pseudojet[j] = (df.iloc[i][j*3], df.iloc[i][j*3+1], df.iloc[i][j*3+2])
    #     weight = pseudojet[:]['pT']/np.max(pseudojet[:]['pT'])
    #     generate_jet_image(pseudojet, weight, i)



