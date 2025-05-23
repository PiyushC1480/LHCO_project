import uproot
import torch
import torch.nn as nn
import torch.optim as optim
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


MAX_LEN = 50


#data tranform for particla net 

def deltaphi(phi1,phi2):
   deltaphilist=[phi1-phi2,phi1-phi2+np.pi*2.,phi1-phi2-np.pi*2.]
   sortind=np.argsort(np.abs(deltaphilist))
   return deltaphilist[sortind[0]]

def deltaR(eta1, phi1, eta2, phi2):
    dphi = deltaphi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def save_csv(Subjets, is_signal):
    Njets=len(Subjets[0])

    pTj=[]  #  pT(jet) 
    eTj = []  # total jet energy 

    log_pT =[]  # log(pT)    log of particle transverse momentum
    e_part = []  #each particle energy e+part = pT * cosh(eta)
    log_eT = []  # log(e_part)    log of particle energy

    ratio_log_pTj = []   # log(pT/pTj)
    ratio_log_eTj = []  # log(e_part/eTj) 
    
    dR = []   # delta_R 

    # calcuate the e_part
    for ijet in range(0,Njets):
        temp_e_jet = [] 
        for isubjet in range(0,len(Subjets[0][ijet])):
            temp_e_jet.append(Subjets[0][ijet][isubjet]*np.cosh(Subjets[1][ijet][isubjet]))
        e_part.append(temp_e_jet)

    #calculate the eTj
    for ijet in range(0,Njets):
        eTj.append(np.sum(e_part[ijet]))
    
    #calculate log(eT)
    for ijet in range(0,Njets):
        log_eT.append(np.log(e_part[ijet]))

    #calculate the ratio log(e_part/eTj)
    for ijet in range(0,Njets):
        ratio_log_eTj.append(np.log(e_part[ijet]/eTj[ijet]))



    #calculate the pTj
    for ijet in range(0,Njets):
        pTj.append(np.sum(Subjets[0][ijet]))

    #calculate log(pT)
    for ijet in range(0,Njets):
        log_pT.append(np.log(Subjets[0][ijet]))

    #calculate the ratio log(pT/pTj)
    for ijet in range(0,Njets):
        ratio_log_pTj.append(np.log(Subjets[0][ijet]/pTj[ijet]))
    


    #calculate d_eta and d_phi
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


    dEta = []
    dPhi = []

    for ijet in range(Njets):
        eta_jet = eta_c[ijet]
        phi_jet = phi_c[ijet]
        
        eta_list = []
        phi_list = []
        
        for isubjet in range(len(Subjets[0][ijet])):
            eta = Subjets[1][ijet][isubjet]
            phi = Subjets[2][ijet][isubjet]
            
            deta = eta - eta_jet
            dphi = np.arctan2(np.sin(phi - phi_jet), np.cos(phi - phi_jet))  # Proper delta phi

            eta_list.append(deta)
            phi_list.append(dphi)
 
        dEta.append(eta_list)
        dPhi.append(phi_list)

    
    #calculate dR
    for ijet in range(Njets):
        dr_jet = []
        for isubjet in range(len(Subjets[0][ijet])):
            eta_particle = Subjets[1][ijet][isubjet]
            phi_particle = Subjets[2][ijet][isubjet]
            dr_part = deltaR(eta_particle, phi_particle, eta_c[ijet], phi_c[ijet])
            dr_jet.append(dr_part)
        dR.append(dr_jet)

    #make everything into numpy array with constant length of MAX_LEN. append 0 if less than MAX_LEN
    for ijet in range(Njets):
        if len(dEta[ijet]) < MAX_LEN:
            dEta[ijet] = np.pad(dEta[ijet], (0, MAX_LEN - len(dEta[ijet])), 'constant')
            dPhi[ijet] = np.pad(dPhi[ijet], (0, MAX_LEN - len(dPhi[ijet])), 'constant')
            log_pT[ijet] = np.pad(log_pT[ijet], (0, MAX_LEN - len(log_pT[ijet])), 'constant')
            log_eT[ijet] = np.pad(log_eT[ijet], (0, MAX_LEN - len(log_eT[ijet])), 'constant')
            ratio_log_pTj[ijet] = np.pad(ratio_log_pTj[ijet], (0, MAX_LEN - len(ratio_log_pTj[ijet])), 'constant')
            ratio_log_eTj[ijet] = np.pad(ratio_log_eTj[ijet], (0, MAX_LEN - len(ratio_log_eTj[ijet])), 'constant')
            dR[ijet] = np.pad(dR[ijet], (0, MAX_LEN - len(dR[ijet])), 'constant')
        else:
            dEta[ijet] = dEta[ijet][:MAX_LEN]
            dPhi[ijet] = dPhi[ijet][:MAX_LEN]
            log_pT[ijet] = log_pT[ijet][:MAX_LEN]
            log_eT[ijet] = log_eT[ijet][:MAX_LEN]
            ratio_log_pTj[ijet] = ratio_log_pTj[ijet][:MAX_LEN]
            ratio_log_eTj[ijet] = ratio_log_eTj[ijet][:MAX_LEN]
            dR[ijet] = dR[ijet][:MAX_LEN]
    #--print shape of returned data---
    # dEta = np.array([list(x) for x in dEta], dtype=object)
    # dPhi = np.array([list(x) for x in dPhi], dtype=object)
    # log_pT = np.array([list(x) for x in log_pT], dtype=object)
    # log_eT = np.array([list(x) for x in log_eT], dtype=object)
    # ratio_log_pTj = np.array([list(x) for x in ratio_log_pTj], dtype=object)
    # ratio_log_eTj = np.array([list(x) for x in ratio_log_eTj], dtype=object)
    # dR =np.array([list(x) for x in dR], dtype=object)

    dEta = np.array(dEta, dtype=object)
    dPhi = np.array(dPhi, dtype=object)
    log_pT = np.array(log_pT, dtype=object)
    log_eT = np.array(log_eT, dtype=object)
    ratio_log_pTj = np.array(ratio_log_pTj, dtype=object)
    ratio_log_eTj = np.array(ratio_log_eTj, dtype=object)
    dR = np.array(dR, dtype=object)

    # --print shape of returned data---
    print('dEta shape = {}'.format(dEta.shape))
    print('dPhi shape = {}'.format(dPhi.shape))
    print('log_pT shape = {}'.format(log_pT.shape))
    print('log_eT shape = {}'.format(log_eT.shape))
    print('ratio_log_pTj shape = {}'.format(ratio_log_pTj.shape))
    print('ratio_log_eTj shape = {}'.format(ratio_log_eTj.shape))
    print('dR shape = {}'.format(dR.shape))

    # save the data to csv
    data = {
        'dEta': dEta.tolist(),
        'dPhi': dPhi.tolist(),
        'log_pT': log_pT.tolist(),
        'log_eT': log_eT.tolist(),
        'ratio_log_pTj': ratio_log_pTj.tolist(),
        'ratio_log_eTj': ratio_log_eTj.tolist(),
        'dR': dR.tolist()
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    if(is_signal):
        df.to_csv('../../output/signal_trans.csv', index=False)
    else:
        df.to_csv('../../output/bg_trans.csv', index=False)




    ## ------------- do we need rotation for this ---------------  ?? 
    # tan_theta=[]#List of the tan(theta) angle to rotate to the principal axis in each jet image
    # Njets=len(Subjets[1])
    # for ijet in range(0,Njets):
    #     M11=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[2][ijet])
    #     M20=np.sum(Subjets[0][ijet]*Subjets[1][ijet]*Subjets[1][ijet])
    #     M02=np.sum(Subjets[0][ijet]*Subjets[2][ijet]*Subjets[2][ijet])
    #     tan_theta_use=2*M11/(M20-M02+np.sqrt(4*M11*M11+(M20-M02)*(M20-M02)))
    #     tan_theta.append(tan_theta_use)

    # rot_subjet = [[],[],[]]
    # Njets=len(Subjets[1])
    # for ijet in range(0,Njets):
    #     rot_subjet[0].append(Subjets[0][ijet]) 
    #     rot_subjet[1].append(Subjets[1][ijet]*np.cos(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.sin(np.arctan(tan_theta[ijet])))
    #     rot_subjet[2].append(-Subjets[1][ijet]*np.sin(np.arctan(tan_theta[ijet]))+Subjets[2][ijet]*np.cos(np.arctan(tan_theta[ijet])))
    #     #print('Rotated phi for jet 1 before fixing -pi<theta<pi = {}'.format(Subjets[2][0]))    
    #     rot_subjet[2][ijet]=np.unwrap(rot_subjet[2][ijet]) #We fix the angle phi to be between (-Pi,Pi]

    #--print shape of returned data---
    
    return dEta, dPhi, log_pT,log_eT, ratio_log_pTj, ratio_log_eTj, dR


def loadfiles(folder_path):
    # extected .txt format  :  event_no, eta, phi , pT, eT

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
            # subjets[1].append(np.array(data[:, 3]))  # eT
            subjets[1].append(np.array(data[:, 0]))  # eta
            subjets[2].append(np.array(data[:, 1]))  # phi
            all_files+=1
            # if(all_files >400):
            #     break
    
    return [np.array(subjets[0], dtype=object), np.array(subjets[1], dtype=object), np.array(subjets[2], dtype=object)]

def combine_sig_bg(sig_csv, bg_csv,val_size=0.1, test_size=0.1,output_dir="../../output/pf_out/"):
    # Load the CSV files
    sig_df = pd.read_csv(sig_csv)
    bg_df = pd.read_csv(bg_csv)

    # Add labels: 1 for signal, 0 for background
    sig_df['label'] = 1
    bg_df['label'] = 0

    # Combine and shuffle the data
    combined_df = pd.concat([sig_df, bg_df], ignore_index=True)
    combined_df = shuffle(combined_df, random_state=42).reset_index(drop=True)

    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(combined_df, test_size=(val_size + test_size), random_state=42, stratify=combined_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (val_size + test_size), random_state=42, stratify=temp_df['label'])

    # Save splits
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    return combined_df


if __name__ == "__main__":

    # signal = "../../dataset/eta_phi/tau_lj_constituents"
    # sig_subjets= loadfiles(signal) 
    # sig_subjets = np.array(sig_subjets)
    # print(sig_subjets.shape)
    # dEta, dPhi, log_pT,log_eT, ratio_log_pTj, ratio_log_eTj, dR = save_csv(sig_subjets, 1)


    # background = "../../dataset/eta_phi/hardqcd_lj_constituents" 
    # bg_subjets= loadfiles(background) 
    # bg_subjets = np.array(bg_subjets)
    # print(bg_subjets.shape)
    # dEta, dPhi, log_pT,log_eT, ratio_log_pTj, ratio_log_eTj, dR = save_csv(bg_subjets, 0)


    combine_sig_bg("../../output/signal_trans.csv", "../../output/bg_trans.csv")