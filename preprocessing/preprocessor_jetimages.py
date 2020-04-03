from __future__ import print_function, division

import h5py
from ImageUtils import *
import pandas as pd
import fastjet as fj

def generator(filename, chunk_size=1000,total_size=1100000, start=0):

    i = 0
    i_max = total_size//chunk_size

    for i in range(i_max):
        starting = start + i*chunk_size
        stopping = start + (i+1)*chunk_size

        yield pd.read_hdf(filename,start=starting, stop=stopping)





# Load files
total_size = 1000000
batch_size = 5000
iters = total_size//batch_size
fin_name = "events_LHCO2020_backgroundMC_Pythia.h5"
fout_name = "jetimages_Background.h5"
npix = 40
img_width = 1.4
start_evt = 0

R = 1.0
do_plots = False
has_sig_bit = False



print("going to read %i events with batch size %i \n \n" % (total_size, batch_size))
l=-1
for batch in generator(fin_name, chunk_size=batch_size, total_size=total_size):
    l+=1

    print("batch %i \n" %(l))

    events_combined = batch.T


    out_vec1 = []
    j1_images = []
    j2_images = []
    images = []
    signal_bits = []
    delta_eta = []
    for i in range(batch_size):

        #index for pd dataframe based on overall row index, so we have to correct for the batches
        idx = l*batch_size + i  + start_evt

        if(has_sig_bit):
            issignal = events_combined[idx][2100]
        else:
            issignal = False


        pjs = []
        for j in range(700):
            if (events_combined[idx][j*3]>0):
                pT = events_combined[idx][j*3]
                eta = events_combined[idx][j*3+1]
                phi = events_combined[idx][j*3+2]
                
                pj = fj.PseudoJet()
                pj.reset_PtYPhiM(pT, eta, phi, 0.)
                pjs.append(pj)

                
        
        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)    
        # Save the two leading jets
        jets = jet_def(pjs)
        jets = [j for j in jets]


        jet1_conts = []
        jet2_conts = []
        for pj in pjs:
            if(jets[0].delta_R(pj) < R):
                vec = [pj.perp(), pj.rapidity() - jets[0].rapidity(), ang_dist(pj.phi(),jets[0].phi()), 2122]
                jet1_conts.append(vec)
            if(jets[1].delta_R(pj) < R):
                vec = [pj.perp(), pj.rapidity()-jets[1].rapidity(), ang_dist(pj.phi(),jets[1].phi()), 2122]
                jet2_conts.append(vec)

        delta_eta.append(np.abs(jets[0].pseudorapidity() - jets[1].pseudorapidity()))
        #convert to format used by ef util
        #print("Jet1: ",jets[0].rapidity(), jets[0].phi())
        #print("Jet2: ", jets[1].rapidity(), jets[1].phi())
        j1_image = np.squeeze(my_pixelate(np.array(jet1_conts), npix = 40, img_width = img_width, norm = True))
        j2_image = np.squeeze(my_pixelate(np.array(jet2_conts), npix = 40, img_width = img_width, norm = True))
        mjj = (jets[0] + jets[1]).m()

        out_vec1.append([issignal, jets[0].perp(), jets[0].rapidity(), jets[0].phi(), jets[0].m(), 
                                  jets[1].perp(), jets[1].rapidity(), jets[1].phi(), jets[1].m(), mjj])
        j1_images.append(j1_image)
        j2_images.append(j2_image)

        #print(events_combined[:30])
        #print(input_format[:10])

    info_batch = np.array(out_vec1)
    #mass order the images
    if(jets[0].m() > jets[1].m()):
        images1_batch = np.array(j1_images)
        images2_batch = np.array(j2_images)
    else:
        images2_batch = np.array(j1_images)
        images1_batch = np.array(j2_images)

    if (l==0):
        with h5py.File(fout_name, "w") as f:
            f.create_dataset("j1_images", data=images1_batch, chunks = True, maxshape=(None,40,40))
            f.create_dataset("j2_images", data=images2_batch, chunks = True, maxshape=(None,40,40))
            f.create_dataset("jet_infos", data=info_batch, chunks = True, maxshape=(None,10))
    else:
        with h5py.File(fout_name, "a") as f:
            f['j1_images'].resize((f['j1_images'].shape[0] + images1_batch.shape[0]), axis=0)
            f['j1_images'][-images1_batch.shape[0]:] = images1_batch
            f['j2_images'].resize((f['j2_images'].shape[0] + images2_batch.shape[0]), axis=0)
            f['j2_images'][-images2_batch.shape[0]:] = images2_batch
            f['jet_infos'].resize((f['jet_infos'].shape[0] + info_batch.shape[0]), axis=0)
            f['jet_infos'][-info_batch.shape[0]:] = info_batch

                
    if(do_plots and l==0):
        images1_batch = np.squeeze(np.array(images1_batch))
        images2_batch = np.squeeze(np.array(images2_batch))
        signal_bits = np.array(info_batch[:,0])
        signal_mask = (signal_bits > 0.9)
        bkg_mask = (signal_bits < 0.1)

        delta_eta = np.array(delta_eta)

        fig = plt.figure(figsize=(12,9))
        plt.hist(delta_eta, bins = 20, color = 'b')
        plt.show()
        
        signal_images1 = images1_batch[signal_mask]
        signal_images2 = images2_batch[signal_mask]
        mean_signal1 = np.mean(signal_images1, axis=0)
        mean_signal2 = np.mean(signal_images2, axis=0)

        bkg_images1 = images1_batch[bkg_mask]
        bkg_images2 = images2_batch[bkg_mask]
        mean_bkg1 = np.mean(bkg_images1, axis=0)
        mean_bkg2 = np.mean(bkg_images2, axis=0)

        print("Avg pixel Sums: ", np.sum(mean_signal1), np.sum(mean_signal2), np.sum(mean_bkg1), np.sum(mean_bkg2))
        max_pix = max(np.max(mean_signal1), np.max(mean_signal2), np.max(mean_bkg1), np.max(mean_bkg2))
        print("Max pix is : ", max_pix)



        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

        ax1.imshow(mean_signal1, cmap='gray') #, vmin=0., vmax=max_pix)
        ax1.set_title("Signal: More Massive Jet")
        ax2.imshow(mean_signal2, cmap='gray')# , vmin=0., vmax=max_pix)
        ax2.set_title("Signal: Less Massive Jet")
        ax3.imshow(mean_bkg1, cmap='gray')#, vmin =0., vmax=max_pix)
        ax3.set_title("Background: More Massive Jet")
        ax4.imshow(mean_bkg2, cmap='gray')#, vmin=0., vmax=max_pix)
        ax4.set_title("Background: Less Massive Jet")
        plt.tight_layout()
        plt.show()
        exit(1)


print("Finished all batches! Output file saved to %s" %(fout_name))


            
                              



