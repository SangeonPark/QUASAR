from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import sys
sys.path.append("/data/t3home000/spark/FastJetInstallDir/install_dir/lib/python3.6/site-packages")
sys.path.append("/data/t3home000/spark/ForSwig/processing")
import fastjet as fj
import NsubjettinessWrapper as ns
import SoftDropWrapper as sd

print('successfully imported fasjet libraries')


def non_zero(arr):
    out = []
    eps = 1e-8
    for i in range(len(arr)):
        out.append(max(arr[i], eps))
    return out

def generator(filename, chunk_size=1000, total_size=1100000):

    i = 0
    i_max = total_size // chunk_size

    for i in range(i_max):

        yield pd.read_hdf(filename, start=i * chunk_size, stop=(i + 1) * chunk_size)


total_size = 1100000
#total_size = 10000
batch_size = 5000
iters = total_size // batch_size
fin_name = "/data/t3home000/spark/LHCOlympics/data/events_anomalydetection.h5"
fout_name = "nsubjettiness_massratio_rnd.h5"


# column_labels = [
#    'Mjj',
#    'Mj1', 'j1 tau21', 'j1 tau32', 'j1 tau43', 'j1 sqrt(tau^2_1)/tau^1_1',
#    'j1 n_trk', 'j1 pT1', 'j1 M_trim', 'j1 M_prun', 'j1 M_mmdt', 'j1 M_sdb1', 'j1 M_sdb2', 'j1 M_sdm1',
#    'Mj2', 'j2 tau21', 'j2 tau32', 'j2 tau43', 'j2 sqrt(tau^2_1)/tau^1_1',
#    'j2 n_trk', 'j2 pT1', 'j2 M_trim', 'j2 M_prun', 'j2 M_mmdt', 'j2 M_sdb1', 'j2 M_sdb2', 'j2 M_sdm1',
#    'isSignal'
#]

column_labels = []
column_labels.append('Mjj')
column_labels.append('Mj1')

for i in range(15):
    column_labels.append(f'j1 tau{i+1}(b=.5)')
    column_labels.append(f'j1 tau{i+1}(b=1)')
    column_labels.append(f'j1 tau{i+1}(b=2)')

column_labels.extend(['j1 n_trk', 'j1 pT1', 'j1 M_trim', 'j1 M_prun',
                      'j1 M_mmdt', 'j1 M_sdb1', 'j1 M_sdb2', 'j1 M_sdm1'])

column_labels.append('Mj2')

for i in range(15):
    column_labels.append(f'j2 tau{i+1}(b=.5)')
    column_labels.append(f'j2 tau{i+1}(b=1)')
    column_labels.append(f'j2 tau{i+1}(b=2)')


column_labels.extend(['j2 n_trk', 'j2 pT1', 'j2 M_trim', 'j2 M_prun',
                      'j2 M_mmdt', 'j2 M_sdb1', 'j2 M_sdb2', 'j2 M_sdm1'])




column_labels.append('isSignal')

print(column_labels)
print(f'length of the labels {len(column_labels)}')

vec_size = len(column_labels)


R = 0.8
beta = 0.5
nsub_bp5 = ns.Nsubjettiness(beta, R, 0, 6)

beta = 1.0
nsub_b1 = ns.Nsubjettiness(beta, R, 0, 6)

beta = 2.0
nsub_b2 = ns.Nsubjettiness(beta, R, 0, 6)

zcut_sd = 0.1
R = 0.8

sd_mmdt = sd.SoftDropWrapper(0.0, zcut_sd, R, 0.)
sd_sdb1 = sd.SoftDropWrapper(1.0, zcut_sd, R, 0.)
sd_sdb2 = sd.SoftDropWrapper(2.0, zcut_sd, R, 0.)
sd_sdm1 = sd.SoftDropWrapper(-1.0, zcut_sd, R, 0.)


pruner1 = fj.Pruner(fj.cambridge_algorithm, 0.1, 0.5)
trimmer1 = fj.Filter(fj.JetDefinition(fj.kt_algorithm, 0.2),
                     fj.SelectorPtFractionMin(0.03))


print("going to read %i events with batch size %i \n \n" %
      (total_size, batch_size))
l = 0

for batch in generator(fin_name, chunk_size=batch_size, total_size=total_size):

    print("batch %i \n" % (l))

    events_combined = batch.T

    out_vec = np.zeros((batch_size, vec_size), dtype=np.float)

    for i in range(batch_size):
        idx = l * batch_size + i
        issignal = events_combined[idx][2100]
        if (i % 10000 == 0):
            print(i)
            pass

        pjs = []
        # Save in an array only the entries which represent particles (i.e. non zero pT)
        for j in range(700):
            if (events_combined[idx][j * 3] > 0):
                pT = events_combined[idx][j * 3]
                eta = events_combined[idx][j * 3 + 1]
                phi = events_combined[idx][j * 3 + 2]

                pj = fj.PseudoJet()
                pj.reset_PtYPhiM(pT, eta, phi, 0.)
                pjs.append(pj)

        # Cluster the jets

        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)
        # Save the two leading jets
        jets = jet_def(pjs)
        jets = [j for j in jets]

        jet1_conts = []
        jet2_conts = []
        # Save the particles associated to the 2 main jets
        for pj in pjs:
            if(jets[0].delta_R(pj) < R):
                vec = [pj.px(), pj.py(), pj.pz(), pj.e()]
                jet1_conts.extend(vec)
            if(jets[1].delta_R(pj) < R):
                vec = [pj.px(), pj.py(), pj.pz(), pj.e()]
                jet2_conts.extend(vec)

        # Get Nsubjettiness properties
        max_tau = 16
        # jet 1, beta = 0.5
        tau1_bp5 = nsub_bp5.getTau(max_tau, jet1_conts)
        # jet 2, beta = 0.5
        tau2_bp5 = nsub_bp5.getTau(max_tau, jet2_conts)
        # jet 1, beta = 1
        tau1_b1 = nsub_b1.getTau(max_tau, jet1_conts)
        # jet 1, beta = 2
        tau2_b1 = nsub_b1.getTau(max_tau, jet2_conts)
        # jet 2, beta = 1
        tau1_b2 = nsub_b2.getTau(max_tau, jet1_conts)
        # jet 2, beta = 2
        tau2_b2 = nsub_b2.getTau(max_tau, jet2_conts)

        # avoid division by zero
        eps = 1e-8

        # First jet

        # raw jet mass of jets[0] ( jet 1 )
        mj1 = jets[0].m()

        j1_tau_list = []
        for k, (x, y, z) in enumerate(zip(tau1_bp5, tau1_b1, tau1_b2)):
            if k == 15:
                break
            j1_tau_list.extend([x, y, z])

        #print('j1list', *j1_tau_list)

        #tau21 = tau1_b1[1] / max(eps, tau1_b1[0])
        #tau32 = tau1_b1[2] / max(eps, tau1_b1[1])
        #tau43 = tau1_b1[3] / max(eps, tau1_b1[2])
        #sqrt_t1_b2_t1_b2 = np.sqrt(tau1_b2[0]) / max(eps, tau1_b1[0])

        n_trk = len(jet1_conts)
        pT1 = jets[0].perp()

        mass_trim = trimmer1(jets[0]).m()
        mass_mmdt = sd_mmdt.result(jet1_conts)[0].m()
        mass_prun = pruner1(jets[0]).m()
        mass_sdb1 = sd_sdb1.result(jet1_conts)[0].m()
        mass_sdb2 = sd_sdb2.result(jet1_conts)[0].m()
        mass_sdm1 = sd_sdm1.result(jet1_conts)[0].m()

        jet_1_props = [mj1, *j1_tau_list, n_trk, pT1,
                       mass_trim, mass_prun, mass_mmdt, mass_sdb1, mass_sdb2, mass_sdm1]

        # Second jet
        mj2 = jets[1].m()

        j2_tau_list = []
        for it, (x, y, z) in enumerate(zip(tau2_bp5, tau2_b1, tau2_b2)):
            if it == 15:
                break
            j2_tau_list.extend([x, y, z])

        #print('j2list',*j2_tau_list)
        #tau21 = tau2_b1[1] / max(eps, tau2_b1[0])
        #tau32 = tau2_b1[2] / max(eps, tau2_b1[1])
        #tau43 = tau2_b1[3] / max(eps, tau2_b1[2])
        #sqrt_t1_b2_t1_b2 = np.sqrt(tau2_b2[0]) / max(eps, tau2_b1[0])
        n_trk = len(jet2_conts)
        pT2 = jets[1].perp()

        mjj = (jets[0] + jets[1]).m()

        mass_mmdt = sd_mmdt.result(jet2_conts)[0].m()
        mass_trim = trimmer1(jets[1]).m()
        mass_prun = pruner1(jets[1]).m()
        mass_sdb1 = sd_sdb1.result(jet2_conts)[0].m()
        mass_sdb2 = sd_sdb2.result(jet2_conts)[0].m()
        mass_sdm1 = sd_sdm1.result(jet2_conts)[0].m()

        jet_2_props = [mj2, *j2_tau_list, n_trk, pT2,
                       mass_trim, mass_prun, mass_mmdt, mass_sdb1, mass_sdb2, mass_sdm1]
        vec = [mjj]

        if mj1 > mj2:
            vec.extend(jet_1_props)
            vec.extend(jet_2_props)
            vec.extend([issignal])
        else:
            vec.extend(jet_2_props)
            vec.extend(jet_1_props)
            vec.extend([issignal])


        out_vec[i] = vec
        #print('vec', out_vec[i])

    #print(out_vec[:5])
    df = pd.DataFrame(out_vec, columns=column_labels)
    print(df.head)
    if (l == 0):
        df.to_hdf(fout_name, "data", mode='w', format='t')
    else:
        df.to_hdf(fout_name, "data", mode='r+', format='t', append=True)
    l += 1


print("Finished all batches! Output file saved to %s" % (fout_name))
