import uproot
import h5py
import numpy as np
import sys
import pandas as pd
sys.path.append("/afs/cern.ch/user/p/pharris/pharris/public/delphes/FastJetInstallDir/fastjet-install/lib/python2.7/site-packages")  # noqa: E501
sys.path.append("/afs/cern.ch/user/p/pharris/pharris/public/delphes/ForSwig/processing")  # noqa: E501
import fastjet as fj  # noqa: E402
import NsubjettinessWrapper as ns  # noqa: E402
import SoftDropWrapper as sd  # noqa: E402

fileName = sys.argv[1]
outputName = sys.argv[2]
print(fileName)
print(outputName)


data = uproot.open(fileName)["Delphes"]

ef1pt = data.array("EFlowTrack.PT")
ef1eta = data.array("EFlowTrack.Eta")
ef1phi = data.array("EFlowTrack.Phi")

ef2pt = data.array("EFlowPhoton.ET")
ef2eta = data.array("EFlowPhoton.Eta")
ef2phi = data.array("EFlowPhoton.Phi")

ef3pt = data.array("EFlowNeutralHadron.ET")
ef3eta = data.array("EFlowNeutralHadron.Eta")
ef3phi = data.array("EFlowNeutralHadron.Phi")

nevt = len(data)
#nevt = 100

# Build Column Labels
column_labels = []
column_labels.append('Mjj')
column_labels.append('j1 pT')
column_labels.append('j2 pT')
column_labels.append('Mj1')

for i in range(7):
    column_labels.append('j1 tau{}{}'.format(i+2,i+1))

column_labels.append('j1 sqrt(tau^2_1)/tau^1_1')
column_labels.extend(['j1 n_trk', 'j1 M_trim', 'j1 M_prun',
                      'j1 M_mmdt', 'j1 M_sdb1', 'j1 M_sdb2', 'j1 M_sdm1'])

column_labels.append('Mj2')

for i in range(7):
    column_labels.append('j2 tau{}{}'.format(i+2,i+1))

column_labels.append('j2 sqrt(tau^2_1)/tau^1_1')
column_labels.extend(['j2 n_trk', 'j2 M_trim', 'j2 M_prun',
                      'j2 M_mmdt', 'j2 M_sdb1', 'j2 M_sdb2', 'j2 M_sdm1'])

print(column_labels)
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


out_vec = np.zeros((nevt, vec_size), dtype=np.float)
for i0 in range(nevt):
    if i0 % 500 == 0:
        print("Processing ", i0, i0 / nevt)
    part1 = np.stack((ef1pt[i0], ef1eta[i0], ef1phi[i0]), axis=1)
    part2 = np.stack((ef2pt[i0], ef2eta[i0], ef2phi[i0]), axis=1)
    part3 = np.stack((ef3pt[i0], ef3eta[i0], ef3phi[i0]), axis=1)
    tot1 = len(part1)
    tot2 = len(part2) + tot1
    pjs = []
    for i1 in range(len(part1)):
        if part1[i1][0] > 0:
            pj = fj.PseudoJet()
            #print(type(part1[i1][0].item()))
            #print(part1[i1][0], part1[i1][1], part1[i1][2], 0.)
            pj.reset_PtYPhiM(part1[i1][0].item(), part1[i1][1].item(), part1[i1][2].item(), 0.)
            pjs.append(pj)

    for i1 in range(len(part2)):
        if part2[i1][0] > 0:
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(part2[i1][0].item(), part2[i1][1].item(), part2[i1][2].item(), 0.)
            pjs.append(pj)

    for i1 in range(len(part3)):
        if part3[i1][0] > 0:
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(part3[i1][0].item(), part3[i1][1].item(), part3[i1][2].item(), 0.)
            pjs.append(pj)

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
    max_tau = 8
    # jet 1, beta = 0.5
    #tau1_bp5 = nsub_bp5.getTau(max_tau, jet1_conts)
    # jet 2, beta = 0.5
    #tau2_bp5 = nsub_bp5.getTau(max_tau, jet2_conts)
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

    #j1_tau_list = []
    # for k, (x, y, z) in enumerate(zip(tau1_bp5, tau1_b1, tau1_b2)):
    #    if k == 15:
    #        break
    #    j1_tau_list.extend([x, y, z])

    #print('j1list', *j1_tau_list)

    tau21 = tau1_b1[1] / max(eps, tau1_b1[0])
    tau32 = tau1_b1[2] / max(eps, tau1_b1[1])
    tau43 = tau1_b1[3] / max(eps, tau1_b1[2])
    tau54 = tau1_b1[4] / max(eps, tau1_b1[3])
    tau65 = tau1_b1[5] / max(eps, tau1_b1[4])
    tau76 = tau1_b1[6] / max(eps, tau1_b1[5])
    tau87 = tau1_b1[7] / max(eps, tau1_b1[6])
    sqrt_t1_b2_t1_b2 = np.sqrt(tau1_b2[0]) / max(eps, tau1_b1[0])

    n_trk = len(jet1_conts)
    pT1 = jets[0].perp()

    mass_trim = trimmer1(jets[0]).m()
    mass_mmdt = sd_mmdt.result(jet1_conts)[0].m()
    mass_prun = pruner1(jets[0]).m()
    mass_sdb1 = sd_sdb1.result(jet1_conts)[0].m()
    mass_sdb2 = sd_sdb2.result(jet1_conts)[0].m()
    mass_sdm1 = sd_sdm1.result(jet1_conts)[0].m()

    jet_1_props = [mj1, tau21, tau32, tau43, tau54, tau65, tau76, tau87, sqrt_t1_b2_t1_b2, n_trk,
                   mass_trim, mass_prun, mass_mmdt, mass_sdb1, mass_sdb2, mass_sdm1]

    # Second jet
    mj2 = jets[1].m()

    # j2_tau_list = []
    # for it, (x, y, z) in enumerate(zip(tau2_bp5, tau2_b1, tau2_b2)):
    #    if it == 15:
    #        break
    #    j2_tau_list.extend([x, y, z])

    # print('j2list',*j2_tau_list)
    #tau21 = tau2_b1[1] / max(eps, tau2_b1[0])
    #tau32 = tau2_b1[2] / max(eps, tau2_b1[1])
    #tau43 = tau2_b1[3] / max(eps, tau2_b1[2])
    #sqrt_t1_b2_t1_b2 = np.sqrt(tau2_b2[0]) / max(eps, tau2_b1[0])
    tau21 = tau2_b1[1] / max(eps, tau2_b1[0])
    tau32 = tau2_b1[2] / max(eps, tau2_b1[1])
    tau43 = tau2_b1[3] / max(eps, tau2_b1[2])
    tau54 = tau2_b1[4] / max(eps, tau2_b1[3])
    tau65 = tau2_b1[5] / max(eps, tau2_b1[4])
    tau76 = tau2_b1[6] / max(eps, tau2_b1[5])
    tau87 = tau2_b1[7] / max(eps, tau2_b1[6])
    sqrt_t1_b2_t1_b2 = np.sqrt(tau2_b2[0]) / max(eps, tau2_b1[0])
    n_trk = len(jet2_conts)
    pT2 = jets[1].perp()

    mjj = (jets[0] + jets[1]).m()

    mass_mmdt = sd_mmdt.result(jet2_conts)[0].m()
    mass_trim = trimmer1(jets[1]).m()
    mass_prun = pruner1(jets[1]).m()
    mass_sdb1 = sd_sdb1.result(jet2_conts)[0].m()
    mass_sdb2 = sd_sdb2.result(jet2_conts)[0].m()
    mass_sdm1 = sd_sdm1.result(jet2_conts)[0].m()

    jet_2_props = [mj2, tau21, tau32, tau43, tau54, tau65, tau76, tau87, sqrt_t1_b2_t1_b2, n_trk,
                   mass_trim, mass_prun, mass_mmdt, mass_sdb1, mass_sdb2, mass_sdm1]
    vec = [mjj, pT1, pT2]

    if mj1 > mj2:
        vec.extend(jet_1_props)
        vec.extend(jet_2_props)
    else:
        vec.extend(jet_2_props)
        vec.extend(jet_1_props)

    out_vec[i0] = vec

df = pd.DataFrame(out_vec, columns=column_labels)
print(df.head)
df.to_hdf(outputName, "data", mode='w', format='t')

print("All Done")
