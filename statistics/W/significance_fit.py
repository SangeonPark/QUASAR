#!/usr/bin/env python
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist
from array import array
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm, kstest
from pvalue import *

fOutput="Output.root"
fHists=[]

def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

def drawFrame(iX,iData,iFuncs,iCat):
    lCan   = r.TCanvas("qcd_"+iCat,"qcd_"+iCat,800,600)
    leg = r.TLegend(0.55,0.63,0.86,0.87)
    lFrame = iX.frame()
    lFrame.SetTitle("")
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    lFrame.GetYaxis().SetTitle("Events")
    #iBkg.plotOn(lFrame,r.RooFit.FillColor(r.TColor.GetColor(100, 192, 232)),r.RooFit.FillStyle(3008), r.RooFit.DrawOption("E3"), r.RooFit.LineColor(r.kBlue))
    iData.plotOn(lFrame)
    iFuncs[1].plotOn(lFrame,r.RooFit.LineColor(r.kGreen+1))
    iFuncs[1].plotOn(lFrame,r.RooFit.LineColor(r.kRed+1),r.RooFit.LineStyle(r.kDashed),r.RooFit.Components(iFuncs[0].GetName()))
    leg.SetFillColor(0)
    lFrame.Draw()
    lTmpData  = r.TH1F("tmpData" ,"tmpData" ,1,0,10); lTmpData .SetMarkerStyle(r.kFullCircle); 
    lTmpBkg   = r.TH1F("tmpBkg"  ,"tmpBkg"  ,1,0,10); lTmpBkg  .SetFillStyle(3008); lTmpBkg.SetLineColor(r.kBlue); lTmpBkg.SetFillColor(r.TColor.GetColor(100, 192, 232));
    lTmpFunc1 = r.TH1F("tmpFunc1","tmpFunc1",1,0,10); lTmpFunc1.SetLineColor(51);                lTmpFunc1.SetLineWidth(2); lTmpFunc1.SetLineStyle(r.kDashed);
    lTmpFunc2 = r.TH1F("tmpFunc2","tmpFunc2",1,0,10); lTmpFunc2.SetLineColor(61);                lTmpFunc2.SetLineWidth(2); lTmpFunc2.SetLineStyle(r.kDashed);
    lTmpFunc3 = r.TH1F("tmpFunc3","tmpFunc3",1,0,10); lTmpFunc3.SetLineColor(r.kGreen+1);        lTmpFunc3.SetLineWidth(2); #lTmpFunc3.SetLineStyle(r.kDashed);
    leg.AddEntry(lTmpData,"data","lpe")
    leg.AddEntry(lTmpBkg ,"loss-sideband data","f")
    leg.AddEntry(lTmpFunc2,"bkg","lp")
    leg.AddEntry(lTmpFunc3,"sig+bkg","lp")
    leg.AddEntry(lTmpFunc1,"loss-sideband","lp")
    leg.Draw()
    lCan.Modified()
    lCan.Update()
    lCan.SaveAs(lCan.GetName()+".png")
    #end()
    
# build workspace
def workspace(iOutput,iDatas,iFuncs,iCat="cat0"):
    print('--- workspace')
    lW = r.RooWorkspace("w_"+str(iCat))
    for pData in iDatas:
        print('adding data ',pData,pData.GetName())
        getattr(lW,'import')(pData,r.RooFit.RecycleConflictNodes())    
    for pFunc in iFuncs:
        print('adding func ',pFunc,pFunc.GetName())
        getattr(lW,'import')(pFunc,r.RooFit.RecycleConflictNodes())
    if iCat.find("pass_cat0") == -1:
        lW.writeToFile(iOutput,False)
    else:
        lW.writeToFile(iOutput)
    return lW

def clip(iData,iMin,iMax):
    pMinBin = 0
    pMaxBin = iData.GetNbinsX()
    for i0 in range(iData.GetNbinsX()+1):
        pLVal = iData.GetBinLowEdge(i0)
        pHVal = iData.GetBinLowEdge(i0)
        if iMin > pLVal:
            pMinBin = i0
        if iMax > pHVal:
            pMaxBin = i0
    NBins = pMaxBin-pMinBin
    pMinLow = iData.GetBinLowEdge(pMinBin)
    pMinMax = iData.GetBinLowEdge(pMaxBin)
    pData = r.TH1F(iData.GetName()+"R",iData.GetName()+"R",NBins,pMinLow,pMinMax)
    for i0 in range(NBins):
        print(iData.GetBinLowEdge(i0+pMinBin),pData.GetBinLowEdge(i0+1),"! Done")
        pData.SetBinContent(i0+1,iData.GetBinContent(i0+pMinBin))
    fHists.append(pData)
    return pData
    
def fitFunc(iData,iCat,iMin=3000,iMax=6000,iStep=150,iFixToSB=False):
    pData = iData
    #pBkg  = iBkg#clip(iBkg ,3000,6200)
    lXMin=pData.GetXaxis().GetXmin()
    lXMax=pData.GetXaxis().GetXmax()
    lNBins=pData.GetNbinsX()
    lX = r.RooRealVar("x","x",lXMin,lXMax)
    lX.setBins(lNBins)
    lNTot   = r.RooRealVar("qcdnorm_"+iCat,"qcdnorm_"+iCat,pData.Integral(),0,3*pData.Integral())
    lA0     = r.RooRealVar   ("a0"+"_"+iCat,"a0"+"_"+iCat,0.00,-1.,1.)          
    lA1     = r.RooRealVar   ("a1"+"_"+iCat,"a1"+"_"+iCat,0.01,-1,1.)
    lA2     = r.RooRealVar   ("a2"+"_"+iCat,"a2"+"_"+iCat,0.01,-1,1)
    lA3     = r.RooRealVar   ("a3"+"_"+iCat,"a3"+"_"+iCat,0.01,-1,1)
    lA4     = r.RooRealVar   ("a4"+"_"+iCat,"a4"+"_"+iCat,0.01,-1,1)
    lA5     = r.RooRealVar   ("a5"+"_"+iCat,"a5"+"_"+iCat,0.01,-1,1)
    lQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,lX,r.RooArgList(lA0,lA1,lA2,lA3,lA4,lA5))
    lQCDP   = r.RooExtendPdf("qcd_"+iCat, "qcd"+iCat,lQFuncP,lNTot)

    #lA0      = r.RooRealVar   ("a0"+"_"+iCat,"a0"+"_"+iCat,1.0,-200.,200.); lA0.setConstant(r.kTRUE)
    #lA1      = r.RooRealVar   ("a1"+"_"+iCat,"a1"+"_"+iCat,1.00,-200.,200.)
    #lA2      = r.RooRealVar   ("a2"+"_"+iCat,"a2"+"_"+iCat,3.00,-200.,200.)          
    #lQFuncP  = r.RooGenericPdf("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,"(1-@0/13000.)**@2*(@1/13000.)**-@2",r.RooArgList(lX,lA1,lA2))#,lA5))
    #lQCDP   = r.RooExtendPdf("qcd_"+iCat, "qcd"+iCat,lQFuncP,lNTot)

    #lBNTot   = r.RooRealVar("bqcdnorm_"+iCat,"bqcdnorm_"+iCat,pData.Integral(),0,3*pData.Integral())
    #lBA0      = r.RooRealVar   ("ba0"+"_"+iCat,"ba0"+"_"+iCat,0.00,-200.,200.)
    #lBA1      = r.RooRealVar   ("ba1"+"_"+iCat,"ba1"+"_"+iCat,0.00,-200.,200.)
    #lBA2      = r.RooRealVar   ("ba2"+"_"+iCat,"ba2"+"_"+iCat,0.00,-200.,200.)          
    #lBQFuncP  = r.RooGenericPdf("btqcd_pass_"+iCat,"btqcd_pass_"+iCat,"(1-@0/13000.)**@1*(@0/13000.)**-@2",r.RooArgList(lX,lBA1,lBA2))
    #lBQCDP    = r.RooExtendPdf ("bqcd_"+iCat, "bqcd"+iCat,lBQFuncP,lBNTot)

    lMass   = r.RooRealVar("mass","mass"  ,82,50,150); lMass.setConstant(r.kTRUE)
    lSigma  = r.RooRealVar("sigma","Width of Gaussian",8,3,20); lSigma.setConstant(r.kTRUE)
    lGaus   = r.RooGaussian("gauss","gauss(x,mean,sigma)",lX,lMass,lSigma)
    lNSig   = r.RooRealVar("signorm_"+iCat,"signorm_"+iCat,0.1*pData.Integral(),-0.1*pData.Integral(),0.1*pData.Integral())
    lSig    = r.RooExtendPdf("sig_"+iCat, "sig_"+iCat,lGaus,lNSig)
    lTot    = r.RooAddPdf("model", "model", r.RooArgList(lSig, lQCDP))
    lHData  = r.RooDataHist("data_obs","data_obs", r.RooArgList(lX),pData)

    #lQCDP.fitTo(lHData);
    #if iFixToSB:
    #    lA1.setConstant(r.kTRUE); lA2.setConstant(r.kTRUE);
    lTot.fitTo(lHData,r.RooFit.Extended(r.kTRUE))
    drawFrame(lX,lHData,[lQCDP,lTot],iCat)

    #print(lNSig.getVal(),lNSig.getError())
    #return lNSig.getVal(),lNSig.getError()

    lW = workspace(fOutput,[lHData],[lTot,lQCDP],iCat)
    lW.defineSet("poi","signorm_"+iCat)
    bmodel = r.RooStats.ModelConfig("b_model",lW)
    bmodel.SetPdf(lW.pdf("model"))
    bmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lA4,lA5,lNTot))
    bmodel.SetObservables(r.RooArgSet(lX))
    bmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(0)
    bmodel.SetSnapshot(lW.set("poi"))

    sbmodel = r.RooStats.ModelConfig("s_model",lW)
    sbmodel.SetPdf(lW.pdf("model"))
    sbmodel.SetNuisanceParameters(r.RooArgSet(lA1,lA2,lA3,lA4,lA5,lNTot))
    sbmodel.SetObservables(r.RooArgSet(lX))
    sbmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(lNSig.getVal())
    sbmodel.SetSnapshot(lW.set("poi"))
    lW.var("mass").setVal(81.)
    ac = r.RooStats.AsymptoticCalculator(lHData, sbmodel, bmodel)
    ac.SetOneSidedDiscovery(True)
    ac.SetPrintLevel(-1)
    asResult = ac.GetHypoTest()
    sig=asResult.Significance() 
    return sig

def setupData(iFileName):
    lDatas=[]
    lFile = r.TFile(iFileName)
    lH    = lFile.Get("data_obs")
    lH2   = lFile.Get("bkgestimate")

    lH.SetDirectory(0)
    lH2.SetDirectory(0)
    for i1 in range(lH.GetNbinsX()+1):
        lH.SetBinError(i1,math.sqrt(lH.GetBinContent(i1)))
        lH2.SetBinError(i1,math.sqrt(lH2.GetBinContent(i1)))
    lFile.Close()

    return lH, lH2

def sigVsMassPlot(masses,pvalues,labels):
    lC0 = r.TCanvas("A","A",800,600)
    leg = r.TLegend(0.55,0.23,0.86,0.47)
    leg.SetFillColor(0)
    
    lGraphs=[]
    sigmas=[]
    for i0 in range(len(masses)):
        graph1 = r.TGraph(len(masses[i0]),masses[i0],pvalues[i0])
        graph1.SetMarkerStyle(20)
        graph1.GetXaxis().SetTitle("m_{jj} (GeV)")
        graph1.GetYaxis().SetTitle("p^{0} value")
        graph1.SetTitle("Significance vs Mass")
        graph1.SetLineColor(2+i0)
        graph1.SetMarkerColor(2+i0)
        graph1.SetLineWidth(2+i0)
        r.gPad.SetLogy(True)
        graph1.GetYaxis().SetRangeUser(1e-8,1.0)
        if i0 == 0:
            graph1.Draw("alp")
        else:
            graph1.Draw("lp")
        lGraphs.append(graph1)
        leg.AddEntry(graph1,labels[i0],"lp")
            
    #sigmas=[0.317,0.045,0.0027,0.0000633721,0.0000005742]
    lines=[]
    for i0 in range(5):#len(sigmas)):
        sigmas.append(1-norm.cdf(i0+1))
        lLine = r.TLine(masses[0][0],sigmas[i0],masses[0][len(masses[0])-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        lPT = r.TPaveText(3500,sigmas[i0],4000,sigmas[i0]+1.5*sigmas[i0])
        lPT.SetFillStyle(4050)
        lPT.SetFillColor(0)
        lPT.SetBorderSize(0)
        lPT.AddText(str(i0+1)+"#sigma")
        lPT.Draw()
        lines.append(lLine)
        lines.append(lPT)

    for pGraph in lGraphs:
        pGraph.Draw("lp")
        
    leg.Draw()
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue.png")
    end()

def pvalue(iData):
    bins=[]
    data=[]
    pData = clip(iData,3200,6500)
    for i0 in range(pData.GetNbinsX()+1):
        bins.append(pData.GetBinLowEdge(i0+1))
        if i0 < pData.GetNbinsX():
            data.append(pData.GetBinContent(i0+1))
    masks       = [[bin_i,bin_i, bin_i] for bin_i in range(1,len(bins)-2)]    
    pvalues_in = [get_p_value(data,bins,mask=mask,verbose=0,plotfile=None) for i, mask in enumerate(masks)]
    masses_in  = [0.5*(bins[i] + bins[i+1]) for i in range(len(bins)-1)]
    masses =  array( 'd' )
    pvalues = array( 'd' )
    print("!!!!",len(masses_in),len(pvalues_in))
    for i0 in range(len(pvalues_in)):
        masses.append(masses_in[i0])
        pvalues.append(pvalues_in[i0])
    return masses,pvalues

def loadPkl(iName,iCut1=0.3,iCut2=0.1,iNBins=60,iXMin=50,iXMax=140):
    df = pd.read_pickle(iName)
    lHTot = r.TH1F("A","A",iNBins,iXMin,iXMax)
    for m in df['Jet Mass'][np.logical_and(df['QCD Model Loss'] > iCut1,df['WQQ Model Loss'] < iCut2)]:
        lHTot.Fill(m)
    fHists.append(lHTot)
    return lHTot
    #lC0 = r.TCanvas("A","A",800,600)
    #lHTot.Draw()
    #lC0.Modified()
    #lC0.Update()
    #lC0.SaveAs(lC0.GetName()+".png")
    #end()
    
if __name__ == "__main__":
    r.gROOT.SetBatch(True)
    iBkgTemp=True
    labels=[]
    masses=[]
    pvalues=[]
    #lData1 = loadPkl('jet_masses_and_losses',0.70,0.25)
    #sig = fitFunc(lData1,"AAA",50,150,100,iBkgTemp)
    lNBins = 20
    lScale1=0.05
    lScale2=0.03
    l2DHist  = r.TH2F("A","A",lNBins,0,lScale1*lNBins,lNBins,0,lScale2*lNBins)
    for cuts1 in range(0,lNBins):
        for cuts2 in range(0,lNBins):
            lData1 = loadPkl('jet_masses_and_losses',cuts1*lScale1,cuts2*lScale2)
            sig = fitFunc(lData1,str(cuts1*lScale1)+" "+str(cuts2*lScale2),50,150,100,iBkgTemp)
            l2DHist.SetBinContent(cuts1+1,cuts2+1,sig)
            #masses1,pvalues1=fitFunc(lData1,lData1,str(cuts1*0.05)+" "+str(cuts2*0.05),50,150,100,iBkgTemp)
            #labels.append(str(cuts1*0.05)+" "+str(cuts2*0.05))
            #pvalues.append(pvalues1)
            #masses.append(masses1)
    #sigVsMassPlot(masses,pvalues,labels)
    lC0 = r.TCanvas("A","A",800,600)
    l2DHist.SetTitle("")
    r.gStyle.SetOptStat(0)
    l2DHist.GetYaxis().SetTitle("W Loss")
    l2DHist.GetXaxis().SetTitle("QCD Loss")
    l2DHist.Draw("colz")
    lC0.Modified()
    lC0.Update()
    lC0.SaveAs(lC0.GetName()+".png")
    end()
