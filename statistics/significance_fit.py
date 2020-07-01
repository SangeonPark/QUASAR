#! /usr/bin/env python
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist
from array import array
import numpy as np
#from stats import *

fOutput="Output.root"

def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]

def drawFrame(iX,iData,iFuncs,iCat):
    lCan   = r.TCanvas("qcd_"+iCat,"qcd_"+iCat,800,600)
    lFrame = iX.frame()
    lFrame.SetTitle("")
    lFrame.GetXaxis().SetTitle("m_{jj} (GeV)")
    iData.plotOn(lFrame)
    iColor=51
    for pFunc in iFuncs:
        pFunc.plotOn(lFrame,r.RooFit.LineColor(iColor),r.RooFit.LineStyle(iColor != 50+1))
        iColor+=10
    lFrame.Draw()
    lCan.Modified()
    lCan.Update()
    lCan.SaveAs(lCan.GetName()+".png")

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

        
def fitFunc(iData,iCat,iMin=3000,iMax=6000,iStep=150):
    lXMin=iData.GetXaxis().GetXmin()
    lXMax=iData.GetXaxis().GetXmax()
    lNBins=iData.GetNbinsX()
    lX = r.RooRealVar("x","x",lXMin,lXMax)
    lX.setBins(lNBins)
    lNTot   = r.RooRealVar("qcdnorm_"+iCat,"qcdnorm_"+iCat,iData.Integral(),0,3*iData.Integral())
    lA0     = r.RooRealVar   ("a0"+"_"+iCat,"a0"+"_"+iCat,0.00,-1.,1.)          
    lA1     = r.RooRealVar   ("a1"+"_"+iCat,"a1"+"_"+iCat,0.01,-1,1.)
    lA2     = r.RooRealVar   ("a2"+"_"+iCat,"a2"+"_"+iCat,0.01,-1,1)
    lA3     = r.RooRealVar   ("a3"+"_"+iCat,"a3"+"_"+iCat,0.01,-1,1)
    lA4     = r.RooRealVar   ("a4"+"_"+iCat,"a4"+"_"+iCat,0.01,-1,1)
    lA5     = r.RooRealVar   ("a5"+"_"+iCat,"a5"+"_"+iCat,0.01,-1,1)
    #lQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,iX,r.RooArgList(lA0,lA1,lA2,lA3))
    lQFuncP = r.RooBernstein("tqcd_pass_"+iCat,"tqcd_pass_"+iCat,lX,r.RooArgList(lA0,lA1,lA2,lA3))#,lA5))
    lQCDP   = r.RooExtendPdf("qcd_"+iCat, "qcd"+iCat,lQFuncP,lNTot)

    lMass   = r.RooRealVar("mass","mass"  ,3875,3000,6000); lMass.setConstant(r.kTRUE)
    lSigma  = r.RooRealVar("sigma","Width of Gaussian",60,20,500); lSigma.setConstant(r.kTRUE)
    lGaus   = r.RooGaussian("gauss","gauss(x,mean,sigma)",lX,lMass,lSigma)
    lNSig   = r.RooRealVar("signorm_"+iCat,"signorm_"+iCat,iData.Integral()*0.05,0,0.3*iData.Integral())
    lSig    = r.RooExtendPdf("sig_"+iCat, "sig_"+iCat,lGaus,lNSig)
    lTot    = r.RooAddPdf("model", "model", r.RooArgList(lSig, lQCDP))

    lHData  = r.RooDataHist("data_obs","data_obs", r.RooArgList(lX),iData)
    lTot.fitTo(lHData,r.RooFit.Extended(r.kTRUE))#,r.RooFit.PrintLevel(-1))
    drawFrame(lX,lHData,[lTot,lQCDP],iCat)

    lW = workspace(fOutput,[lHData],[lTot,lQCDP],iCat)
    lW.defineSet("poi","signorm_"+iCat)
    bmodel = r.RooStats.ModelConfig("b_model",lW)
    bmodel.SetPdf(lW.pdf("model"))
    bmodel.SetNuisanceParameters(r.RooArgSet(lA0,lA1,lA2,lA3,lNTot))
    bmodel.SetObservables(r.RooArgSet(lX))
    bmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(0)
    bmodel.SetSnapshot(lW.set("poi"))

    sbmodel = r.RooStats.ModelConfig("s_model",lW)
    sbmodel.SetPdf(lW.pdf("model"))
    sbmodel.SetNuisanceParameters(r.RooArgSet(lA0,lA1,lA2,lA3,lNTot))
    sbmodel.SetObservables(r.RooArgSet(lX))
    sbmodel.SetParametersOfInterest(lW.set("poi"))
    lW.var("signorm_"+iCat).setVal(lNSig.getVal())
    sbmodel.SetSnapshot(lW.set("poi"))

    masses =  array( 'd' )
    pvalues = array( 'd' )
    stepsize = (iMax-iMin)/iStep
    masslist = [iMin + i*stepsize for i in range(iStep+1)]
    for mass in masslist:
        lW.var("mass").setVal(mass)
        ac = r.RooStats.AsymptoticCalculator(lHData, sbmodel, bmodel)
        ac.SetOneSidedDiscovery(True)
        ac.SetPrintLevel(-1)
        asResult = ac.GetHypoTest()
        pvalue=asResult.NullPValue()
        masses.append(mass)
        pvalues.append(pvalue)
        print(mass,pvalue)
    return masses,pvalues 

def setupData(iFileName):
    lDatas=[]
    lFile = r.TFile(iFileName)
    lH    = lFile.Get("data_obs")
    lH.SetDirectory(0)
    for i1 in range(lH.GetNbinsX()+1):
        lH.SetBinError(i1,math.sqrt(lH.GetBinContent(i1)))
    lFile.Close()
    return lH

def sigVsMassPlot(masses,pvalues):
    lC0 = r.TCanvas("A","A",800,600)
    graph1 = r.TGraph(len(masses),masses,pvalues)
    graph1.SetMarkerStyle(20)
    graph1.GetXaxis().SetTitle("mass")
    graph1.GetYaxis().SetTitle("p0 value")
    graph1.SetTitle("Significance vs Mass")
    graph1.SetLineColor(4)
    graph1.SetMarkerColor(4)
    r.gPad.SetLogy(True)
    graph1.GetYaxis().SetRangeUser(1e-8,1.0)
    sigmas=[0.317,0.045,0.0027,0.0000633721,0.0000005742]
    graph1.Draw("alp")
    lines=[]
    for i0 in range(len(sigmas)):
        lLine = r.TLine(masses[0],sigmas[i0],masses[len(masses)-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        #lPT = r.TPaveText(masses[0+2],sigmas[i0],masses[len(masses)],sigmas[i0]+0.01)
        #lPT.AddText(str(i0)+"#sigma")
        #lPT.Draw()
        lines.append(lLine)
        #lines.append(lPT)
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue.png")
    end()
    
if __name__ == "__main__":
    lData        = setupData("blackbox1.root")
    masses,pvalues=fitFunc(lData,"test",3000,6000,300)
    sigVsMassPlot(masses,pvalues)
