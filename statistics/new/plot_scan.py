#!/usr/bin/env python
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist
from array import array
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm, kstest, chi2
from pvalue import *

fHists=[]

fColor=[r.kBlack,r.kRed+1,r.kGreen+1,r.kBlue+1,r.kCyan+2,r.kOrange+1,r.kViolet+1,r.kYellow+1,r.kBlack,r.kRed+1,r.kGreen+1,r.kBlue+1,r.kCyan+2,r.kOrange+1,r.kViolet+1,r.kYellow+1]
#fNames=["0.0 0.0","0.0 0.05","0.0 0.1","0.0 0.15","0.0 0.2","0.0 0.25","0.0 0.3","0.0 0.4"]
#fNames=["0.0 0.0","0.05 0.05","0.05 0.1","0.05 0.15","0.05 0.2","0.05 0.25","0.05 0.3","0.05 0.4"]
#fNames=["0.0_0.0","0.0_0.4","0.0_0.8","0.0_1.2","0.0_1.6","0.0_2.0","0.0_2.4","0.0_2.8","0.0_3.2","0.0_3.6"]
#fNames=["0.0_0.0","0.4_0.0","0.8_0.0","1.2_0.0","1.6_0.0","2.0_0.0","2.4_0.0","2.8_0.0","3.2_0.0","3.6_0.0"]
#fNames=["0.0_0.0","0.4_0.4","0.8_0.8","1.2_1.2","1.6_1.6"]#,"0.0 1.8"]
#fNames=["0.0_0.4","0.4_0.4","0.8_0.4","1.2_0.4","1.6_0.4","2.0_0.4","2.4_0.4","2.8_0.4","3.2_0.4","3.6_0.4"]#,"0.0 1.8"]
#fNames =["0.0_0.0","0.4_0.4","0.8_0.8","1.2_1.2","1.6_1.6","2.0_2.0","2.4_2.4","2.8_2.8","3.2_3.2","3.6_3.6"]
#fNames+=["0.2_0.2","0.6_0.6","1.0_1.0","1.4_1.4","1.8_1.8","2.2_2.2","2.6_2.6","3.0_3.0","3.4_3.4","3.8_3.8"]
fNames=[]
for i0 in range(25):
    fNames.append("graph"+str(i0))
    
def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]


def combine(iGraphs):
    lC0 = r.TCanvas("A","A",800,600)
    masses  = array( 'd' )
    pvalues = array( 'd' )

    lN = iGraphs[0].GetN()
    for i0 in range(lN):
        pX = iGraphs[0].GetX()[i0]
        masses.append(pX)
        pvals = []
        pTotal = 0
        pCount=0
        for pGraph in iGraphs:
            pCount = pCount+1
            try:
                pPVal = pGraph.GetY()[i0]
                pvals.append(pPVal)
                pTotal += math.log(pPVal)
            except:
                print("missing",i0,pGraph.GetName())
        pValue = 1-chi2.cdf(-2.*pTotal,2*len(pvals))
        pvalues.append(pValue)
    graph1 = r.TGraph(len(masses),masses,pvalues)
    graph1.SetMarkerStyle(20)
    graph1.GetXaxis().SetTitle("m_{jj} (GeV)")
    graph1.GetYaxis().SetTitle("p^{0} value")
    graph1.SetTitle("")#Significance vs Mass")
    graph1.SetLineColor(2)
    graph1.SetMarkerColor(2)
    graph1.SetLineWidth(2)
    r.gPad.SetLogy(True)
    graph1.GetYaxis().SetRangeUser(1e-12,1.0)
    graph1.Draw("alp")

    lines=[]
    sigmas=[]
    for i0 in range(7):#len(sigmas)):
        sigmas.append(1-norm.cdf(i0+1))
        lLine = r.TLine(masses[0],sigmas[i0],masses[len(masses)-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        lPT = r.TPaveText(3000,sigmas[i0],3500,sigmas[i0]+1.5*sigmas[i0])
        lPT.SetFillStyle(4050)
        lPT.SetFillColor(0)
        lPT.SetBorderSize(0)
        lPT.AddText(str(i0+1)+"#sigma")
        lPT.Draw()
        lines.append(lLine)
        lines.append(lPT)

    graph1.Draw("lp")
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue.png")
    return graph1

def runCombine(iFile,iN):
    lFile = r.TFile(iFile)
    lCan   = r.TCanvas("scan","scan",800,600)
    lCan.SetLogy()
    leg = r.TLegend(0.55,0.17,0.88,0.47)
    leg.SetFillColor(0)
    fHists=[]
    for i0 in range(iN):
        lGraph = r.TGraph(lFile.Get(fNames[i0]))
        lGraph.SetMarkerColor(fColor[i0 % 10])
        lGraph.SetLineColor(fColor[i0 % 10])
        lGraph.SetLineWidth(2)
        lGraph.GetYaxis().SetRangeUser(1e-10,1)
        fHists.append(lGraph)
        if i0 == 0:
            lGraph.Draw("alp")
        else:
            lGraph.Draw("lp")
        leg.AddEntry(lGraph,lGraph.GetTitle(),"lp")
    leg.Draw()
    lCan.Modified()
    lCan.Update()
    lCan.SaveAs(lCan.GetName()+iFile+".png")
    end()
    lCombo = combine(fHists)
    lFile.Close()
    return lCombo
    
if __name__ == "__main__":
    lCombo1 = runCombine("Graphs_out_3_3x3.root",9)
    lCombo2 = runCombine("Graphs_out_3_4x4.root",16)
    lCombo3 = runCombine("Graphs_out_3_5x5.root",25)
    lCombo1.SetLineColor(r.kBlue+1)
    lCombo2.SetLineColor(r.kGreen+1)
    lCombo3.SetLineColor(r.kViolet+1)
    lCombo1.SetMarkerColor(r.kBlue+1)
    lCombo2.SetMarkerColor(r.kGreen+1)
    lCombo3.SetMarkerColor(r.kViolet+1)
    
    lC0 = r.TCanvas("A","A",800,600)
    lC0.SetLogy()
    masses = lCombo1.GetX()
    lines=[]
    sigmas=[]
    lCombo1.Draw("alp")
    for i0 in range(7):#len(sigmas)):
        sigmas.append(1-norm.cdf(i0+1))
        lLine = r.TLine(masses[0],sigmas[i0],masses[len(masses)-1],sigmas[i0])
        lLine.SetLineStyle(r.kDashed)
        lLine.SetLineWidth(2)
        lLine.Draw()
        lPT = r.TPaveText(3000,sigmas[i0],3500,sigmas[i0]+1.5*sigmas[i0])
        lPT.SetFillStyle(4050)
        lPT.SetFillColor(0)
        lPT.SetBorderSize(0)
        lPT.AddText(str(i0+1)+"#sigma")
        lPT.Draw()
        lines.append(lLine)
        lines.append(lPT)
    lLegend = r.TLegend(0.6,0.6,0.9,0.9)
    lLegend.SetBorderSize(0)
    lLegend.SetFillStyle(0)
    lLegend.AddEntry(lCombo1,"3x3 bins","lp")
    lLegend.AddEntry(lCombo2,"4x4 bins","lp")
    lLegend.AddEntry(lCombo3,"5x5 bins","lp")
    lCombo2.Draw("lp")
    lCombo3.Draw("lp")
    lLegend.Draw()
    lC0.Update()
    lC0.Draw()
    lC0.SaveAs("pvalue.png")
    end()
