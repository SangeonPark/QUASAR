#!/usr/bin/env python
import ROOT as r,sys,math,array,os
from optparse import OptionParser
from ROOT import std,RooDataHist
from array import array
import numpy as np
from scipy.stats import poisson, norm, kstest
from pvalue import *

fHists=[]

def end():
    if __name__ == '__main__':
        rep = ''
        while not rep in [ 'q', 'Q','a',' ' ]:
            rep = raw_input( 'enter "q" to quit: ' )
            if 1 < len(rep):
                rep = rep[0]


def makeHist(iName,iCut,iBBTree,iBkgTree):
    lData1 = r.TH1F("bbhist"+iName,"bbhist"+iName,40,2500,10000)
    lBkg1  = r.TH1F("bkhist"+iName,"bkhist"+iName,40,2500,10000) 

    #cut="loss2 > 5.5 && loss1 < 10.0"
    iBBTree .Draw("mass>>bbhist"+iName,iCut)
    iBkgTree.Draw("mass>>bkhist"+iName,iCut)
    lBkg1.Scale(lData1.Integral()/lBkg1.Integral())
    lData1.SetMarkerStyle(r.kFullCircle)
    lBkg1.SetMarkerStyle(r.kFullCircle)
    lBkg1.SetLineColor(r.kRed)
    lBkg1.SetMarkerColor(r.kRed)
    lData1.GetXaxis().SetTitle("m_{jj} (GeV)")
    lBkg1.GetXaxis().SetTitle("m_{jj} (GeV)")
    lData1.GetYaxis().SetTitle("N")
    lBkg1.GetYaxis().SetTitle("N")
    lData1.SetTitle("")
    lBkg1.SetTitle("")
    return lData1,lBkg1 
                
if __name__ == "__main__":#blackbox2-CutFromMap.root
    r.gStyle.SetOptStat(0)
    lBBFile = r.TFile("bb.root")
    lBBTree = lBBFile.Get("output")

    lBkgFile = r.TFile("bkg.root")
    lBkgTree = lBkgFile.Get("output")
    label=[]
    cuts=[]
    lData,lBkg = makeHist("A","mass > 0",lBBTree,lBkgTree)
    lLegend = r.TLegend(0.65,0.65,0.9,0.9)
    lLegend.AddEntry(lData,"Toy Data","pe")
    lLegend.AddEntry(lBkg,"MC","l")    
    lData.Draw("ep")
    lBkg.Draw("hist sames")
    lLegend.Draw()
    end()
