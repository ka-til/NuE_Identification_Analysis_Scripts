# ===================================================================================================
# Author:       Akanksha Katil <katil@ualberta.ca>
# Date:         2026-08-20
# Project:      Electron Neutrino Identification in IceCube Upgrade
# Description:  Extracts frame variables and applies 2 BDT models to get the classification score.
# Software:     Install Icetray and scikit-learn to run this module.
# ===================================================================================================

#import necessary NuE modules
import extract_variables
import apply_BDT_model_to_i3

#import general modules
import sys
import argparse
import numpy as np
import os

#import IceTray modules
from icecube import icetray, dataio

class NuEClassificationModule(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)

        self.AddParameter("GCDFile", 
                          "Path to the GCD file", 
                          "/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2")
        self.AddOutBox("OutBox")

    def Configure(self):
        self.gcd_file = self.GetParameter("GCDFile")

        self.gcd_infile = dataio.I3File(self.gcd_file)

        f_geo = self.gcd_infile
        geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
        self.geo = geo_frame['I3Geometry']
        
        self.ux, self.uy, self.uz = extract_variables.get_mean_upgrade_positions(self.geo)

    def Physics(self, frame):
        # Apply the necessary variable extraction and BDT scoring

        if not extract_variables.upgrade_events(frame, 
                                                mean_x=self.ux, 
                                                mean_y=self.uy, 
                                                mean_z=self.uz):
            return

        extract_variables.reco_time(frame, geo=self.geo) ##Run this only if the reco time is not already present in the frame.

        
        if not extract_variables.get_var(frame, geo=self.geo):
            return

        apply_BDT_model_to_i3.bdt_score(frame)

        self.PushFrame(frame)