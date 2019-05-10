#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python script to launch OpenMVG SfM tools on an image dataset
#
# usage : python tutorial_demo.py
#

# Indicate the openMVG binary directory
# TODO - make environmental variables
OPENMVG_SFM_BIN = "/usr/local/bin/"
OPENMVS_SFM_BIN = "/usr/local/bin/OpenMVS/"

# Indicate the openMVG camera sensor width directory
# CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/nism/Documents/csci1430/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

import os
import subprocess
import sys
import numpy as np
import logging

log = logging.getLogger('SafetyNet')


def openMVS_pipe(out_dir, sfm_data, pc_filename):
  depth_dir = os.path.join(out_dir, "depth")
  if not os.path.exists(depth_dir):
    os.mkdir(depth_dir)

  log.info( "5. Pre-process for openMVS")
  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_openMVG2openMVS"),  "-i", sfm_data,
      "-o", os.path.join(depth_dir,"scene.mvs")] )
  pRecons.wait()

  log.info("6. Densify")
  pRecons = subprocess.Popen( [os.path.join(OPENMVS_SFM_BIN,
      "DensifyPointCloud"),  "-i",
      os.path.join(depth_dir, "scene.mvs"),
      "-o", pc_filename] )
  pRecons.wait()

  return pc_filename


def openMVG_pipe(img_dir, out_dir, sfm_data_bin, sfm_data_json):

  matches_dir = os.path.join(out_dir, "matches")
  if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

  log.info("1. Intrinsics analysis")
  pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_SfMInit_ImageListing"),  "-i", img_dir,
      "-o", matches_dir, "-k", "2246.742;0;1920;0;2246.742;1080;0;0;1", # FIXME - parametize
      "-c", "3"] )
  pIntrisics.wait()

  log.info("2a. Compute features")
  pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_ComputeFeatures"),  "-i",
      os.path.join(matches_dir, "sfm_data.json"),
      "-o", matches_dir, "-m", "SIFT", "-f" , "1"] )
  pFeatures.wait()

  log.info("2b. Compute matches")
  pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_ComputeMatches"), "-i",
      os.path.join(matches_dir, "sfm_data.json"),
      "-o", matches_dir, "-f", "1", "-n", "ANNL2"] )
  pMatches.wait()

  log.info("3. Do Incremental/Sequential reconstruction") #set manually the initial pair to avoid the prompt question
  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_IncrementalSfM"),
      "-i", os.path.join(matches_dir, "sfm_data.json"),
      "-m", matches_dir, "-o", out_dir] )
  pRecons.wait()

  # # print ("5. Colorize Structure")
  # # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
  # # pRecons.wait()

  log.info("4. Structure from Known Poses (robust triangulation)")
  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_ComputeStructureFromKnownPoses"),  "-i",
      sfm_data_bin,
      "-m", matches_dir, "-o", os.path.join(out_dir,"robust.ply")] )
  pRecons.wait()

  pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
      "openMVG_main_ConvertSfM_DataFormat"),  "-i",
      sfm_data_bin, "-o", sfm_data_json, "-V", "-I", "-E", "-S"] )
  pRecons.wait()

  return sfm_data_bin, sfm_data_json

  # ## Reconstruction for the global SfM pipeline
  # ## - global SfM pipeline use matches filtered by the essential matrices
  # ## - here we reuse photometric matches and perform only the essential matrix filering
  # print ("2. Compute matches (for the global SfM Pipeline)")
  # # pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-r", "0.8", "-g", "e"] )
  # pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
  #     "openMVG_main_ComputeMatches"), "-i",
  #     os.path.join(matches_dir, "sfm_data.json"),
  #     "-o", matches_dir, "-r", "0.8", "-g", "e"] )
  # pMatches.wait()
  # #
  # #reconstruction_dir = os.path.join(output_dir,"reconstruction_global")
  # #print ("3. Do Global reconstruction")
  # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GlobalSfM"),
  #      "-i", os.path.join(matches_dir, "sfm_data.json"), "-m",
  #      matches_dir, "-o", out_dir] )
  # #pRecons.wait()
  # #
  # #print ("5. Colorize Structure")
  # #pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
  # #pRecons.wait()
  # #
  # #print ("4. Structure from Known Poses (robust triangulation)")
  # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),
  #     "-i", sfm_data_bin, 
  #     "-m", matches_dir, "-o", os.path.join(out_dir,"robust.ply")] )
  # pRecons.wait()

  # pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN,
  #     "openMVG_main_ConvertSfM_DataFormat"),  "-i",
  #     sfm_data_bin, "-o", sfm_data_json, "-V", "-I", "-E", "-S"] )
  # pRecons.wait()

  # return sfm_data_bin, sfm_data_json


