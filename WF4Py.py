#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import sys

#SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
#sys.path.append(SCRIPT_DIR)

import WFutils as utils

from waveform_models.WFclass_definition import NewtInspiral
from waveform_models.TaylorF2_Restricted import TaylorF2_RestrictedPN
from waveform_models.IMRPhenomD import IMRPhenomD
from waveform_models.IMRPhenomD_NRTidalv2 import IMRPhenomD_NRTidalv2
from waveform_models.IMRPhenomHM import IMRPhenomHM
from waveform_models.IMRPhenomNSBH import IMRPhenomNSBH
from waveform_models.IMRPhenomXAS import IMRPhenomXAS
