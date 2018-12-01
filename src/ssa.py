#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 00:52:37 2018

@author: kacper
"""

from pyts.decomposition import SSA
h1 = result['004']['h1']
valid_sig = h1.values.reshape(-1,1)[70000:88000]
ssa = SSA(window_size = 100)
X = ssa.fit_transform(valid_sig.T)