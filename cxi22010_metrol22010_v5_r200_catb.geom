; OY: From this version individual panels are optimized
; OY: this is the same as r193, just moved by 0.3, -0.7 px
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Manually optimized with hdfsee
; Manually optimized with hdfsee
photon_energy = /LCLS/photon_energy_eV
res = 9097.52
adu_per_eV = 0.00338
;clen = 0.0775
;clen = 0.1425
clen =  /LCLS/detector_1/EncoderValue
;clen = 0.1425
;catb good coffset = 0.57775
coffset = 0.57783

data = /entry_1/data_1/data
dim0 = %
dim1 = ss
dim2 = fs

mask = /entry_1/data_1/mask
mask_good = 0x0000
mask_bad = 0xffff

; The following lines define "rigid groups" which express the physical
; construction of the detector.  This is used when refining the detector
; geometry.

rigid_group_q0 = q0a0,q0a1,q0a2,q0a3,q0a4,q0a5,q0a6,q0a7,q0a8,q0a9,q0a10,q0a11,q0a12,q0a13,q0a14,q0a15
rigid_group_q1 = q1a0,q1a1,q1a2,q1a3,q1a4,q1a5,q1a6,q1a7,q1a8,q1a9,q1a10,q1a11,q1a12,q1a13,q1a14,q1a15
rigid_group_q2 = q2a0,q2a1,q2a2,q2a3,q2a4,q2a5,q2a6,q2a7,q2a8,q2a9,q2a10,q2a11,q2a12,q2a13,q2a14,q2a15
rigid_group_q3 = q3a0,q3a1,q3a2,q3a3,q3a4,q3a5,q3a6,q3a7,q3a8,q3a9,q3a10,q3a11,q3a12,q3a13,q3a14,q3a15

rigid_group_a0 = q0a0,q0a1
rigid_group_a1 = q0a2,q0a3
rigid_group_a2 = q0a4,q0a5
rigid_group_a3 = q0a6,q0a7
rigid_group_a4 = q0a8,q0a9
rigid_group_a5 = q0a10,q0a11
rigid_group_a6 = q0a12,q0a13
rigid_group_a7 = q0a14,q0a15
rigid_group_a8 = q1a0,q1a1
rigid_group_a9 = q1a2,q1a3
rigid_group_a10 = q1a4,q1a5
rigid_group_a11 = q1a6,q1a7
rigid_group_a12 = q1a8,q1a9
rigid_group_a13 = q1a10,q1a11
rigid_group_a14 = q1a12,q1a13
rigid_group_a15 = q1a14,q1a15
rigid_group_a16 = q2a0,q2a1
rigid_group_a17 = q2a2,q2a3
rigid_group_a18 = q2a4,q2a5
rigid_group_a19 = q2a6,q2a7
rigid_group_a20 = q2a8,q2a9
rigid_group_a21 = q2a10,q2a11
rigid_group_a22 = q2a12,q2a13
rigid_group_a23 = q2a14,q2a15
rigid_group_a24 = q3a0,q3a1
rigid_group_a25 = q3a2,q3a3
rigid_group_a26 = q3a4,q3a5
rigid_group_a27 = q3a6,q3a7
rigid_group_a28 = q3a8,q3a9
rigid_group_a29 = q3a10,q3a11
rigid_group_a30 = q3a12,q3a13
rigid_group_a31 = q3a14,q3a15

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31

; -----------------------------------------------

q0a0/min_fs = 0
q0a0/min_ss = 0
q0a0/max_fs = 193
q0a0/max_ss = 184
q0a0/fs = -0.002756x +0.999997y
q0a0/ss = -0.999997x -0.002756y
q0a0/corner_x = 425.092000
q0a0/corner_y = -11.405100
q0a0/no_index = 0

q0a1/min_fs = 194
q0a1/min_ss = 0
q0a1/max_fs = 387
q0a1/max_ss = 184
q0a1/fs = -0.002756x +0.999997y
q0a1/ss = -0.999997x -0.002756y
q0a1/corner_x = 424.549000
q0a1/corner_y = 185.594000
q0a1/no_index = 0

q0a2/min_fs = 0
q0a2/min_ss = 185
q0a2/max_fs = 193
q0a2/max_ss = 369
q0a2/fs = +0.002451x +0.999998y
q0a2/ss = -0.999998x +0.002451y
q0a2/corner_x = 211.160000
q0a2/corner_y = -11.980700
q0a2/no_index = 0

q0a3/min_fs = 194
q0a3/min_ss = 185
q0a3/max_fs = 387
q0a3/max_ss = 369
q0a3/fs = +0.002451x +0.999998y
q0a3/ss = -0.999998x +0.002451y
q0a3/corner_x = 211.643000
q0a3/corner_y = 185.018000
q0a3/no_index = 0

q0a4/min_fs = 0
q0a4/min_ss = 370
q0a4/max_fs = 193
q0a4/max_ss = 554
q0a4/fs = -0.999995x -0.003237y
q0a4/ss = +0.003236x -0.999995y
q0a4/corner_x = 846.412000
q0a4/corner_y = 387.851000
q0a4/no_index = 0

q0a5/min_fs = 194
q0a5/min_ss = 370
q0a5/max_fs = 387
q0a5/max_ss = 554
q0a5/fs = -0.999995x -0.003237y
q0a5/ss = +0.003236x -0.999995y
q0a5/corner_x = 649.414000
q0a5/corner_y = 387.213000
q0a5/no_index = 0

q0a6/min_fs = 0
q0a6/min_ss = 555
q0a6/max_fs = 193
q0a6/max_ss = 739
q0a6/fs = -0.999999x +0.001512y
q0a6/ss = -0.001512x -0.999999y
q0a6/corner_x = 847.062000
q0a6/corner_y = 174.729000
q0a6/no_index = 0

q0a7/min_fs = 194
q0a7/min_ss = 555
q0a7/max_fs = 387
q0a7/max_ss = 739
q0a7/fs = -0.999999x +0.001512y
q0a7/ss = -0.001512x -0.999999y
q0a7/corner_x = 650.062000
q0a7/corner_y = 175.027000
q0a7/no_index = 0

q0a8/min_fs = 0
q0a8/min_ss = 740
q0a8/max_fs = 193
q0a8/max_ss = 924
q0a8/fs = +0.000225x -1.000000y
q0a8/ss = +1.000000x +0.000225y
q0a8/corner_x = 450.580000
q0a8/corner_y = 809.672000
q0a8/no_index = 0

q0a9/min_fs = 194
q0a9/min_ss = 740
q0a9/max_fs = 387
q0a9/max_ss = 924
q0a9/fs = +0.000225x -1.000000y
q0a9/ss = +1.000000x +0.000225y
q0a9/corner_x = 450.625000
q0a9/corner_y = 612.672000
q0a9/no_index = 0

q0a10/min_fs = 194
q0a10/min_ss = 1295
q0a10/max_fs = 387
q0a10/max_ss = 1479
q0a10/fs = -0.000228x -1.000000y
q0a10/ss = +1.000000x -0.000228y
q0a10/corner_x = 662.820000
q0a10/corner_y = 809.933000
q0a10/no_index = 0

q0a11/min_fs = 0
q0a11/min_ss = 1295
q0a11/max_fs = 193
q0a11/max_ss = 1479
q0a11/fs = -0.000228x -1.000000y
q0a11/ss = +1.000000x -0.000228y
q0a11/corner_x = 662.777000
q0a11/corner_y = 612.932000
q0a11/no_index = 0

q0a12/min_fs = 0
q0a12/min_ss = 925
q0a12/max_fs = 193
q0a12/max_ss = 1109
q0a12/fs = -0.999994x +0.003649y
q0a12/ss = -0.003649x -0.999994y
q0a12/corner_x = 421.606000
q0a12/corner_y = 789.939000
q0a12/no_index = 0

q0a13/min_fs = 194
q0a13/min_ss = 925
q0a13/max_fs = 387
q0a13/max_ss = 1109
q0a13/fs = -0.999994x +0.003649y
q0a13/ss = -0.003649x -0.999994y
q0a13/corner_x = 224.607000
q0a13/corner_y = 790.658000
q0a13/no_index = 0

q0a14/min_fs = 0
q0a14/min_ss = 1110
q0a14/max_fs = 193
q0a14/max_ss = 1294
q0a14/fs = -0.999955x +0.009444y
q0a14/ss = -0.009444x -0.999955y
q0a14/corner_x = 421.060000
q0a14/corner_y = 576.228000
q0a14/no_index = 0

q0a15/min_fs = 194
q0a15/min_ss = 1110
q0a15/max_fs = 387
q0a15/max_ss = 1294
q0a15/fs = -0.999955x +0.009444y
q0a15/ss = -0.009444x -0.999955y
q0a15/corner_x = 224.069000
q0a15/corner_y = 578.088000
q0a15/no_index = 0

q1a0/min_fs = 388
q1a0/min_ss = 0
q1a0/max_fs = 581
q1a0/max_ss = 184
q1a0/fs = -1.000000x +0.000521y
q1a0/ss = -0.000507x -1.000000y
q1a0/corner_x = 13.473100
q1a0/corner_y = 421.652000
q1a0/no_index = 0

q1a1/min_fs = 582
q1a1/min_ss = 0
q1a1/max_fs = 775
q1a1/max_ss = 184
q1a1/fs = -1.000000x +0.000521y
q1a1/ss = -0.000507x -1.000000y
q1a1/corner_x = -183.526000
q1a1/corner_y = 421.755000
q1a1/no_index = 0

q1a2/min_fs = 388
q1a2/min_ss = 185
q1a2/max_fs = 581
q1a2/max_ss = 369
q1a2/fs = -0.999997x -0.002285y
q1a2/ss = +0.002301x -0.999997y
q1a2/corner_x = 13.943900
q1a2/corner_y = 208.711000
q1a2/no_index = 0

q1a3/min_fs = 582
q1a3/min_ss = 185
q1a3/max_fs = 775
q1a3/max_ss = 369
q1a3/fs = -0.999997x -0.002285y
q1a3/ss = +0.002301x -0.999997y
q1a3/corner_x = -183.055000
q1a3/corner_y = 208.260000
q1a3/no_index = 0

q1a4/min_fs = 388
q1a4/min_ss = 370
q1a4/max_fs = 581
q1a4/max_ss = 554
q1a4/fs = +0.000679x -1.000000y
q1a4/ss = +1.000000x +0.000692y
q1a4/corner_x = -388.035000
q1a4/corner_y = 842.758000
q1a4/no_index = 0

q1a5/min_fs = 582
q1a5/min_ss = 370
q1a5/max_fs = 775
q1a5/max_ss = 554
q1a5/fs = +0.000679x -1.000000y
q1a5/ss = +1.000000x +0.000692y
q1a5/corner_x = -387.902000
q1a5/corner_y = 645.761000
q1a5/no_index = 0

q1a6/min_fs = 388
q1a6/min_ss = 555
q1a6/max_fs = 581
q1a6/max_ss = 739
q1a6/fs = +0.000128x -1.000000y
q1a6/ss = +1.000000x +0.000145y
q1a6/corner_x = -174.520000
q1a6/corner_y = 843.568000
q1a6/no_index = 0

q1a7/min_fs = 582
q1a7/min_ss = 555
q1a7/max_fs = 775
q1a7/max_ss = 739
q1a7/fs = +0.000128x -1.000000y
q1a7/ss = +1.000000x +0.000145y
q1a7/corner_x = -174.496000
q1a7/corner_y = 646.571000
q1a7/no_index = 0

q1a8/min_fs = 388
q1a8/min_ss = 740
q1a8/max_fs = 581
q1a8/max_ss = 924
q1a8/fs = +1.000000x +0.000741y
q1a8/ss = -0.000760x +1.000000y
q1a8/corner_x = -806.578000
q1a8/corner_y = 440.729000
q1a8/no_index = 0

q1a9/min_fs = 582
q1a9/min_ss = 740
q1a9/max_fs = 775
q1a9/max_ss = 924
q1a9/fs = +1.000000x +0.000741y
q1a9/ss = -0.000760x +1.000000y
q1a9/corner_x = -609.579000
q1a9/corner_y = 440.875000
q1a9/no_index = 0

q1a10/min_fs = 388
q1a10/min_ss = 925
q1a10/max_fs = 581
q1a10/max_ss = 1109
q1a10/fs = +1.000000x +0.000582y
q1a10/ss = -0.000596x +1.000000y
q1a10/corner_x = -807.571000
q1a10/corner_y = 654.169000
q1a10/no_index = 0

q1a11/min_fs = 582
q1a11/min_ss = 925
q1a11/max_fs = 775
q1a11/max_ss = 1109
q1a11/fs = +1.000000x +0.000582y
q1a11/ss = -0.000596x +1.000000y
q1a11/corner_x = -610.572000
q1a11/corner_y = 654.283000
q1a11/no_index = 0

q1a12/min_fs = 388
q1a12/min_ss = 1110
q1a12/max_fs = 581
q1a12/max_ss = 1294
q1a12/fs = +0.003882x -0.999993y
q1a12/ss = +0.999992x +0.003900y
q1a12/corner_x = -788.465000
q1a12/corner_y = 415.231000
q1a12/no_index = 0

q1a13/min_fs = 582
q1a13/min_ss = 1110
q1a13/max_fs = 775
q1a13/max_ss = 1294
q1a13/fs = +0.003882x -0.999993y
q1a13/ss = +0.999992x +0.003900y
q1a13/corner_x = -787.701000
q1a13/corner_y = 218.234000
q1a13/no_index = 0

q1a14/min_fs = 388
q1a14/min_ss = 1295
q1a14/max_fs = 581
q1a14/max_ss = 1479
q1a14/fs = +0.002393x -0.999997y
q1a14/ss = +0.999997x +0.002412y
q1a14/corner_x = -574.976000
q1a14/corner_y = 415.783000
q1a14/no_index = 0

q1a15/min_fs = 582
q1a15/min_ss = 1295
q1a15/max_fs = 775
q1a15/max_ss = 1479
q1a15/fs = +0.002393x -0.999997y
q1a15/ss = +0.999997x +0.002412y
q1a15/corner_x = -574.504000
q1a15/corner_y = 218.786000
q1a15/no_index = 0

q2a0/min_fs = 776
q2a0/min_ss = 0
q2a0/max_fs = 969
q2a0/max_ss = 184
q2a0/fs = +0.001640x -0.999998y
q2a0/ss = +0.999998x +0.001640y
q2a0/corner_x = -429.355000
q2a0/corner_y = 9.123800
q2a0/no_index = 0

q2a1/min_fs = 970
q2a1/min_ss = 0
q2a1/max_fs = 1163
q2a1/max_ss = 184
q2a1/fs = +0.001640x -0.999998y
q2a1/ss = +0.999998x +0.001640y
q2a1/corner_x = -429.031000
q2a1/corner_y = -187.875000
q2a1/no_index = 0

q2a2/min_fs = 776
q2a2/min_ss = 185
q2a2/max_fs = 969
q2a2/max_ss = 369
q2a2/fs = -0.004059x -0.999992y
q2a2/ss = +0.999992x -0.004059y
q2a2/corner_x = -215.713000
q2a2/corner_y = 9.563900
q2a2/no_index = 0

q2a3/min_fs = 970
q2a3/min_ss = 185
q2a3/max_fs = 1163
q2a3/max_ss = 369
q2a3/fs = -0.004059x -0.999992y
q2a3/ss = +0.999992x -0.004059y
q2a3/corner_x = -216.512000
q2a3/corner_y = -187.434000
q2a3/no_index = 0

q2a4/min_fs = 776
q2a4/min_ss = 370
q2a4/max_fs = 969
q2a4/max_ss = 554
q2a4/fs = +0.999994x +0.003385y
q2a4/ss = -0.003384x +0.999994y
q2a4/corner_x = -846.242000
q2a4/corner_y = -393.039000
q2a4/no_index = 0

q2a5/min_fs = 970
q2a5/min_ss = 370
q2a5/max_fs = 1163
q2a5/max_ss = 554
q2a5/fs = +0.999994x +0.003385y
q2a5/ss = -0.003384x +0.999994y
q2a5/corner_x = -649.244000
q2a5/corner_y = -392.372000
q2a5/no_index = 0

q2a6/min_fs = 776
q2a6/min_ss = 555
q2a6/max_fs = 969
q2a6/max_ss = 739
q2a6/fs = +0.999982x +0.005959y
q2a6/ss = -0.005959x +0.999982y
q2a6/corner_x = -847.155000
q2a6/corner_y = -181.119000
q2a6/no_index = 0

q2a7/min_fs = 970
q2a7/min_ss = 555
q2a7/max_fs = 1163
q2a7/max_ss = 739
q2a7/fs = +0.999982x +0.005959y
q2a7/ss = -0.005959x +0.999982y
q2a7/corner_x = -650.159000
q2a7/corner_y = -179.945000
q2a7/no_index = 0

q2a8/min_fs = 776
q2a8/min_ss = 740
q2a8/max_fs = 969
q2a8/max_ss = 924
q2a8/fs = -0.003259x +0.999995y
q2a8/ss = -0.999995x -0.003259y
q2a8/corner_x = -449.793000
q2a8/corner_y = -811.293000
q2a8/no_index = 0

q2a9/min_fs = 970
q2a9/min_ss = 740
q2a9/max_fs = 1163
q2a9/max_ss = 924
q2a9/fs = -0.003259x +0.999995y
q2a9/ss = -0.999995x -0.003259y
q2a9/corner_x = -450.436000
q2a9/corner_y = -614.294000
q2a9/no_index = 0

q2a10/min_fs = 776
q2a10/min_ss = 925
q2a10/max_fs = 969
q2a10/max_ss = 1109
q2a10/fs = -0.004198x +0.999991y
q2a10/ss = -0.999991x -0.004198y
q2a10/corner_x = -661.863000
q2a10/corner_y = -812.637000
q2a10/no_index = 0

q2a11/min_fs = 970
q2a11/min_ss = 925
q2a11/max_fs = 1163
q2a11/max_ss = 1109
q2a11/fs = -0.004198x +0.999991y
q2a11/ss = -0.999991x -0.004198y
q2a11/corner_x = -662.690000
q2a11/corner_y = -615.639000
q2a11/no_index = 0

q2a12/min_fs = 776
q2a12/min_ss = 1110
q2a12/max_fs = 969
q2a12/max_ss = 1294
q2a12/fs = +0.999998x -0.002091y
q2a12/ss = +0.002092x +0.999998y
q2a12/corner_x = -420.844000
q2a12/corner_y = -791.931000
q2a12/no_index = 0

q2a13/min_fs = 970
q2a13/min_ss = 1110
q2a13/max_fs = 1163
q2a13/max_ss = 1294
q2a13/fs = +0.999998x -0.002091y
q2a13/ss = +0.002092x +0.999998y
q2a13/corner_x = -223.844000
q2a13/corner_y = -792.343000
q2a13/no_index = 0

q2a14/min_fs = 776
q2a14/min_ss = 1295
q2a14/max_fs = 969
q2a14/max_ss = 1479
q2a14/fs = +0.999999x -0.000390y
q2a14/ss = +0.000390x +0.999999y
q2a14/corner_x = -420.018000
q2a14/corner_y = -579.465000
q2a14/no_index = 0

q2a15/min_fs = 970
q2a15/min_ss = 1295
q2a15/max_fs = 1163
q2a15/max_ss = 1479
q2a15/fs = +0.999999x -0.000390y
q2a15/ss = +0.000390x +0.999999y
q2a15/corner_x = -223.018000
q2a15/corner_y = -579.542000
q2a15/no_index = 0

q3a0/min_fs = 1164
q3a0/min_ss = 0
q3a0/max_fs = 1357
q3a0/max_ss = 184
q3a0/fs = +0.999998x +0.001818y
q3a0/ss = -0.001818x +0.999998y
q3a0/corner_x = -13.782200
q3a0/corner_y = -423.637000
q3a0/no_index = 0

q3a1/min_fs = 1358
q3a1/min_ss = 0
q3a1/max_fs = 1551
q3a1/max_ss = 184
q3a1/fs = +0.999998x +0.001818y
q3a1/ss = -0.001818x +0.999998y
q3a1/corner_x = 183.217000
q3a1/corner_y = -423.280000
q3a1/no_index = 0

q3a2/min_fs = 1164
q3a2/min_ss = 185
q3a2/max_fs = 1357
q3a2/max_ss = 369
q3a2/fs = +0.999993x -0.003642y
q3a2/ss = +0.003642x +0.999993y
q3a2/corner_x = -13.998800
q3a2/corner_y = -209.599000
q3a2/no_index = 0

q3a3/min_fs = 1358
q3a3/min_ss = 185
q3a3/max_fs = 1551
q3a3/max_ss = 369
q3a3/fs = +0.999993x -0.003642y
q3a3/ss = +0.003642x +0.999993y
q3a3/corner_x = 182.999000
q3a3/corner_y = -210.316000
q3a3/no_index = 0

q3a4/min_fs = 1164
q3a4/min_ss = 370
q3a4/max_fs = 1357
q3a4/max_ss = 554
q3a4/fs = +0.001343x +0.999999y
q3a4/ss = -0.999999x +0.001342y
q3a4/corner_x = 384.900000
q3a4/corner_y = -846.603000
q3a4/no_index = 0

q3a5/min_fs = 1358
q3a5/min_ss = 370
q3a5/max_fs = 1551
q3a5/max_ss = 554
q3a5/fs = +0.001343x +0.999999y
q3a5/ss = -0.999999x +0.001342y
q3a5/corner_x = 385.165000
q3a5/corner_y = -649.603000
q3a5/no_index = 0

q3a6/min_fs = 1164
q3a6/min_ss = 555
q3a6/max_fs = 1357
q3a6/max_ss = 739
q3a6/fs = +0.002968x +0.999996y
q3a6/ss = -0.999996x +0.002967y
q3a6/corner_x = 171.910000
q3a6/corner_y = -846.786000
q3a6/no_index = 0

q3a7/min_fs = 1358
q3a7/min_ss = 555
q3a7/max_fs = 1551
q3a7/max_ss = 739
q3a7/fs = +0.002968x +0.999996y
q3a7/ss = -0.999996x +0.002967y
q3a7/corner_x = 172.495000
q3a7/corner_y = -649.786000
q3a7/no_index = 0

q3a8/min_fs = 1358
q3a8/min_ss = 1295
q3a8/max_fs = 1551
q3a8/max_ss = 1479
q3a8/fs = -1.000000x -0.000599y
q3a8/ss = +0.000598x -1.000000y
q3a8/corner_x = 805.417000
q3a8/corner_y = -449.200000
q3a8/no_index = 0

q3a9/min_fs = 1164
q3a9/min_ss = 1295
q3a9/max_fs = 1357
q3a9/max_ss = 1479
q3a9/fs = -1.000000x -0.000599y
q3a9/ss = +0.000598x -1.000000y
q3a9/corner_x = 608.417000
q3a9/corner_y = -449.317000
q3a9/no_index = 0

q3a10/min_fs = 1164
q3a10/min_ss = 740
q3a10/max_fs = 1357
q3a10/max_ss = 924
q3a10/fs = -0.999963x +0.008673y
q3a10/ss = -0.008672x -0.999963y
q3a10/corner_x = 807.616000
q3a10/corner_y = -663.951000
q3a10/no_index = 0

q3a11/min_fs = 1358
q3a11/min_ss = 740
q3a11/max_fs = 1551
q3a11/max_ss = 924
q3a11/fs = -0.999963x +0.008673y
q3a11/ss = -0.008672x -0.999963y
q3a11/corner_x = 610.625000
q3a11/corner_y = -662.243000
q3a11/no_index = 0

q3a12/min_fs = 1164
q3a12/min_ss = 1110
q3a12/max_fs = 1357
q3a12/max_ss = 1294
q3a12/fs = +0.008256x +0.999966y
q3a12/ss = -0.999966x +0.008258y
q3a12/corner_x = 785.143000
q3a12/corner_y = -420.263000
q3a12/no_index = 0

q3a13/min_fs = 1358
q3a13/min_ss = 1110
q3a13/max_fs = 1551
q3a13/max_ss = 1294
q3a13/fs = +0.008256x +0.999966y
q3a13/ss = -0.999966x +0.008258y
q3a13/corner_x = 786.768000
q3a13/corner_y = -223.269000
q3a13/no_index = 0

q3a14/min_fs = 1164
q3a14/min_ss = 925
q3a14/max_fs = 1357
q3a14/max_ss = 1109
q3a14/fs = +0.002383x +0.999997y
q3a14/ss = -0.999997x +0.002382y
q3a14/corner_x = 574.457000
q3a14/corner_y = -420.896000
q3a14/no_index = 0

q3a15/min_fs = 1358
q3a15/min_ss = 925
q3a15/max_fs = 1551
q3a15/max_ss = 1109
q3a15/fs = +0.002383x +0.999997y
q3a15/ss = -0.999997x +0.002382y
q3a15/corner_x = 574.925000
q3a15/corner_y = -223.897000
q3a15/no_index = 0
