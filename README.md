PALACE was aimed to align the pockets of the frames and then cluster the pocket-aligned ligand coordinates. 

The full function could be performed using the script "PALACE.py". 

Before running the PALACE, you have to fix some of the details here: 

Step (0): please make sure you have installed all of packages in need at the beginng of: MDAnalysis, torch, shutil......................

Step (1): You should put the topology file and trajectory file in a fold, and write the pathway of your fold in line 12, write the file name of your top and trajectory in line 20. 
          (Here using the MDAnalysis, fit to the Amber, Gromacs, CHARMM..............as: https://docs.mdanalysis.org/stable/index.html) 

Step (2): You have to know how many frames are in your trajectory, and change the lines 27, 32, and 55. In the pilot test, there are 10000 frames total. 

Step (3): Select the pocket area, which would be aligned in line 81 and 88, from x to y should be changed into: x:y as shown in the script. 

Step (4): Stop here and check the aligned pdb, select which atoms you want to use for the clustering, and put the atoms' lines into the script in the line 113 and 114. 

Step (5): How many clusters you want to get?: Line 237,  "k_range = range(2, 31)" means from 2 to 31, including 2 and not including 31, you can change it into your number. 
