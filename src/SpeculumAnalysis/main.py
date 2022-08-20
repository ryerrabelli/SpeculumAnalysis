# This is a sample Python script.

if __name__ == '__main__':
    print('Hello world')



import os
import os.path

src = '/Users/ryerrabelli/Library/CloudStorage/GoogleDrive-ryerrabelli@gmail.com/My Drive/Computer Backups/Rahul Yerrabelli drive/Academics/UIUC/UIUC ECs/Rahul_Ashkhan_Projects/SpeculumProjects_Shared/Experiments/V2/Photos_Rahul/Key/'
files = os.listdir( src )
day = 2

for ind in range(len(files)):
    filename_old = files[ind]
    filename_old = files[ind].split("-")[-1]
    old_no_ext, ext = os.path.splitext ( filename_old )
    # f-string with leading zeros
    filename_new = f"{day:02}_{(ind+1):03}-{filename_old}"
    print(filename_new)
    os.rename( os.path.join( src, filename_old ), os.path.join( src, filename_new ))