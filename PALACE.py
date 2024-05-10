import os
import MDAnalysis as mda
from MDAnalysis.analysis import align
import time
import csv
import torch
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import numpy as np

path = 'pathway/'
folders = [path + "Extraction", path + "Superimpose", path + "k-means"]


###########Calculate the average Strucutre During the MD ##########################
start_average = time.perf_counter()

# Load topology files and trajectory files
u = mda.Universe(path + 'com.top', path + 'md_complex.nc')

# Select all atoms
all_atoms = u.select_atoms('all')

# Initialize an array to store the coordinates of all frames
n_atoms = all_atoms.n_atoms
n_frames = 10000 
average_structure = np.zeros((n_atoms, 3))

# Traverse each frame to ensure that it contains the 9999th frame. 
# The range of Python is closed on the left and open on the right.
for ts in u.trajectory[0:10000]:
    average_structure += all_atoms.positions

average_structure /= n_frames

# Create a new group of atoms containing the average structure
average_atoms = mda.AtomGroup(all_atoms.indices, u)
average_atoms.positions = average_structure

# Save average structure to file
average_atoms.write(path + 'average_structure.pdb')   

end_average = time.perf_counter()
print('Average Structure Calculation consumes：{} s'.format(end__average-start_average))



###########Extract the Strucutres from the trajectory ##########################

start_extraction = time.perf_counter()

#Extraction, casue the first frame in the MDAnlysis is 0 rather than 1, 
#so here the extraction starts from the No. 0 frame, 0.01 ns, totally 10000 frames
for i in range(0, 10000):
    j = i +1
    # read top and NetCDF
    u = mda.Universe(path + 'com.top', 'md_complex.nc')
    # Select the frame
    u.trajectory[i]

    # save as pdb
    u.atoms.write(f'{path}Extraction/{j}.pdb')


end_extraction = time.perf_counter()
print('Extraction consumes：{} s'.format(end_extraction-start_extraction))


###########Aligan the Strucutres ##########################

# Run the superposing
start_superposing = time.perf_counter()

# Set the input folder, output folder and reference strucutre (initial docking conformation here)
input_folder = path + 'Extraction'
output_folder = path + 'Superimpose'
ref_pdb = path + 'average_structure.pdb' 


# load reference strcture##################Need to fix to your own systems####################################
ref = mda.Universe(ref_pdb)
ref_atoms = ref.select_atoms('backbone and (resid 1:156)')

# Iterate through all PDB files in the input folder
for pdb_file in os.listdir(input_folder):
    if pdb_file.endswith('.pdb'):
        mobile_path = os.path.join(input_folder, pdb_file)
        mobile = mda.Universe(mobile_path)
        mobile_atoms = mobile.select_atoms('backbone and (resid 1:156)')

        # Superpose
        align.alignto(mobile_atoms, ref_atoms)

        # Save the superposed structure into fold
        output_filename = '0' + pdb_file
        output_path = os.path.join(output_folder, output_filename)
        mobile.atoms.write(output_path)

end_superposing = time.perf_counter()
print('Alignment consumes：{} s'.format(end_superposing - start_superposing))



###########Extract the normalised coordinates of ligand ##########################

start_cor = time.perf_counter()

# Specify the folder path
folder_path = path + "Superimpose"
output_csv_path = path + "/k-means/all_atom_coordinates.csv"

# Specify the range of line numbers to read
###########NEED to fix to your systems
start_line = 4150
end_line = 4177

# Function to read specific lines from a file
def read_specific_lines(file, start, end):
    lines = []
    with open(file, 'r') as f:
        for current_line_number, line in enumerate(f, start=1):
            if start <= current_line_number <= end:
                lines.append(line)
            elif current_line_number > end:
                break
    return lines

# Initialize column headers and row data
column_headers = None
rows = []

# Iterate through all PDB files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".pdb"):
        # Remove the .pdb extension from the filename
        clean_filename = filename[:-4]  # This slices off the last 4 characters, ".pdb"
        file_path = os.path.join(folder_path, clean_filename + ".pdb")
        specific_lines = read_specific_lines(file_path, start_line, end_line)

        # Set column headers only for the first file
        if column_headers is None:
            column_headers = []
            for line in specific_lines:
                if line.startswith("ATOM"):
                    parts = line.split()
                    atom_type = parts[2]  # Atom type is typically the 3rd element in each line
                    column_headers.extend([f"{atom_type}_x", f"{atom_type}_y", f"{atom_type}_z"])
        
        # Extract the coordinates of each atom
        column_values = []
        for line in specific_lines:
            if line.startswith("ATOM"):
                parts = line.split()
                x, y, z = parts[6:9]  # Coordinates are typically the 7th, 8th, and 9th elements in each line
                column_values.extend([x, y, z])
        
        # Add row data (clean filename without .pdb and coordinate values)
        rows.append([clean_filename] + column_values)

# Write the results to a CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write column headers (first column is for filenames)
    csvwriter.writerow(["Filename"] + column_headers)
    # Write row data
    for row in rows:
        csvwriter.writerow(row)

folders_tem = [path + "Extraction", path + "Superimpose"]

for folder in folders_tem:
    shutil.rmtree(folder, ignore_errors=True)

end_cor = time.perf_counter()
print('Coordinates Reading consumes: {} s'.format(end_cor - start_cor))

###########Perform the k-means ##########################


start_cluster = time.perf_counter()

class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True, device=torch.device("cpu")):
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            sample_tensor = torch.tensor(sample).to(self.device)  
            dist = torch.sum(torch.mul(sample_tensor - self.centers, sample_tensor - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        self.representative_samples = torch.argmin(self.dists, (0))

def sf_kmeans(matrix, device):
    k_range = range(2, 31)  # range of k
    wcss_list = []

    output_dir = path + "k-means"  

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for k in k_range:
        kmeans = KMEANS(n_clusters=k, max_iter=None, verbose=False, device=device)
        
        # The first colume equal to Filename
        names = data.iloc[:, 0].values
        kmeans.fit(matrix)
        
        # Calculate WCSS into the list
        wcss = torch.mean(kmeans.dists).item() / k  # Calculate WCSS
        wcss_list.append(wcss)
        
        # the size of every group
        cluster_sizes = [kmeans.labels.tolist().count(i) for i in range(k)]
        
        # Put out the processes
        cluster_centers_names = [names[i] for i in kmeans.representative_samples.tolist()]
        print(f"Number of clusters: {k}")
        print(f"Cluster centers (names only):\n{cluster_centers_names}")
        print(f"Cluster sizes:\n{cluster_sizes}")
        print(f"Average WCSS (Average Within-Cluster Sum of Squares): {wcss}")
        print('\n')
        
        # save csv
        output_path = os.path.join(output_dir, f"clusters_info_k{k}.csv")
        with open(output_path, 'w') as f:
            f.write("Name,Cluster,RMSD\n")
            for i, label in enumerate(kmeans.labels.tolist()):
                rmsd = torch.sqrt(kmeans.dists[i][label] / matrix.shape[1]).item()  # 调整后的RMSD计算
                f.write(f"{names[i]},{label},{rmsd}\n")

        # Add the following code to save a CSV file containing cluster centers
        output_centers_path = os.path.join(output_dir, f"cluster_centers_k{k}.csv")
        with open(output_centers_path, 'w') as f:
            f.write("Cluster,Center\n")
            for i, center in enumerate(kmeans.centers.tolist()):
                f.write(f"{i},{','.join(map(str, center))}\n")

    # generate elbow chart
    plt.plot(k_range, wcss_list, marker='o')
    
    # Title
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average WCSS (Average Within-Cluster Sum of Squares)')
    plt.title('Elbow Method')
    
    # save as a chart
    plt.savefig(os.path.join(output_dir, "elbow_method.pdf"))
    plt.close()


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

if __name__ == "__main__":

    dir_in = path + "k-means/all_atom_coordinates.csv"

    data = pd.read_csv(dir_in)
    matrix = data.iloc[:, 1:].values  
    
    # device choosing
    device = choose_device(False)  
    
    # convert into PyTorch
    matrix = torch.from_numpy(matrix).to(device)
    matrix = matrix.float()
    
    # k-means
    sf_kmeans(matrix, device)

end_cluster = time.perf_counter()
print('Cluster consumes: {} s'.format(end_cluster - start_cluster))


print('Totally, the calculation consumes: {} s'.format(end_cluster - start_average))
