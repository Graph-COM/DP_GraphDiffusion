import os
import pandas as pd
import numpy as np

class BlogCatalogDataset:
    def __init__(self):
        self.current_path = os.getcwd()
        self.edges_file_path = os.path.join(self.current_path, 'datasets/blogcatalog/edges.csv')
        self.group_edges_file_path = os.path.join(self.current_path, 'datasets/blogcatalog/group-edges.csv')
        
        self.nodes = 10312
        self.classes = 39

        self.edges = pd.DataFrame(columns=['node1', 'node2'])
        self.group_edges = pd.DataFrame(columns=['node', 'label'])
        
        # Load the data from csv files
        self.load_data()

    def load_data(self):
        # Load edges from csv
        if os.path.exists(self.edges_file_path):
            # Change node index from 1 - 10312 to 0 - 10311
            self.edges = pd.read_csv(self.edges_file_path, header=None) - 1
            self.edges.columns = ['node1', 'node2']
        else:
            print(f"Edges file not found at path: {self.edges_file_path}") 
        
        # Load labels from csv
        if os.path.exists(self.group_edges_file_path):
            # Change node index from 1 - 10312 to 0 - 10311
            self.group_edges = pd.read_csv(self.group_edges_file_path, header=None) - 1
            self.group_edges.columns = ['node', 'label']
        else:
            print(f"Group edges file not found at path: {self.group_edges_file_path}")

    def compute_max_degree(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        max_degree = degree_count.max()
        return max_degree
    
    def compute_degree_vector(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        # Creating a vector of zeros for all nodes
        degree_vector = np.zeros(self.nodes)
        # Assign the degree for each node based on the count
        for node, degree in degree_count.items():
            degree_vector[node] = degree
        return degree_vector

    def compute_density(self):
        num_edges = len(self.edges)
        max_edges = self.nodes * (self.nodes - 1) / 2
        return num_edges / max_edges

class ThemarkerDataset:
    def __init__(self):
        self.current_path = os.getcwd()
        self.edges_file_path = os.path.join(self.current_path, 'datasets/themarker/soc-themarker.edges')
        
        self.nodes = None  # This will be determined after loading the edges
        self.edges = pd.DataFrame(columns=['node1', 'node2'])
        
        # Load the data from csv files
        self.load_data()

    def load_data(self):
        # Load edges from csv
        if os.path.exists(self.edges_file_path):
            # Assuming the csv file has no header and nodes indexed from 1
            self.edges = pd.read_csv(self.edges_file_path, header=None, sep=' ') - 1
            self.edges.columns = ['node1', 'node2']
            # Update nodes count based on the max node index
            self.nodes = self.edges.max().max() + 1
        else:
            print(f"Edges file not found at path: {self.edges_file_path}")

    def compute_max_degree(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        max_degree = degree_count.max()
        return max_degree
    
    def compute_degree_vector(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        # Creating a vector of zeros for all nodes
        degree_vector = np.zeros(self.nodes)
        # Assign the degree for each node based on the count
        for node, degree in degree_count.items():
            degree_vector[node] = degree
        return degree_vector

    def compute_density(self):
        num_edges = len(self.edges)
        max_edges = self.nodes * (self.nodes - 1) / 2
        return num_edges / max_edges
    
class FlickrDataset:
    def __init__(self):
        self.current_path = os.getcwd()
        self.edges_file_path = os.path.join(self.current_path, 'datasets/flickr/edges.csv')
        self.group_edges_file_path = os.path.join(self.current_path, 'datasets/flickr/group-edges.csv')
        
        self.nodes = 80513
        self.classes = 195

        self.edges = pd.DataFrame(columns=['node1', 'node2'])
        self.group_edges = pd.DataFrame(columns=['node', 'label'])
        
        # Load the data from csv files
        self.load_data()

    def load_data(self):
        # Load edges from csv
        if os.path.exists(self.edges_file_path):
            # Change node index from 1 - 80513 to 0 - 80512
            self.edges = pd.read_csv(self.edges_file_path, header=None) - 1
            self.edges.columns = ['node1', 'node2']
        else:
            print(f"Edges file not found at path: {self.edges_file_path}") 
        
        # Load labels from csv
        if os.path.exists(self.group_edges_file_path):
            # Change node index from 1 - 80513 to 0 - 80512
            self.group_edges = pd.read_csv(self.group_edges_file_path, header=None) - 1
            self.group_edges.columns = ['node', 'label']
        else:
            print(f"Group edges file not found at path: {self.group_edges_file_path}")

    def compute_max_degree(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        max_degree = degree_count.max()
        return max_degree
    
    def compute_degree_vector(self):
        all_nodes = pd.concat([self.edges['node1'], self.edges['node2']])
        degree_count = all_nodes.value_counts()
        # Creating a vector of zeros for all nodes
        degree_vector = np.zeros(self.nodes)
        # Assign the degree for each node based on the count
        for node, degree in degree_count.items():
            degree_vector[node] = degree
        return degree_vector

    def compute_density(self):
        num_edges = len(self.edges)
        max_edges = self.nodes * (self.nodes - 1) / 2
        return num_edges / max_edges
    
