import numpy as np
import scipy.sparse as sp


class PageRankGlobal:
    def __init__(self, graph, beta=0.8, max_iter=100):
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.beta = beta
        # Create the adjacency and degree matrices
        self.adjacency_matrix, self.degree_matrix = self.create_matrices()
        # Create the transition matrix P as AD^{-1}
        self.transition_matrix = self.create_transition_matrix()
        # Max propagation iteration
        self.max_iter = max_iter

    def create_matrices(self):
        num_nodes = self.edges.max().max() + 1 # Assuming the largest node index is the node count
        # num_nodes = self.nodes + 1
        # Create an adjacency matrix A
        A = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), (self.edges.iloc[:, 0], self.edges.iloc[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float64
        )
        # Make sure the adjacency matrix is symmetric
        A = A + A.T - sp.diags(A.diagonal())
        # Create a degree matrix D
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), offsets=0)
        return A, D

    def create_transition_matrix(self):
        # Compute the inverse of the degree matrix, avoiding division by zero
        degrees = self.degree_matrix.diagonal()
        inv_degrees = np.reciprocal(degrees, where=degrees != 0)
        inv_degree_matrix = sp.diags(inv_degrees, offsets=0)
        # Compute the transition matrix as AD^{-1}
        P = self.adjacency_matrix.dot(inv_degree_matrix)
        return P.tocsr()
        # return P.todense()
    
    def create_lazy_walk_matrix(self):
        return 0.5 * (sp.eye(self.transition_matrix.shape[0]) + self.transition_matrix.copy())

    def compute_lazy_pagerank(self, s, beta, max_iter):
        # First, create the lazy walk matrix
        lazy_walk_matrix = self.create_lazy_walk_matrix()
        
        # Initialize the PageRank vector p to the personalization vector s
        p = s.copy()
        
        # Perform the power iteration using the lazy walk matrix
        for _ in range(max_iter):
            p = (1 - beta) * s + beta * lazy_walk_matrix.dot(p)
        
        return p
    
    def propagate_selected(self, sampled_indices):
        num_nodes = self.transition_matrix.shape[0]
        score_matrix = np.zeros((sampled_indices.shape[0], num_nodes))

        for idx, node_index in enumerate(sampled_indices):
            s = np.zeros((num_nodes, 1))
            s[node_index] = 1

            p = self.compute_lazy_pagerank(s, self.beta, self.max_iter)
            score_matrix[idx, :] = p.T

        return score_matrix
    
class PrivatePageRankClip:
    def __init__(self, graph, epsilon, delta, eta, sigma, beta, max_iter, sample):
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.beta = beta
        self.eta = eta # Threshold
        self.sample = sample
        self.epsilon = epsilon
        # self.epsilon = epsilon #RDP parameter
        self.delta = delta
        # self.alpha = alpha
        # Create the adjacency and degree matrices
        self.adjacency_matrix, self.degree_matrix = self.create_matrices()
        # Create the transition matrix P as AD^{-1}
        self.transition_matrix_unclip = self.create_transition_matrix()
        self.transition_matrix = self.clip_transition_matrix(self.transition_matrix_unclip.copy(), self.eta)
        # Max propagation iteration
        self.max_iter = max_iter
        self.sigma = sigma

    def create_matrices(self):
        num_nodes = self.edges.max().max() + 1 # Assuming the largest node index is the node count
        # num_nodes = self.nodes + 1
        # Create an adjacency matrix A
        A = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), (self.edges.iloc[:, 0], self.edges.iloc[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float64
        )
        # Make sure the adjacency matrix is symmetric
        A = A + A.T - sp.diags(A.diagonal())
        # Create a degree matrix D
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), offsets=0)
        return A, D

    def create_transition_matrix(self):
        # Compute the inverse of the degree matrix, avoiding division by zero
        degrees = self.degree_matrix.diagonal()
        inv_degrees = np.reciprocal(degrees, where=degrees != 0)
        inv_degree_matrix = sp.diags(inv_degrees, offsets=0)
        # Compute the transition matrix as AD^{-1}
        P = self.adjacency_matrix.dot(inv_degree_matrix)
        return P.tocsr()
        # return P.todense()

    def clip_transition_matrix(self, P, eta):
        P = sp.csr_matrix(P)  # Convert to CSR format
        min_eta = np.vectorize(lambda x: min(x, eta))
        P.data = min_eta(P.data)

        P = P.tolil()  # Convert to LIL format for efficient modification
        non_diagonal_sums = np.array(P.sum(axis=0) - P.diagonal()).flatten()
        new_diagonals = 1 - non_diagonal_sums
        P.setdiag(new_diagonals)
        return P.tocsr()  # Convert back to CSR format if needed
    
    def project_onto_simplex(self, v, z=1):
        v = np.asarray(v, dtype=np.float64)
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features, dtype=np.float64) + 1 
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def compute_private_pagerank(self, s, beta, transition_matrix, max_iter, sigma):
        p = s.copy()
        # Generate all Gaussian noise in advance
        noise = np.random.normal(scale=sigma, size=(transition_matrix.shape[0], max_iter, 1))
        for i in range(max_iter):
            p = (1 - beta) * s + beta * transition_matrix.dot(p) + noise[:, i]
            # print(f'p shape: {p.shape}')
            p = self.project_onto_simplex(p.flatten())  # Project onto simplex and reshape
            p = p.reshape(-1, 1)  # Reshape back to column vector
        return p

    def compute_private_pagerank_threshold(self, s, beta, eta, transition_matrix_unclip, max_iter, sigma, noise_type, personalize, middlestep, c):
        num_nodes = transition_matrix_unclip.shape[0]
        p = s.copy()
        source_node_index = np.argmax(s)  # Assuming s is a one-hot vector representing the source node

        # Create the lazy random walk matrix
        W = 0.5 * (sp.eye(num_nodes) + transition_matrix_unclip)

        # Define eta_D vector
        degree_vector = self.degree_matrix.diagonal()
        eta_D = eta * degree_vector
        if personalize == True:
            eta_D[source_node_index] = 1  # Ensure eta_D is 1 for the source node
        elif personalize == False:
            eta_D[source_node_index] *= c
        eta_D = eta_D.reshape(-1, 1)
        # print(f'l1 sum of eta_D: {np.linalg.norm(eta_D, ord=1)}')

        # Define tilde_eta vector for graph-independent clipping
        ones_vector = np.ones(self.degree_matrix.diagonal().shape)
        tilde_eta = eta * ones_vector
        if personalize == True:
            tilde_eta[source_node_index] = 1  # Ensure eta_D is 1 for the source node
        elif personalize == False:
            tilde_eta[source_node_index] *= c
        tilde_eta = tilde_eta.reshape(-1, 1)

        for i in range(max_iter):
            if personalize == False:
                p = np.minimum(p, eta_D)
            # Calculate the update rule without noise for the first 'm' iterations
            p = (1 - beta) * s + beta * W.dot(p)

            if i > middlestep:
                if noise_type == 'None':
                    # Apply clipping
                    p = np.minimum(p, eta_D)
                    # Ensure p is a column vector
                    p = p.reshape(-1, 1)
                elif noise_type == 'Laplacian':
                    noise1 = np.random.laplace(scale=sigma / np.sqrt(2), size=p.shape)
                    noise2 = np.random.laplace(scale=sigma / np.sqrt(2), size=p.shape)
                    p = p + noise1 + noise2
                    # Apply clipping and ensure p is a column vector if not the last iteration
                    if i < max_iter - 1:
                        p = self.project_onto_simplex(p.flatten())
                        p = p.reshape(-1, 1)  # Reshape back to column vector
                        p = np.minimum(np.maximum(p, 0), eta_D)
                    p = p.reshape(-1, 1)
                elif noise_type == 'Gaussian':
                    noise = np.random.normal(scale=sigma, size=p.shape)
                    p = p + noise
                    if i < max_iter - 1:
                        p = np.minimum(np.maximum(p, 0), eta_D)
                    p = p.reshape(-1, 1)
                elif noise_type == 'Laplacian_independent':
                    noise1 = np.random.laplace(scale=sigma / np.sqrt(2), size=p.shape)
                    noise2 = np.random.laplace(scale=sigma / np.sqrt(2), size=p.shape)
                    p = p + noise1 + noise2
                    if i < max_iter - 1:
                        p = np.minimum(np.maximum(p, 0), tilde_eta)
                    p = p.reshape(-1, 1)
                else:
                    raise ValueError("Unsupported noise type. Choose either 'Laplacian', 'Gaussian' or 'None'.")
            else:
                # Ensure p is a column vector in all cases
                p = p.reshape(-1, 1)
        return p
    
    def propagate_selected(self, sampled_indices, noise_type, personalize = True, middlestep = 0, c = 1):
        num_nodes = self.transition_matrix.shape[0]
        score_matrix = np.zeros((sampled_indices.shape[0], num_nodes))

        for idx, node_index in enumerate(sampled_indices):
            s = np.zeros((num_nodes, 1))
            s[node_index] = 1

            p = self.compute_private_pagerank_threshold(s, self.beta, self.eta, self.transition_matrix_unclip, self.max_iter, self.sigma, noise_type, personalize = personalize, middlestep = middlestep, c = c)
            score_matrix[idx, :] = p.T

        return score_matrix

class PushFlow:
    def __init__(self, graph, beta=0.8, zeta=1e-6, max_iter=100):
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.alpha = 1 - beta
        self.zeta = zeta
        self.max_iter = max_iter
        self.adjacency_matrix, self.degree_matrix = self.create_matrices()
        self.transition_matrix = self.create_transition_matrix()
        self.beta = beta

    def create_matrices(self):
        num_nodes = self.edges.max().max() + 1
        A = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), (self.edges.iloc[:, 0], self.edges.iloc[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float64
        )
        A = A + A.T - sp.diags(A.diagonal())
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), offsets=0)
        return A, D

    def create_transition_matrix(self):
        degrees = self.degree_matrix.diagonal()
        inv_degrees = np.reciprocal(degrees, where=degrees != 0)
        inv_degree_matrix = sp.diags(inv_degrees, offsets=0)
        P = self.adjacency_matrix.dot(inv_degree_matrix)
        return P.tocsr()

class PrivatePushFlowEfficient(PushFlow):
    def __init__(self, graph, epsilon, delta, eta, beta=0.8, zeta=1e-6, max_iter=100):
        super().__init__(graph, beta, zeta, max_iter)
        self.eta = eta
        self.epsilon = epsilon
        self.delta = delta
        self.lazy_walk_matrix = self.create_lazy_walk_matrix()

    def create_lazy_walk_matrix(self):
        # Using the transition matrix to create the lazy walk matrix
        lazy_walk_matrix = 0.5 * (sp.eye(self.transition_matrix.shape[0]) + self.transition_matrix.copy())

        return lazy_walk_matrix
    
    def compute_private_pushflow(self, s):
        num_nodes = self.transition_matrix.shape[0]
        p = np.zeros(num_nodes)  # PPR vector
        r = np.zeros(num_nodes)  # Residual vector initialized with zeros
        r[s] = 1  # Inject flow into the source node
        h = np.zeros(num_nodes)  # Pushed flow vector
        degree_vector = self.degree_matrix.diagonal()
        one_vector = np.ones(num_nodes) * max(degree_vector)

        # Different thresholds for the source node and the rest
        threshold_vector = one_vector * self.eta
        threshold_vector[s] = 1  # Special threshold for the source node

        for _ in range(self.max_iter):
            f = np.minimum(r, threshold_vector - h)  # Flow to push, with special treatment for source node

            # Update the total pushed flow and the PPR vector
            h += f
            p += self.alpha * f
            r -= f  # Update residual flow

            # Update the residual flow with distributed flow
            # Assuming lazy_walk_matrix is a transition probability matrix
            r += (1 - self.alpha) * self.lazy_walk_matrix.dot(f)

        return p


    def compute_private_pushflow_selected(self, sampled_indices, noise_type, personalize = True):
        num_nodes = self.transition_matrix.shape[0]
        score_matrix = np.zeros((len(sampled_indices), num_nodes))

        for idx, node_index in enumerate(sampled_indices):
            # Initialize the personalization vector s
            s = np.zeros(num_nodes)
            s[node_index] = 1  # Set the source node

            if personalize == True:
                # Compute the Private PushFlow
                p = self.compute_private_pushflow(node_index)
                score_matrix[idx, :] = p


        if noise_type == 'Laplacian':
            noise = np.random.laplace(0, (2 + self.beta) * self.eta / self.epsilon, score_matrix.shape)
            score_matrix_noisy = score_matrix + noise
        elif noise_type == 'Gaussian':
            # Compute sigma for Gaussian noise based on L2 sensitivity, epsilon, and delta
            sigma = (2 + self.beta) * self.eta * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            print('Calibrated Noise: ' + str(sigma))
            # Add Gaussian noise for differential privacy
            noise = np.random.normal(0, sigma, score_matrix.shape)
            score_matrix_noisy = score_matrix + noise
        elif noise_type == 'None':
            score_matrix_noisy = score_matrix
        else:
            raise ValueError("Unsupported noise type. Choose either 'Laplacian', 'Gaussian' or 'None'.")

        return score_matrix_noisy

class EdgeFlipping:
    def __init__(self, graph, epsilon, d_min = 1, beta=0.85, max_iter = 100):
            self.nodes = graph.nodes
            self.edges = graph.edges
            self.beta = beta
            self.epsilon = epsilon
            # Create the adjacency and degree matrices
            self.adjacency_matrix, self.degree_matrix = self.create_matrices()
            self.adjacency_matrix_DP, self.degree_matrix_DP = self.create_DP_matrices(self.epsilon)
            # Create the transition matrix P as AD^{-1}
            self.transition_matrix_DP = self.create_transition_matrix()
            # Max propagation iteration
            self.max_iter = max_iter
            self.d_min = d_min

    def create_matrices(self):
        num_nodes = self.edges.max().max() + 1 # Assuming the largest node index is the node count
        # num_nodes = self.nodes + 1
        # Create an adjacency matrix A
        A = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), (self.edges.iloc[:, 0], self.edges.iloc[:, 1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float64
        )
        # Make sure the adjacency matrix is symmetric
        A = A + A.T - sp.diags(A.diagonal())
        # Create a degree matrix D
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), offsets=0)
        return A, D
    
    def create_DP_matrices(self, epsilon):
        A, _ = self.create_matrices()
        A_dp = self.dp_edge_flipping(A.toarray(), epsilon)
        A_dp = sp.coo_matrix(A_dp)
        D_dp = sp.diags(np.array(A_dp.sum(axis=1)).flatten(), offsets=0)
        return A_dp, D_dp

    def dp_edge_flipping(self, adjacency, epsilon):
        n_nodes = len(adjacency)
        random_flip = np.random.uniform(0, 1, (n_nodes, n_nodes))
        random_flip = np.triu(random_flip, 1) + np.triu(random_flip, 1).T

        random_flip[random_flip > 0.5] = 1
        random_flip[random_flip <= 0.5] = 0

        p = 2 / (1 + np.exp(epsilon))
        random_positions = np.random.uniform(0, 1, (n_nodes, n_nodes))
        random_positions = np.triu(random_positions, 1) + np.triu(random_positions, 1).T

        random_positions = random_positions <= p
        dp_adjacency = adjacency.copy()
        dp_adjacency[random_positions] = random_flip[random_positions]
        return dp_adjacency

    def create_transition_matrix(self):
        # Compute the inverse of the degree matrix, avoiding division by zero
        degrees = self.degree_matrix_DP.diagonal()
        inv_degrees = np.reciprocal(degrees, where=degrees != 0)
        inv_degree_matrix = sp.diags(inv_degrees, offsets=0)
        # Compute the transition matrix as AD^{-1}
        P = self.adjacency_matrix_DP.dot(inv_degree_matrix)
        return P.tocsr()
        # return P.todense()

    def create_lazy_walk_matrix(self):
        return 0.5 * (sp.eye(self.transition_matrix_DP.shape[0]) + self.transition_matrix_DP.copy())
    
    def create_DP_matrices_personalized(self, node_index):
        # Convert dp_adjacency to LIL for efficient row/column operations
        dp_adjacency = sp.lil_matrix(self.adjacency_matrix_DP.copy())
        
        # Access original adjacency matrix rows and columns
        original_adjacency = self.adjacency_matrix.copy()
        dp_adjacency[node_index, :] = original_adjacency[node_index, :]
        dp_adjacency[:, node_index] = original_adjacency[:, node_index]
        
        # Convert back to COO format and recompute the degree matrix
        dp_adjacency = dp_adjacency.tocoo()
        D_dp_personalized = sp.diags(np.array(dp_adjacency.sum(axis=1)).flatten(), offsets=0)
        
        # Compute the transition matrix
        degrees = D_dp_personalized.diagonal()
        inv_degrees = np.reciprocal(degrees, where=degrees != 0)
        inv_degree_matrix = sp.diags(inv_degrees, offsets=0)
        P = dp_adjacency.dot(inv_degree_matrix)

        # Calculate 0.5 * (I + P)
        n = P.shape[0]
        I = sp.eye(n)
        lazy_random_walk = 0.5 * (I + P)
        return lazy_random_walk.tocsr()  # Return the CSR format of the transition matrix for better performance

        
    def compute_lazy_pagerank(self, s, beta, max_iter):
        # First, create the lazy walk matrix
        lazy_walk_matrix = self.create_lazy_walk_matrix()
        
        # Initialize the PageRank vector p to the personalization vector s
        p = s.copy()
        
        # Perform the power iteration using the lazy walk matrix
        for _ in range(max_iter):
            p = (1 - beta) * s + beta * lazy_walk_matrix.dot(p)
        
        return p
    
    def compute_lazy_pagerank_personalize(self, s, beta, max_iter, node_index):
        # First, create the lazy walk matrix
        lazy_walk_matrix_personalize = self.create_DP_matrices_personalized(node_index)
        
        # Initialize the PageRank vector p to the personalization vector s
        p = s.copy()
        
        # Perform the power iteration using the lazy walk matrix
        for _ in range(max_iter):
            p = (1 - beta) * s + beta * lazy_walk_matrix_personalize.dot(p)
        return p
    
    
    def propagate_selected_personalized(self, sampled_indices):
        num_nodes = self.transition_matrix_DP.shape[0]
        score_matrix = np.zeros((sampled_indices.shape[0], num_nodes))

        for idx, node_index in enumerate(sampled_indices):
            s = np.zeros((num_nodes, 1))
            s[node_index] = 1

            p = self.compute_lazy_pagerank_personalize(s, self.beta, self.max_iter, node_index)
            score_matrix[idx, :] = p.T

        return score_matrix







