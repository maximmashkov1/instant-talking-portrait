import numpy as np
from typing import Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def find_ijlr_vertices(edge: np.ndarray, faces: np.ndarray):

    lr_indices = [np.nan, np.nan]
    count = 0
    for i, face in enumerate(faces):
        if np.any(face == edge[0]):
            if np.any(face == edge[1]):
                neighbour_index = np.where(face[np.where(face != edge[0])] != edge[1])[0][0]
                n = face[np.where(face != edge[0])]
                lr_indices[count] = int(n[neighbour_index])
                count += 1

                if count == 2:
                    break
    l_index, r_index = lr_indices
    return [l_index, r_index]

class StepOne(object):
    @staticmethod
    def compute_g_matrix(
            vertices: np.ndarray,
            edges: np.ndarray,
            faces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        g_product = np.zeros((np.size(edges, 0), 2, 8), dtype=np.float32)
        gi = np.zeros((np.size(edges, 0), 4), dtype=np.float32)

        if edges.dtype not in [np.int32]:
            raise ValueError('Invalid dtype of edge indices. Requires np.uint32, np.uint64 or int.')

        # Compute G_k matrix for each `k`.
        for k, edge in enumerate(edges):
            i_vert, j_vert = vertices[edge]
            i_index, j_index = edge

            l_index, r_index = find_ijlr_vertices(edge, faces)
            l_vert = vertices[l_index]

            # For 3 neighbour points (when at the graph edge).
            if np.isnan(r_index):
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1]],
                             dtype=np.float32)
                _slice = 6
            # For 4 neighbour points (when not at the graph edge).
            else:
                r_vert = vertices[r_index]
                g = np.array([[i_vert[0], i_vert[1], 1, 0],
                              [i_vert[1], -i_vert[0], 0, 1],
                              [j_vert[0], j_vert[1], 1, 0],
                              [j_vert[1], -j_vert[0], 0, 1],
                              [l_vert[0], l_vert[1], 1, 0],
                              [l_vert[1], -l_vert[0], 0, 1],
                              [r_vert[0], r_vert[1], 1, 0],
                              [r_vert[1], -r_vert[0], 0, 1]],
                             dtype=np.float32)
                _slice = 8

            # G[k,:,:]
            gi[k, :] = [i_index, j_index, l_index, np.nan if np.isnan(r_index) else r_index]
            x_matrix_pad = np.linalg.lstsq(g.T @ g, g.T, rcond=None)[0]
            g_product[k, :, :_slice] = x_matrix_pad[0:2]

        return gi, g_product

    @staticmethod
    def compute_h_matrix(
            edges: np.ndarray,
            g_product: np.ndarray,
            gi: np.ndarray,
            vertices: np.ndarray
    ) -> np.ndarray:
        """
        Transformed term (v′_j − v′_i) − T_{ij} (v_j − v_i) from paper requires
        computation of matrix H. To be able compute matrix H, we need matrix G
        from other method.

        :param edges: np.ndarray; requires dtype int/np.uint32/np.uint64
        :param g_product: np.ndarray;
        :param gi: np.ndarray;
        :param vertices: np.ndarray;
        :return: np.ndarray;
        """
        h_matrix = np.zeros((np.size(edges, 0) * 2, 8), dtype=np.float32)
        for k, edge in enumerate(edges):
            # ...where e is an edge vector..
            ek = np.subtract(*vertices[edge[::-1]])
            ek_matrix = np.array([[ek[0], ek[1]], [ek[1], -ek[0]]], dtype=np.float32)

            # Ful llength of ones/zero matrix (will be sliced in case on the contour of graph).
            _oz = np.array([[-1, 0, 1, 0, 0, 0, 0, 0],
                            [0, -1, 0, 1, 0, 0, 0, 0]],
                           dtype=np.float32)
            if np.isnan(gi[k, 3]):
                _slice = 6
            else:
                _slice = 8

            g = g_product[k, :, :_slice]
            oz = _oz[:, :_slice]
            h_calc = oz - (ek_matrix @ g)
            h_matrix[k * 2, :_slice] = h_calc[0]
            h_matrix[k * 2 + 1, :_slice] = h_calc[1]

        return h_matrix

    @staticmethod
    def compute_v_prime(
        edges: np.ndarray,
        vertices: np.ndarray,
        gi: np.ndarray,
        h_matrix: np.ndarray,
        c_indices: np.ndarray,
        c_vertices: np.ndarray,
        weight: np.float32 = np.float32(1000.)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        
        num_edges = np.size(edges, axis=0)
        num_vertices = np.size(vertices, axis=0)
        num_c_indices = np.size(c_indices)

        rows = num_edges * 2 + num_c_indices * 2
        cols = num_vertices * 2

        # Using csr_matrix for efficient arithmetic
        a1_matrix = sp.lil_matrix((rows, cols), dtype=np.float32)
        b1_vector = np.zeros((rows, 1), dtype=np.float32)
        v_prime = np.zeros((num_vertices, 2), dtype=np.float32)

        # Fill values in prepared matrices/vectors
        for k, g_indices in enumerate(gi):
            for i, point_index in enumerate(g_indices):
                if not np.isnan(point_index):
                    point_index = int(point_index)
                    a1_matrix[k * 2, point_index * 2] = h_matrix[k * 2, i * 2]
                    a1_matrix[k * 2 + 1, point_index * 2] = h_matrix[k * 2 + 1, i * 2]
                    a1_matrix[k * 2, point_index * 2 + 1] = h_matrix[k * 2, i * 2 + 1]
                    a1_matrix[k * 2 + 1, point_index * 2 + 1] = h_matrix[k * 2 + 1, i * 2 + 1]

        for c_enum_index, c_vertex_index in enumerate(c_indices):
            row_index = num_edges * 2 + c_enum_index * 2
            a1_matrix[row_index, c_vertex_index * 2] = weight
            a1_matrix[row_index + 1, c_vertex_index * 2 + 1] = weight
            b1_vector[row_index] = weight * c_vertices[c_enum_index, 0]
            b1_vector[row_index + 1] = weight * c_vertices[c_enum_index, 1]

        # Convert a1_matrix to CSR format for efficient arithmetic
        a1_matrix = a1_matrix.tocsr()
        ata = a1_matrix.T @ a1_matrix
        atb = a1_matrix.T @ b1_vector

        # Adding a small regularization term to improve numerical stability
        regularization = sp.eye(ata.shape[0]) * 1e-10
        ata += regularization

        # Solve the normal equation using spsolve
        v = spla.spsolve(ata, atb)
        v_prime[:, 0] = v[0::2]
        v_prime[:, 1] = v[1::2]

        return v_prime, a1_matrix, b1_vector





class StepTwo(object):
    @staticmethod
    def compute_t_matrix(
            edges: np.ndarray,
            g_product: np.ndarray,
            gi: np.ndarray,
            v_prime: np.ndarray
    ) -> np.ndarray:
        """
        From paper:

        The second step takes the rotation information from the result of the first step
        (i.e., computing the explicit values of T′k and normalizing them to remove the
        scaling factor), rotates the original edge vectors ek by the amount T′k, and
        then solves Equation (1) using the original rotated edge vectors. That is, we
        compute the rotation of each edge by using the result of the first step,
        and then normalize it.
        
        :param edges: np.ndarray; 
        :param g_product: np.ndarray; 
        :param gi: np.ndarray;
        :param v_prime: np.ndarray; transformed point in sense of rotation from step one
        :return: np.ndarray;
        """
        t_matrix = np.zeros(((np.size(edges, axis=0)), 2, 2,), dtype=np.float32)

        # We compute T′k for each edge.
        for k, edge in enumerate(edges):
            if np.isnan(gi[k, 3]):
                _slice = 6
                v = np.array([
                    [v_prime[int(gi[k, 0]), 0]],
                    [v_prime[int(gi[k, 0]), 1]],
                    [v_prime[int(gi[k, 1]), 0]],
                    [v_prime[int(gi[k, 1]), 1]],
                    [v_prime[int(gi[k, 2]), 0]],
                    [v_prime[int(gi[k, 2]), 1]]
                ], dtype=np.float32)
            else:
                _slice = 8
                v = np.array([
                    [v_prime[int(gi[k, 0]), 0]],
                    [v_prime[int(gi[k, 0]), 1]],
                    [v_prime[int(gi[k, 1]), 0]],
                    [v_prime[int(gi[k, 1]), 1]],
                    [v_prime[int(gi[k, 2]), 0]],
                    [v_prime[int(gi[k, 2]), 1]],
                    [v_prime[int(gi[k, 3]), 0]],
                    [v_prime[int(gi[k, 3]), 1]]],
                    dtype=np.float32
                )
            # We compute the rotation of each edge by using the result of the first step,
            g = g_product[k, :, :_slice]
            t = g @ v
            rot = np.array([[t[0], t[1]], [-t[1], t[0]]], dtype=np.float32)
            # and then normalize it.
            t_normalized = (np.float32(1) / np.sqrt(np.power(t[0], 2) + np.power(t[1], 2))) * rot
            # Store result.
            t_matrix[k, :, :] = t_normalized[:, :, 0]
        return t_matrix

    @staticmethod
    
    def compute_v_2prime(
        edges: np.ndarray,
        vertices: np.ndarray,
        t_matrix: np.ndarray,
        c_indices: np.ndarray,
        c_vertices: np.ndarray,
        weight: np.float32 = np.float32(1000)
    ) -> np.ndarray:
        """
        Compute the updated vertex positions.

        :param edges: np.ndarray; array of edge indices
        :param vertices: np.ndarray; array of vertex positions
        :param t_matrix: np.ndarray; transformation matrix
        :param c_indices: np.ndarray; constraint indices
        :param c_vertices: np.ndarray; constraint vertex positions
        :param weight: np.float; weight for the constraints
        :return: np.ndarray; updated vertex positions
        """
        num_edges = np.size(edges, axis=0)
        num_vertices = np.size(vertices, axis=0)
        num_constraints = np.size(c_indices)
        
        # Prepare blueprints.
        data = []
        row_indices = []
        col_indices = []

        b2_vector = np.zeros((num_edges + num_constraints, 2), dtype=np.float32)

        # Update values from precomputed components.
        for k, edge in enumerate(edges):
            data.extend([-1., 1.])
            row_indices.extend([k, k])
            col_indices.extend([int(edge[0]), int(edge[1])])

            e = np.subtract(*vertices[edge[::-1]])
            t_e = t_matrix[k, :, :] @ e
            b2_vector[k, :] = t_e[0], t_e[1]

        for c_index, c in enumerate(c_indices):
            data.append(weight)
            row_indices.append(num_edges + c_index)
            col_indices.append(c)
            b2_vector[num_edges + c_index, :] = weight * c_vertices[c_index, :]

        a2_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_edges + num_constraints, num_vertices))

        # Solve the linear system using sparse solver
        v_2prime_x = spla.spsolve(a2_matrix.T @ a2_matrix, a2_matrix.T @ b2_vector[:, 0])
        v_2prime_y = spla.spsolve(a2_matrix.T @ a2_matrix, a2_matrix.T @ b2_vector[:, 1])

        return np.vstack((v_2prime_x, v_2prime_y)).T
    
def graph_warp(
        vertices: np.ndarray,
        faces: np.ndarray= None,
        control_indices: np.ndarray= None,
        shifted_locations: np.ndarray= None,
        edges: np.ndarray= None,
        precomputed: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
) -> np.ndarray:
    """
    cnp.ndarray[float, ndim=2] vertices,
    cnp.ndarray[int, ndim=2] faces,
    cnp.ndarray[int, ndim=2] edges,
    cnp.ndarray[float, ndim=2] gi,
    cnp.ndarray[float, ndim=3] g_product,
    cnp.ndarray[float, ndim=2] h,
    cnp.ndarray[index, ndim=1] control_pts,
    cnp.ndarray[float, ndim=2] shift
    """

    gi, g_product, h = precomputed
    args = edges, vertices, gi, h, control_indices, shifted_locations

    new_vertices, _, _ = StepOne.compute_v_prime(*args)

    # Compute v'' from paper.
    t_matrix = StepTwo.compute_t_matrix(edges, g_product, gi, new_vertices)

    new_vertices = StepTwo.compute_v_2prime(edges, vertices, t_matrix, control_indices, shifted_locations)


    return new_vertices