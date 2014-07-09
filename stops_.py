__author__ = 'Kamil Koziara & Taiyeb Zahir'

import numpy


def mmap(fun, mat):
    f = numpy.vectorize(fun)
    return f(mat)


class Stops2:
    def __init__(self, gene_mat, pop, adj_mat, bound=None, secretion=None, reception=None, receptors=None,
                 init_env = None, secr_amount=1.0, leak=1.0, max_con = 1000.0, max_dist=None,
                 opencl = False):
        """
        Init of Stops
        Parameters:
         - gene_mat - matrix of gene interactions [GENE_NUM, GENE_NUM]
         - pop - array with initial population [POP_SIZE, GENE_NUM]
         - adj_mat - matrix with distances between each cell in population[POP_SIZE, POP_SIZE]
         - bound - vector of max value of each gene [GENE_NUM]
         - secretion - vector of length LIG_NUM where secretion[i] contains index
            of a gene which must be on to secrete ligand i
         - reception - vector of length LIG_NUM where reception[i] contains index
            of a gene which will be set to on when ligand i is accepted
         - receptors - vector of length LIG_NUM where receptors[i] contains index
            of a gene which has to be on to accept ligand i; special value -1 means that there is no
            need for specific gene expression for the ligand
         - secr_amount - amount of ligand secreted to the environment each time
         - leak - amount of ligand leaking from the environment each time
         - max_con - maximal ligand concentration
         - max_dist - maximal distance between a cell and an environment needed for
            the cell to accept ligands from the environment
         - opencl - if set to True opencl is used to boost the speed
        """
        self.gene_mat = numpy.array(gene_mat).astype(numpy.float32)
        self.pop = numpy.array(pop).astype(numpy.float32)
        self.adj_mat = numpy.array(adj_mat).astype(numpy.float32)
        self.secr_amount = secr_amount
        self.leak = leak
        self.max_con = max_con
        self.row_size = self.gene_mat.shape[0]
        self.pop_size = self.pop.shape[0]

        self.max_dist = numpy.max(adj_mat) if max_dist is None else max_dist

        if bound != None:
            self.bound = numpy.array(bound).astype(numpy.float32)
        else:
            # bound default - all ones
            self.bound = numpy.ones(self.row_size).astype(numpy.float32)

        if secretion != None:
            self.secretion = numpy.array(secretion).astype(numpy.int32)
        else:
            self.secretion = numpy.array([]).astype(numpy.int32)

        if reception != None:
            self.reception = numpy.array(reception).astype(numpy.int32)
        else:
            self.reception = numpy.array([]).astype(numpy.int32)

        self.max_lig = len(secretion)

        if init_env is None:
            self.init_env = numpy.zeros(self.max_lig)
        else:
            self.init_env = init_env

        self.env = numpy.array([self.init_env] * self.pop.shape[0]).astype(numpy.float32)

        if receptors != None:
            self.receptors = numpy.array(receptors).astype(numpy.int32)
        else:
            # receptors - default value "-1" - no receptor for ligand is necessary
            self.receptors = numpy.array([-1] * self.max_lig).astype(numpy.int32)

        self._random = numpy.random.random

        self.opencl = opencl
        self.pop_hit = numpy.zeros((self.pop_size, self.max_lig)).astype(numpy.int32)

        if opencl:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            self.mf = cl.mem_flags
            #init kernel
            self.program = self.__prepare_kernel()
            self.rand_state_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.pop.shape[0] * 112)
            self.program.init_ranlux(self.queue, (self.pop.shape[0], 1), None,
                                     numpy.uint32(numpy.random.randint(4e10)), self.rand_state_buf)
            # prepare multiplication matrix
            adj_mat_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.adj_mat)
            self.mul_mat_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.adj_mat.nbytes)
            self.program.init_mul_mat(self.queue, (self.pop.shape[0], 1), None, self.mul_mat_buf, adj_mat_buf,
                                   numpy.float32(self.max_dist))
        else:
            self.mul_mat = mmap(lambda x: 1. / (x**2) if x != 0 and x <= max_dist else 0., adj_mat)
            n_density = numpy.sum(abs(self.mul_mat), axis=1)
            sel_mat=numpy.cumsum(abs(self.mul_mat),axis=1,dtype=numpy.float32) # cumulative influence by cell
            self.mul_mat = (sel_mat.T / n_density).T # what if density is 0
            self.mul_mat = self.mul_mat.astype(numpy.float32)


    def step(self):
        if self.opencl:
            self._step_opencl()
        else:
            self._step_numpy()

    def __prepare_kernel(self):
        with open("init_kernel.c") as f:
            init_kernel = f.read() % {"pop_size": self.pop.shape[0]}
        with open("mat_mul_kernel.c") as f:
            mat_mul_kernel = f.read()
        with open("ranlux_random.c") as f:
            rand_fun = f.read()
        with open("expression_kernel.c") as f:
            expr_kernel = f.read() % {"row_size": self.pop.shape[1]}
        with open("secretion_kernel.c") as f:
            secr_kernel = f.read() % {"row_size": self.pop.shape[1], "pop_size": self.pop.shape[0],
                                      "max_lig": self.max_lig}
        with open("reception_kernel.c") as f:
            rec_kernel = f.read() % {"row_size": self.pop.shape[1], "pop_size": self.pop.shape[0],
                                      "max_lig": self.max_lig}
        #dbg = "# pragma OPENCL EXTENSION cl_intel_printf :enable\n"
        return cl.Program(self.ctx,
            init_kernel + "\n" +
            mat_mul_kernel + "\n" +
            rand_fun + "\n" +
            expr_kernel + "\n" +
            secr_kernel + "\n" +
            rec_kernel).build()

    def _step_opencl(self):
        # expression
        pop_size = self.pop.shape[0]
        gene_mat_size = self.gene_mat.shape[0]

        pop_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.pop)
        gene_mat_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.gene_mat)
        tokens_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = 4 * pop_size * gene_mat_size)
        # generate matrix of tokens simulating probability of particular actions taken by a cell
        # generate one random number for each cell
        self.program.mat_mul(self.queue, (pop_size, gene_mat_size), None, tokens_buf,
                     pop_buf, gene_mat_buf, numpy.int32(gene_mat_size))
        rand_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = 4 * pop_size)
        self.program.get_random_vector(self.queue, (int(pop_size / 4 + 1), 1), None,
                                  rand_buf, numpy.int32(pop_size), self.rand_state_buf)
        bound_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.bound)
        # generating new population state
        self.program.choice(self.queue, (pop_size, 1), None, pop_buf, tokens_buf, rand_buf, bound_buf)

        # self._secretion()
        # secretion
        env_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.env)
        secr_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.secretion)

        self.program.secretion(self.queue, (pop_size, 1), None, pop_buf, env_buf, secr_buf,
                               numpy.float32(self.max_con), numpy.float32(self.leak),
                               numpy.float32(self.secr_amount))

        # reception
        pop_hit_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = self.pop_size * self.max_lig * 4)
        self.program.fill_buffer(self.queue, (self.pop_size * self.max_lig, 1), None, pop_hit_buf, numpy.int32(0))

        rec_gene_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.reception)
        receptors_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.receptors)
        self.program.reception(self.queue, (pop_size, 1), None, pop_hit_buf, env_buf, self.mul_mat_buf, pop_buf,
                               receptors_buf, self.rand_state_buf)

        self.program.update_pop_with_reception(self.queue, (pop_size, 1), None,
                                               pop_buf, pop_hit_buf, rec_gene_buf, bound_buf)

        # storing state
        cl.enqueue_copy(self.queue, self.env, env_buf)
        cl.enqueue_copy(self.queue, self.pop, pop_buf)

    def _expression(self):
        # generate matrix of tokens simulating probability of particular actions taken by a cell
        # generate one random number for each cell
        tokens_mat = self.pop.dot(self.gene_mat)
        rnd_mat = self._random(self.pop.shape[0]) # random number for each cell

        # cumulative influence by cell
        sel_mat = numpy.cumsum(abs(tokens_mat), axis=1)
        # total influence by cell
        sums = numpy.sum(abs(tokens_mat), axis=1).reshape(self.pop.shape[0], 1)
        # normalized influence by cell
        norm_mat = numpy.array(sel_mat, dtype=numpy.float32) / sums
        # as a vertical vector
        rnd_mat.resize((self.pop.shape[0], 1))
        # boolean matrix with values greater than random
        bool_mat = (norm_mat - rnd_mat) > 0
        ind_mat = numpy.resize(numpy.array(range(self.pop.shape[1]) * self.pop.shape[0]) + 1, self.pop.shape)
        # matrix of indices
        sel_arr = numpy.select(list(bool_mat.transpose()), list(ind_mat.transpose())) - 1
        # the index of the first value greater than random (-1 if no such value)
        dir_arr = numpy.select(list(bool_mat.transpose()), list(numpy.array(tokens_mat).transpose()))
        for i, (s, d) in enumerate(zip(sel_arr, dir_arr)):
            if s >= 0:
                self.pop[i, s] = max(0, min(self.bound[s], self.pop[i, s] + (d / abs(d))))

    def _secretion(self):
        # secretion
        for i in range(self.pop.shape[0]):
            # for each cell
            for j, k in enumerate(self.secretion):
                # for each ligand
                if self.pop[i, k] > 0:
                    # if ligand is expressed
                    self.env[i, j] = min(self.max_con, self.env[i, j] + self.secr_amount)
                    self.pop[i, k] -= 1 # or get down to 0?

        # leaking
        leak_fun = numpy.vectorize(lambda x : max(0.0, x - self.leak))
        self.env = leak_fun(self.env)

    def _reception(self):
        # reception
        for i in range(self.pop.shape[0]):
            # for all cells
            for j,k in enumerate(self.env[i]):
                # for all type of ligands
                for num_lig in range(int(k)):
                    # for all ligands
                    rnd_matrix=[numpy.random.random()]*len(self.env)
                    bool_matrix=(self.mul_mat[i]-rnd_matrix)>0
                    bool_matrix=(bool_matrix).astype(numpy.int32)
                    sel_matrix=numpy.resize(numpy.insert(bool_matrix,0,0),self.pop_size)
                    rec_matrix=bool_matrix-sel_matrix
                    if rec_matrix.any():
                        index=numpy.where( rec_matrix > 0 )
                        if self.can_receive(j, self.pop[index[0][0]]):
                            self.pop[index[0][0],self.reception[j]]=min(self.pop[index[0][0],self.reception[j]]+1,self.bound[self.reception[j]])
                            k-=1                          
                                    
                    else:
                        if self.can_receive(j, self.pop[self.pop_size-1]):
                            self.pop[self.pop_size-1,self.reception[j]]=min(self.pop[self.pop_size-1,self.reception[j]]+1,self.bound[self.reception[j]])
                            k-=1

    def _step_numpy(self):
        self._expression()
        self._secretion()
        self._reception()

    def sim(self, steps=100):
        for i in range(steps):
            self.step()

    def can_receive(self, ligand, row):
        """Function describes if a specific cell (defined by its state) can receive specified ligand"""
        rec = self.receptors[ligand]
        return rec == -1 or row[rec] > 0
