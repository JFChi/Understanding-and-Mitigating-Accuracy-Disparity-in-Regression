
import logging
import sys

import numpy as np
import torch

def get_logger(filename):
	# Logging configuration: set the basic configuration of the logging system
	log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
									  datefmt='%m-%d %H:%M')
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	# File logger
	file_handler = logging.FileHandler("{}.log".format(filename))
	file_handler.setFormatter(log_formatter)
	file_handler.setLevel(logging.DEBUG)
	logger.addHandler(file_handler)
	# Stderr logger
	std_handler = logging.StreamHandler(sys.stdout)
	std_handler.setFormatter(log_formatter)
	std_handler.setLevel(logging.DEBUG)
	logger.addHandler(std_handler)
	return logger


# def conditional_errors(preds, labels, attrs):
#     """
#     Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
#     :param preds: The predicted label given by a model.
#     :param labels: The groundtruth label.
#     :param attrs: The label of sensitive attribute.
#     :return: Overall classification error, error | A = 0, error | A = 1.
#     """
#     assert preds.shape == labels.shape and labels.shape == attrs.shape
#     cls_error = 1 - np.mean(preds == labels)
#     idx = attrs == 0
#     error_0 = 1 - np.mean(preds[idx] == labels[idx])
#     error_1 = 1 - np.mean(preds[~idx] == labels[~idx])
#     return cls_error, error_0, error_1

def conditional_mse_errors(preds, labels, attrs):
	"""
	Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
	:param preds: The predicted label given by a model.
	:param labels: The groundtruth label.
	:param attrs: The label of sensitive attribute.
	:return: Overall classification error, error | A = 0, error | A = 1.
	"""
	assert preds.shape == labels.shape
	cls_error = np.mean((preds-labels)**2)
	idx = attrs == 0
	error_0 = np.mean((preds[idx]-labels[idx])**2)
	error_1 = np.mean((preds[~idx]-labels[~idx])**2)
	return cls_error, error_0, error_1

# MMD unbiasd distance
# code adapted from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
class MMDStatistic:
	r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.

	The kernel used is equal to:

	.. math ::
		k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},

	for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.."""

	def __init__(self, alphas, kernel_name="gaussian"):
		self.alphas = alphas
		self.kernel_name = kernel_name
		assert kernel_name in ["gaussian", "laplacian"]

	def __call__(self, sample_1, sample_2, ret_matrix=False):
		r"""
		Arguments
		---------
		sample_1: :class:`torch:torch.autograd.Variable`
			The first sample, of size ``(n_1, d)``.
		sample_2: variable of shape (n_2, d)
			The second sample, of size ``(n_2, d)``.
		alphas : list of :class:`float`
			The kernel parameters.
		ret_matrix: bool
			If set, the call with also return a second variable.

			This variable can be then used to compute a p-value using
			:py:meth:`~.MMDStatistic.pval`.

		Returns
		-------
		:class:`float`
			The test statistic.
		:class:`torch:torch.autograd.Variable`
			Returned only if ``ret_matrix`` was set to true."""

		self.n_1 = sample_1.shape[0]
		self.n_2 = sample_2.shape[0]

		# The three constants used in the test.
		self.a00 = 1. / (self.n_1 * (self.n_1 - 1))
		self.a11 = 1. / (self.n_2 * (self.n_2 - 1))
		self.a01 = - 1. / (self.n_1 * self.n_2)


		sample_12 = torch.cat((sample_1, sample_2), 0)
		if self.kernel_name == "gaussian":
			distances = pdist(sample_12, sample_12, norm=2)
		elif self.kernel_name == "laplacian":
			distances = pdist(sample_12, sample_12, norm=1)
		else:
			raise NotImplementedError

		kernels = None
		for alpha in self.alphas:
			# For single kernel
			if self.kernel_name == "gaussian":
				kernels_a = torch.exp(- alpha * distances ** 2)
			elif self.kernel_name == "laplacian":
				kernels_a = torch.exp(- alpha * distances)
			else:
				raise NotImplementedError
			# For multiple kernel, append kernel
			if kernels is None:
				kernels = kernels_a
			else:
				kernels = kernels + kernels_a

		k_1 = kernels[:self.n_1, :self.n_1]
		k_2 = kernels[self.n_1:, self.n_1:]
		k_12 = kernels[:self.n_1, self.n_1:]

		mmd = (2 * self.a01 * k_12.sum() +
			   self.a00 * (k_1.sum() - torch.trace(k_1)) +
			   self.a11 * (k_2.sum() - torch.trace(k_2)))
		if ret_matrix:
			return mmd, kernels
		else:
			return mmd

def pdist(sample_1, sample_2, norm=2, eps=1e-9):
	r"""Compute the matrix of all squared pairwise distances.

	Arguments
	---------
	sample_1 : torch.Tensor or Variable
		The first sample, should be of shape ``(n_1, d)``.
	sample_2 : torch.Tensor or Variable
		The second sample, should be of shape ``(n_2, d)``.
	norm : float
		The l_p norm to be used.

	Returns
	-------
	torch.Tensor or Variable
		Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
		``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
	n_1, n_2 = sample_1.size(0), sample_2.size(0)
	norm = float(norm)
	if norm == 2.:
		norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
		norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
		norms = (norms_1.expand(n_1, n_2) +
				 norms_2.transpose(0, 1).expand(n_1, n_2))
		distances_squared = norms - 2 * sample_1.mm(sample_2.t())

		### test shape ####
		# print("In pdist")
		# print(norms_1)
		# print(norms_2)
		# print(norms_1.expand(n_1, n_2))
		# print(norms_2.transpose(0, 1).expand(n_1, n_2))
		# print(norms_1.shape, norms_2.shape, norms.shape)
		# print(distances_squared)
		# print(distances_squared.shape)
		###################

		return torch.sqrt(eps + torch.abs(distances_squared))
	else:
		dim = sample_1.size(1)
		expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
		expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
		differences = torch.abs(expanded_1 - expanded_2) ** norm
		inner = torch.sum(differences, dim=2, keepdim=False)
		return (eps + inner) ** (1. / norm)

# MMD unbiasd distance
# code adapted from https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/statistics_diff.py
class MMDBiasedStatistic:
	r"""The *biased* MMD test of :cite:`gretton2012kernel`.
	"""

	def __init__(self, alphas, kernel_name="gaussian"):
		self.alphas = alphas
		self.kernel_name = kernel_name
		assert kernel_name in ["gaussian", "laplacian"]

	def __call__(self, sample_1, sample_2, ret_matrix=False):

		self.n_1 = sample_1.shape[0]
		self.n_2 = sample_2.shape[0]

		# The three constants used in the test.
		self.a00 = 1. / (self.n_1 * self.n_1)
		self.a11 = 1. / (self.n_2 * self.n_2)
		self.a01 = - 1. / (self.n_1 * self.n_2)


		sample_12 = torch.cat((sample_1, sample_2), 0)
		if self.kernel_name == "gaussian":
			distances = pdist(sample_12, sample_12, norm=2)
		elif self.kernel_name == "laplacian":
			distances = pdist(sample_12, sample_12, norm=1)
		else:
			raise NotImplementedError

		kernels = None
		for alpha in self.alphas:
			# For single kernel
			if self.kernel_name == "gaussian":
				kernels_a = torch.exp(- alpha * distances ** 2)
			elif self.kernel_name == "laplacian":
				kernels_a = torch.exp(- alpha * distances)
			else:
				raise NotImplementedError
			# For multiple kernel, append kernel
			if kernels is None:
				kernels = kernels_a
			else:
				kernels = kernels + kernels_a

		k_1 = kernels[:self.n_1, :self.n_1]
		k_2 = kernels[self.n_1:, self.n_1:]
		k_12 = kernels[:self.n_1, self.n_1:]

		mmd = (2 * self.a01 * k_12.sum() +
			   self.a00 * k_1.sum() +
			   self.a11 * k_2.sum())
		if ret_matrix:
			return mmd, kernels
		else:
			return mmd

if __name__ == "__main__":
	# test MMD
	torch.manual_seed(42)
	x = torch.FloatTensor([[3], [4], [5]])
	y = torch.FloatTensor([[1], [2]])
	# y = torch.FloatTensor([[1,2,3,4], [3,7,1,6], [3,5,1,6]]) * 0.1

	#### for guassian kernel ####
	print("test guassian kernels")
	alphas = [1.0] # coeiffient of rbf kernel 
	print(x)
	print(y)
	n1, n2 = x.shape[0], y.shape[0]
	print("n1, n2", n1, n2)
	mmd_dist = MMDBiasedStatistic(alphas, kernel_name="gaussian") # MMDStatistic(alphas, kernel_name="guassian")
	mmd, dist_matrix  = mmd_dist(x, y, ret_matrix=True)
	print("dist_matrix")
	print(dist_matrix)
	print("mmd", mmd)
	############################

	#### for laplacian kernel ####
	print("test laplacian kernels")
	alphas = [1.0] # coeiffient of laplacian kernel 
	print(x)
	print(y)
	n1, n2 = x.shape[0], y.shape[0]
	print("n1, n2", n1, n2)
	mmd_dist = MMDBiasedStatistic(alphas, kernel_name="laplacian") # MMDStatistic(alphas, kernel_name="laplacian")
	mmd, dist_matrix  = mmd_dist(x, y, ret_matrix=True)
	print("dist_matrix")
	print(dist_matrix)
	print("mmd", mmd)
	############################



