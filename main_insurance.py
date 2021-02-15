import argparse
import time
import csv
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import InsuranceDataset
from models import MLPNet, WassersteinNet, CENet
from utils import conditional_mse_errors
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="insurance")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the adversarial loss",
                    type=float, default=0.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=750)
parser.add_argument("-r", "--lr", type=float, help="Learning rate of optimization", default=0.1)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=64)
parser.add_argument("-m", "--model", type=str,
                    help="Which model to run: [mlp|wmlp|CENet]",
                    default="mlp") 
parser.add_argument("-c", '--clip', type=float, default=0.2, help="parameters in WassersteinNet")

# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

logger = get_logger(args.name)
# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dtype = np.float32
logger.info("--------------------------------------------------")
logger.info("Running Insurance data set regression analysis")
logger.info("seed: {}".format(args.seed))
# Load insurance dataset.
time_start = time.time()
ins_train = InsuranceDataset(root_dir='data', phase='train')
ins_test = InsuranceDataset(root_dir='data', phase='test')
train_loader = DataLoader(ins_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(ins_test, batch_size=args.batch_size, shuffle=False)
time_end = time.time()
logger.info("Time used to load all the data sets: {} seconds.".format(time_end - time_start))
input_dim = ins_train.xdim
num_groups = 2
use_sigmoid = True

configs = {"num_groups": num_groups, 
            "num_epochs": args.epoch,
           "batch_size": args.batch_size, 
           "lr": args.lr, 
           "mu": args.mu, 
           "use_sigmoid": use_sigmoid, 
           "input_dim": input_dim,
           "weight_clipping": args.clip, # parameters in Wass' Net
           "hidden_layers": [7],
           "adversary_layers": [7]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]

lr = configs["lr"]

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 250))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

if args.model == "mlp":
	# Train MLPNet to get baseline results.
	logger.info("Experiment without debiasing:")
	logger.info("Hyperparameter setting = {}.".format(configs))
	# Train MLPNet without debiasing.
	time_start = time.time()
	net = MLPNet(configs).to(device)
	logger.info("Model architecture: {}".format(net))
	optimizer = optim.SGD(net.parameters(), lr=lr)
	net.train()
	for t in range(num_epochs):
		running_loss, total = 0.0, 0
		for xs, ys, attrs in train_loader:
			xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
			optimizer.zero_grad()
			ypreds = net(xs)
			loss = F.mse_loss(ypreds, ys)
			running_loss += loss.item() * len(ys)
			total += len(ys)
			loss.backward()
			optimizer.step()
		running_loss = running_loss / total
		logger.info("Iteration {}, loss value = {}".format(t, running_loss))
	time_end = time.time()
	logger.info("Time used for training = {} seconds.".format(time_end - time_start))
	# inference
	net.eval()
	running_loss, total = 0.0, 0
	ypreds_numpy, ys_numpy, attrs_numpy = [], [], []
	for xs, ys, attrs in test_loader:
		xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
		ypreds = net(xs)
		loss = F.mse_loss(ypreds, ys)
		# logging and saving 
		running_loss += loss.item() * len(ys)
		total += len(ys)
		ypreds_numpy.append(ypreds.detach().cpu().numpy())
		ys_numpy.append(ys.cpu().numpy())
		attrs_numpy.append(attrs.cpu().numpy())
	# summation and logging
	running_loss = running_loss / total
	ypreds_numpy = np.concatenate(ypreds_numpy, axis=0).squeeze()
	ys_numpy = np.concatenate(ys_numpy, axis=0).squeeze()
	attrs_numpy = np.concatenate(attrs_numpy, axis=0)
	cls_error, error_0, error_1 = conditional_mse_errors(ypreds_numpy, ys_numpy, attrs_numpy)
	logger.info("Inference, loss value = {}".format(running_loss))
	logger.info("Overall predicted error = {}, Err|A=0 = {}, Err|A=1 = {}".format(cls_error, error_0, error_1))
	logger.info("Error gap = {}".format(np.abs(error_0-error_1)))
	ys_var = np.var(ys_numpy) 
	r_squared = 1 - cls_error/ys_var
	logger.info("R squared = {}".format(r_squared))
	nmse = cls_error/ys_var
	# save data to csv
	csv_data = {"cls_error": cls_error,
				"error_0": error_0,
				"error_1": error_1,
				"err_gap": np.abs(error_0-error_1),
				"R^2": r_squared,
				"nmse": nmse
				}
	csv_fn = args.name + ".csv"
	with open(csv_fn, "a") as csv_file:
		fieldnames = ["cls_error", "error_0", "error_1", "err_gap", "R^2", "nmse"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		if os.path.exists(csv_fn):
			pass # no need to write headers
		else:
			writer.writeheader()
		writer.writerow(csv_data)
	# save prediction to npy file
	npy_fn = args.name + "_seed_%d"% args.seed + ".npz"
	np.savez(npy_fn, 
			 y_pred=ypreds_numpy, 
			 y_true=ys_numpy, 
			 A_true=attrs_numpy)	
elif args.model == "wmlp":
	# train wass MLP
	logger.info("Experiment with Wasserstein mlp: {} ".format(args.model))
	logger.info("Hyperparameter setting = {}.".format(configs))
	time_start = time.time()
	net = WassersteinNet(configs).to(device)
	logger.info("Model architecture: {}".format(net))
	optimizer = optim.SGD(net.parameters(), lr=lr)
	mu = args.mu
	net.train()
	for t in range(num_epochs):
		running_loss, running_adv_loss, total = 0.0, 0.0, 0
		for xs, ys, attrs in train_loader:
			## add clip norm in advesarial network to achieve Lipschitzness ##
			for p in net.adversaries.parameters():
				p.data.clamp_(-args.clip, args.clip)
			for p in net.sensitive_output_layer.parameters():
				p.data.clamp_(-args.clip, args.clip)
			# forward and calculate loss
			xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
			optimizer.zero_grad()
			ypreds, advesary_out = net(xs, ys)
			idx = attrs == 0 # index of sensitive '0'
			fw_0 = torch.mean(advesary_out[idx], dim=0).squeeze()
			fw_1 = torch.mean(advesary_out[~idx], dim=0).squeeze()
			loss = F.mse_loss(ypreds, ys)
			adv_loss = torch.abs(fw_0 - fw_1)
			running_loss += loss.item() * len(ys)
			adv_loss.item()
			if not math.isnan(adv_loss.item()):
				running_adv_loss += adv_loss.item() * len(ys)
			if not math.isnan(adv_loss.item()):
				loss -= mu * adv_loss
			total += len(ys)
			loss.backward()
			optimizer.step()
		running_loss = running_loss / total
		running_adv_loss = running_adv_loss / total
		logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
		# sys.exit(0) #TODO: delete
	time_end = time.time()
	logger.info("Time used for training = {} seconds.".format(time_end - time_start))
	# inference
	net.eval()
	running_loss, total = 0.0, 0
	ypreds_numpy, ys_numpy, attrs_numpy = [], [], []
	for xs, ys, attrs in test_loader:
		xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
		ypreds = net.inference(xs)
		loss = F.mse_loss(ypreds, ys)
		# logging and saving 
		running_loss += loss.item() * len(ys)
		total += len(ys)
		ypreds_numpy.append(ypreds.detach().cpu().numpy())
		ys_numpy.append(ys.cpu().numpy())
		attrs_numpy.append(attrs.cpu().numpy())
	# summation and logging
	running_loss = running_loss / total
	ypreds_numpy = np.concatenate(ypreds_numpy, axis=0).squeeze()
	ys_numpy = np.concatenate(ys_numpy, axis=0).squeeze()
	attrs_numpy = np.concatenate(attrs_numpy, axis=0)
	cls_error, error_0, error_1 = conditional_mse_errors(ypreds_numpy, ys_numpy, attrs_numpy)
	logger.info("Inference, loss value = {}".format(running_loss))
	logger.info("Overall predicted error = {}, Err|A=0 = {}, Err|A=1 = {}".format(cls_error, error_0, error_1))
	logger.info("Error gap = {}".format(np.abs(error_0-error_1)))
	ys_var = np.var(ys_numpy) 
	r_squared = 1 - cls_error/ys_var
	logger.info("R squared = {}".format(r_squared))
	nmse = cls_error/ys_var
	# save data to csv
	csv_data = {"cls_error": cls_error,
				"error_0": error_0,
				"error_1": error_1,
				"err_gap": np.abs(error_0-error_1),
				"R^2": r_squared,
				"nmse": nmse
				}
	csv_fn = args.name + ".csv"
	with open(csv_fn, "a") as csv_file:
		fieldnames = ["cls_error", "error_0", "error_1", "err_gap", "R^2", "nmse"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		if os.path.exists(csv_fn):
			pass # no need to write headers
		else:
			writer.writeheader()
		writer.writerow(csv_data)
	# save prediction to npy file
	npy_fn = args.name + "_seed_%d"% args.seed + ".npz"
	np.savez(npy_fn, 
			 y_pred=ypreds_numpy, 
			 y_true=ys_numpy, 
			 A_true=attrs_numpy)
elif args.model == "CENet":
	logger.info("Experiment with CENet: {} ".format(args.model))
	logger.info("Hyperparameter setting = {}.".format(configs))
	time_start = time.time()
	net = CENet(configs).to(device)
	logger.info("Model architecture: {}".format(net))
	optimizer = optim.SGD(net.parameters(), lr=lr)
	mu = args.mu
	net.train()
	for t in range(num_epochs):
		running_loss, running_adv_loss, total = 0.0, 0.0, 0
		for xs, ys, attrs in train_loader:
			xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
			optimizer.zero_grad()
			ypreds, apreds = net(xs, ys)
			# Compute both the prediction loss and the adversarial loss
			loss = F.mse_loss(ypreds, ys)
			adv_loss = F.nll_loss(apreds, attrs) 
			running_loss += loss.item() * len(ys)
			running_adv_loss += adv_loss.item() * len(ys)
			total += len(ys)
			loss += mu * adv_loss
			loss.backward()
			optimizer.step()
		running_loss = running_loss / total
		running_adv_loss = running_adv_loss / total
		logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
	time_end = time.time()
	logger.info("Time used for training = {} seconds.".format(time_end - time_start))
	# inference
	net.eval()
	running_loss, total = 0.0, 0
	ypreds_numpy, ys_numpy, attrs_numpy = [], [], []
	for xs, ys, attrs in test_loader:
		xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
		ypreds = net.inference(xs)
		loss = F.mse_loss(ypreds, ys)
		# logging and saving 
		running_loss += loss.item() * len(ys)
		total += len(ys)
		ypreds_numpy.append(ypreds.detach().cpu().numpy())
		ys_numpy.append(ys.cpu().numpy())
		attrs_numpy.append(attrs.cpu().numpy())
	# summation and logging
	running_loss = running_loss / total
	ypreds_numpy = np.concatenate(ypreds_numpy, axis=0).squeeze()
	ys_numpy = np.concatenate(ys_numpy, axis=0).squeeze()
	attrs_numpy = np.concatenate(attrs_numpy, axis=0)
	cls_error, error_0, error_1 = conditional_mse_errors(ypreds_numpy, ys_numpy, attrs_numpy)
	logger.info("Inference, loss value = {}".format(running_loss))
	logger.info("Overall predicted error = {}, Err|A=0 = {}, Err|A=1 = {}".format(cls_error, error_0, error_1))
	logger.info("Error gap = {}".format(np.abs(error_0-error_1)))
	ys_var = np.var(ys_numpy) 
	r_squared = 1 - cls_error/ys_var
	logger.info("R squared = {}".format(r_squared))
	nmse = cls_error/ys_var
	# save data to csv
	csv_data = {"cls_error": cls_error,
				"error_0": error_0,
				"error_1": error_1,
				"err_gap": np.abs(error_0-error_1),
				"R^2": r_squared,
				"nmse": nmse
				}
	csv_fn = args.name + ".csv"
	with open(csv_fn, "a") as csv_file:
		fieldnames = ["cls_error", "error_0", "error_1", "err_gap", "R^2", "nmse"]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		if os.path.exists(csv_fn):
			pass # no need to write headers
		else:
			writer.writeheader()
		writer.writerow(csv_data)
	# save prediction to npy file
	npy_fn = args.name + "_seed_%d"% args.seed + ".npz"
	np.savez(npy_fn, 
			 y_pred=ypreds_numpy, 
			 y_true=ys_numpy, 
			 A_true=attrs_numpy)
else:
	raise NotImplementedError