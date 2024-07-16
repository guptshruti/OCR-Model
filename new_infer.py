from __future__ import division, print_function

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import Compose

import datasets.dataset as dataset
import utils
# from config import *
from datasets.ae_transforms import *
from datasets.imprint_dataset import Rescale as IRescale
from models.model import ModelBuilder
from utils import get_vocabulary

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.enabled = True


		# self.test_data = dataset.ImageDataset(
		# 	root=self.test_root,
		# 	voc=self.opt.alphabet,
		# 	transform=self.test_transforms,
		# 	voc_type='file',
		# 	return_list=True
		# )


    # def __init__(self, root, voc, num_samples=np.inf,
    #              transform=None, label_transform=None,
    #              voc_type='file', lowercase=False,
    #              alphanumeric=False, ctc_blank='<b>',
    #              return_list=True):

class BaseHTR(object):
	def __init__(self, opt):
		self.opt = opt
		



		# label_transform = False
		self.voc = self.opt.alphabet
		print("self.alphabet:", self.opt.alphabet)
		# self.transform = transform
		# self.label_transform = label_transform
		voc_type = 'file'
		lowercase = False
		alphanumeric = False
        # self.nSamples = len(self.get_images(root))
        # self.nSamples = int(float(self.txn.get(b"num-samples")))
        # self.nSamples = min(self.nSamples, num_samples)
		self.voc = get_vocabulary(self.voc, voc_type, lowercase, alphanumeric)
		print("self.voc:", self.voc)
		self.char2id = dict(zip(self.voc, range(1, len(self.voc)+1))) # 0 reserved for ctc blank
		self.id2char = dict(zip(range(1, len(self.voc)+1), self.voc))
		print("self.char2id:", self.char2id)
		ctc_blank='<b>'
		self.char2id[ctc_blank] = 0
		self.id2char[0] = ctc_blank
		self.ctc_blank = ctc_blank
		# self.lowercase = lowercase
		# self.alphanumeric = alphanumeric
		self.rec_num_classes = len(self.id2char)
		# self.return_list = return_list












		self.test_transforms = Compose([
			IRescale(max_width=256, height=96),
			ToTensor()
		])
		self.identity_matrix = torch.tensor(
			[1, 0, 0, 0, 1, 0],
			dtype=torch.float
		).cuda()
		
		##################################################################
		self.test_root = opt.test_root

		if torch.cuda.is_available() and not self.opt.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			self.opt.gpu_id = list(map(int, self.opt.gpu_id.split(',')))
			torch.cuda.set_device(self.opt.gpu_id[0])

		# self.test_data = dataset.ImageDataset(
		# 	root=self.test_root,
		# 	voc=self.opt.alphabet,
		# 	transform=self.test_transforms,
		# 	voc_type='file',
		# 	return_list=True
		# )
        
		self.converter = utils.strLabelConverter(
			self.id2char,
			self.char2id,
			self.ctc_blank
		)
		self.nclass = self.rec_num_classes

		crnn = ModelBuilder(
			96, 256,
			[48,128], [96,256],
			20, [0.05, 0.05],
			'none',
			256, 1, 1,
			self.nclass,
			STN_type='TPS',
			nheads=1,
			stn_attn=None,
			use_loc_bn=False,
			loc_block = 'LocNet',
			CNN='ResCRNN'
		)
		if self.opt.cuda:
			crnn.cuda()
			print("self.opt.gpu_id", self.opt.gpu_id)
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id, dim=1)
		else:
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id)
		print('Using pretrained model', self.opt.pretrained)
		crnn.load_state_dict(torch.load(self.opt.pretrained))
		self.model = crnn
		self.model.eval()
		print('Model loading complete')

		self.init_variables()
		print('Classes: ', self.voc)
		# print('#Test Samples: ', self.nSamples)

		# data_loader = torch.utils.data.DataLoader(
		# 	self.test_data,
		# 	batch_size=64,
		# 	num_workers=2,
		# 	pin_memory=True,
		# 	collate_fn=dataset.collatedict(),
		# 	drop_last=False
		# )
		gts = []
		decoded_preds = []
		# val_iter = iter(data_loader)
		# max_iter = min(np.inf, len(data_loader))
		image = Image.open("data/english/eng_hand.jpg").convert('L')
		image = self.test_transforms(image)
		print("image:", image)
		print("image shpae:", image.shape)
		cpu_images = image[None, :, :, :]
		print("cpu_images:", cpu_images.shape)
		cpu_texts = "image_new"
		with torch.no_grad():
			# for i in range(max_iter):
				# cpu_images, cpu_texts = next(val_iter)
			utils.loadData(self.image, cpu_images)
			output_dict = self.model(self.image)
			batch_size = cpu_images.size(0)

			preds = F.log_softmax(output_dict['probs'], 2)

			preds_size = torch.IntTensor([preds.size(0)] * batch_size)
			_, preds = preds.max(2)
			preds = preds.transpose(1, 0).contiguous().view(-1)
			decoded_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

			gts += list(cpu_texts)
			decoded_preds += list(decoded_pred)
			print(decoded_preds)
			print("decoded_preds:", ''.join(decoded_preds))
		directory = self.opt.out_dir
		writepath1 = directory + '/' + "output" + ".txt" 
		print(writepath1)
		with open(writepath1, 'w', encoding='utf-8') as f1:
			for target, pred in zip(gts, decoded_preds):         
				print(target, pred)
				f1.write(str(target))
				f1.write("\t")
				f1.write(str(pred))
				f1.write("\n") 
		return


	def init_variables(self):
		self.image = torch.FloatTensor(64, 3, 96, 256)
		self.text = torch.LongTensor(64 * 5)
		self.length = torch.LongTensor(64)
		if self.opt.cuda:
			self.image = self.image.cuda()
			self.text = self.text.cuda()
			self.length = self.length.cuda()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_root', help='path to dataset')
	parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
	parser.add_argument('--gpu_id', type=str, default='0', help='gpu device ids')
	parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz*')
	parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
	parser.add_argument('--out_dir', type=str, default="out", help='predictions folder')
	parser.add_argument('--language', help='path to dataset')

	opt = parser.parse_args()
	opt.alphabet = f'{opt.language}_lexicon.txt'
	print("alphabet: ",opt.alphabet)
	obj = BaseHTR(opt)
	# obj.run()
