import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from pdb import set_trace


class LordModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.content_embedding = RegularizedEmbedding(config['n_imgs'], config['content_dim'], config['content_std'])
		self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
		self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
		self.generator = Generator(config['content_dim'], config['n_adain_layers'], config['adain_dim'], config['img_shape'])

	def forward(self, img_id, class_id):
		content_code = self.content_embedding(img_id)
		class_code = self.class_embedding(class_id)
		class_adain_params = self.modulation(class_code)
		generated_img = self.generator(content_code, class_adain_params)

		return generated_img, content_code, class_code

	def init(self):
		self.apply(self.weights_init)

	@staticmethod
	def weights_init(m):
		if isinstance(m, nn.Embedding):
			nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class RegularizedEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, stddev):
		super().__init__()

		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		self.stddev = stddev

	def forward(self, x):
		x = self.embedding(x)

		if self.training and self.stddev != 0:
			noise = torch.zeros_like(x)
			noise.normal_(mean=0, std=self.stddev)

			x = x + noise

		return x


class Modulation(nn.Module):

	def __init__(self, code_dim, n_adain_layers, adain_dim):
		super().__init__()

		self.__n_adain_layers = n_adain_layers
		self.__adain_dim = adain_dim

		self.adain_per_layer = nn.ModuleList([
			nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
			for _ in range(n_adain_layers)
		])

	def forward(self, x):
		adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
		adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

		return adain_params


class Generator(nn.Module):

	def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
		super().__init__()

		self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
		self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
		self.__adain_dim = adain_dim

		self.fc_layers = nn.Sequential(
			nn.Linear(
				in_features=content_dim,
				out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
			),

			nn.LeakyReLU(),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
				out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
			),

			nn.LeakyReLU(),

			nn.Linear(
				in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
				out_features=self.__initial_height * self.__initial_width * adain_dim
			),

			nn.LeakyReLU()
		)

		self.adain_conv_layers = nn.ModuleList()
		for i in range(n_adain_layers):
			self.adain_conv_layers += [
				nn.Upsample(scale_factor=(2, 2)),
				nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
				nn.LeakyReLU(),
				AdaptiveInstanceNorm2d(adain_layer_idx=i)
			]

		self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

		self.last_conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
			nn.LeakyReLU(),

			nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
			nn.Sigmoid()
		)

	def assign_adain_params(self, adain_params):
		for m in self.adain_conv_layers.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				m.bias = adain_params[:, m.adain_layer_idx, :, 0]
				m.weight = adain_params[:, m.adain_layer_idx, :, 1]

	def forward(self, content_code, class_adain_params):
		self.assign_adain_params(class_adain_params)

		x = self.fc_layers(content_code)
		x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
		x = self.adain_conv_layers(x)
		x = self.last_conv_layers(x)

		return x


class AdaptiveInstanceNorm2d(nn.Module):

	def __init__(self, adain_layer_idx):
		super().__init__()
		self.weight = None
		self.bias = None
		self.adain_layer_idx = adain_layer_idx

	def forward(self, x):
		b, c = x.shape[0], x.shape[1]

		x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
		weight = self.weight.contiguous().view(-1)
		bias = self.bias.contiguous().view(-1)

		out = F.batch_norm(
			x_reshaped, running_mean=None, running_var=None,
			weight=weight, bias=bias, training=True
		)

		out = out.view(b, c, *x.shape[2:])
		return out


class NetVGGFeatures(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vggnet = models.vgg16(pretrained=True)
		self.layer_ids = layer_ids

	def forward(self, x):
		output = []
		for i in range(self.layer_ids[-1] + 1):
			x = self.vggnet.features[i](x)

			if i in self.layer_ids:
				output.append(x)

		return output


class VGGDistance(nn.Module):

	def __init__(self, layer_ids):
		super().__init__()

		self.vgg = NetVGGFeatures(layer_ids)
		self.layer_ids = layer_ids

	def forward(self, I1, I2):
		b_sz = I1.size(0)
		
		if I1.size(1) == 1:
			I1 = I1.repeat(1, 3, 1, 1)
			I2 = I2.repeat(1, 3, 1, 1)
		
		f1 = self.vgg(I1)
		f2 = self.vgg(I2)

		loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)

		for i in range(len(self.layer_ids)):
			layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
			loss = loss + layer_loss

		return loss.mean()
