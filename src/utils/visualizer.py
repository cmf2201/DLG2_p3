import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np
import cv2


class Extractor():
    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        x = Variable(x, requires_grad=True)
        x.register_hook(self.save_gradient)
        output = self.model(x)
        return output[0]


class InputGradient:
    def __init__(self, model, h, w):
        self.model = model
        self.model.eval()
        self.extractor = Extractor(self.model)
        self.height = h
        self.width = w

    def __call__(self, input_img, y, x):
        output = self.extractor(input_img.cuda())

        one_hot = np.zeros((self.height, self.width), dtype=np.float32)
        one_hot[y][x] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output[0, y, x])

        self.model.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        return grads_val[0]


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def get_activations(self):
        return self.feature_extractor.activations

    def __call__(self, x):
        x = Variable(x, requires_grad=True)
        output = self.feature_extractor(x)
        return output[0]


class FeatureExtractor(nn.Module):
    def __init__(self, model, target_layer):
        super(FeatureExtractor, self).__init__()
        self.model = model

    def init(self):
        self.gradients = []
        self.activations = []
        self.handles = {}

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            self.activations.insert(0, output)

        self.handles['f_conv1'] = self._modules['model'].conv1.register_forward_hook(forward_hook)
        self.handles['f_conv2'] = self._modules['model'].conv2.register_forward_hook(forward_hook)
        self.handles['f_conv3'] = self._modules['model'].conv3.register_forward_hook(forward_hook)
        self.handles['f_conv4'] = self._modules['model'].conv4.register_forward_hook(forward_hook)
        self.handles['f_conv5'] = self._modules['model'].conv5.register_forward_hook(forward_hook)
        self.handles['b_conv1'] = self._modules['model'].conv1.register_backward_hook(backward_hook)
        self.handles['b_conv2'] = self._modules['model'].conv2.register_backward_hook(backward_hook)
        self.handles['b_conv3'] = self._modules['model'].conv3.register_backward_hook(backward_hook)
        self.handles['b_conv4'] = self._modules['model'].conv4.register_backward_hook(backward_hook)
        self.handles['b_conv5'] = self._modules['model'].conv5.register_backward_hook(backward_hook)

    def remove(self):
        self.gradients = []
        self.activations = []
        for i, j in self.handles.items():
            self.handles[i].remove()

    def upsample_nn_nearest(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

    def __call__(self, x):
        conv1 = self.model.conv1(x)
        conv2 = self.model.conv2(conv1)

        conv3 = self.model.conv3(conv2)
        conv4 = self.model.conv4(conv3)
        conv5 = self.model.conv5(conv4)

        skip1 = conv1
        skip2 = conv2
        skip3 = conv3
        skip4 = conv4

        upconv4 = self.model.upconv4(conv5)  # H/16
        concat4 = torch.cat((upconv4, skip4), 1)
        iconv4 = self.model.iconv4(concat4)

        upconv3 = self.model.upconv3(iconv4)  # H/8
        concat3 = torch.cat((upconv3, skip3), 1)
        iconv3 = self.model.iconv3(concat3)
        disp3 = self.model.disp3(iconv3)
        disp3up = self.upsample_nn_nearest(disp3)

        upconv2 = self.model.upconv2(iconv3)  # H/4
        concat2 = torch.cat((upconv2, skip2, disp3up), 1)
        iconv2 = self.model.iconv2(concat2)
        disp2 = self.model.disp2(iconv2)
        disp2up = self.upsample_nn_nearest(disp2)

        upconv1 = self.model.upconv1(iconv2)  # H/2
        concat1 = torch.cat((upconv1, skip1, disp2up), 1)
        iconv1 = self.model.iconv1(concat1)
        disp1 = self.model.disp1(iconv1)
        disp1up = self.upsample_nn_nearest(disp1)

        upconv0 = self.model.upconv0(iconv1)
        concat0 = torch.cat((upconv0, disp1up), 1)
        iconv0 = self.model.iconv0(concat0)
        disp0 = self.model.disp0(iconv0)

        x = disp0*20

        return x


class ActivationMap:
    def __init__(self, model, layer, h, w):
        super(ActivationMap, self).__init__()
        self.model = model
        self.model.eval()
        self.target_layer = layer
        self.extractor = ModelOutputs(self.model, self.target_layer)
        self.height = h
        self.width = w

    def normalize_map(maps, h, w):
        maps = np.maximum(maps, 0)
        maps = cv2.resize(maps, (w, h))
        maps_min, maps_max = maps.min(), maps.max()
        maps = (maps-maps_min)/(maps_max-maps_min)
        maps = np.reshape(maps, (1, h, w))
        return maps

    def show_maps_on_image(img, maps):
        img = np.transpose(img, (1, 2, 0))
        maps = np.transpose(maps, (1, 2, 0))
        heatmap = cv2.applyColorMap(np.uint8(255*maps), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        out_maps = heatmap + np.float32(img)
        out_maps = out_maps / np.max(out_maps)
        out_maps = np.uint8(255 * out_maps)
        return out_maps

    def __call__(self, input_img, y, x):
        self.extractor.feature_extractor.init()

        output = self.extractor(input_img.cuda())

        one_hot = np.zeros((self.height, self.width), dtype=np.float32)
        one_hot[y][x] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)

        one_hot = torch.sum(one_hot.cuda() * output[0, y, x])

        self.model.model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.extractor.get_gradients()
        activations = self.extractor.get_activations()

        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        for i in range(len(gradients)):
            gradient = gradients[i].cpu().data.numpy()[0]
            activation = activations[i].cpu().data.numpy()[0]

            for i, w in enumerate(gradient):
                active = abs(w) * activation[i, :, :]
                heatmap += cv2.resize(active, (self.width, self.height))

        self.extractor.feature_extractor.remove()

        return heatmap
