import torch
try:
    from .guo.monocular_model import MonocularVGG16
except ImportError:
    from guo.monocular_model import MonocularVGG16


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret
    return wrapper


class AdversarialModels():
    def __init__(self, args):
        self.args = args
        if 'distill' in self.args.model:
            self.distill = DistillModel(require_grad=True).cuda().eval()
            self.fix_distill = DistillModel(require_grad=False).cuda().eval()
            print('=> Load Guo\'s model')

    def load_ckpt(self):
        if hasattr(self, 'distill'):
            ckpt = torch.load(self.args.distill_ckpt)
            self.distill.load_state_dict(ckpt['model'])
            self.fix_distill.load_state_dict(ckpt['model'])
            print('=> Load Guo\'s model weight')

    @make_nograd_func
    def get_original_disp(self, sample):
        add_sample = {}
        if 'distill' in self.args.model:
            distill_disp = self.fix_distill(sample['left'])
            add_sample.update({"original_distill_disp": distill_disp.detach()})
        return add_sample


# Load the copied Guo's network (Learning-Monocular-Depth-by-Stereo-master)
class DistillModel(torch.nn.Module):

    def __init__(self, require_grad=True):
        super(DistillModel, self).__init__()
        self.model = MonocularVGG16(use_pretrained_weights=True)
        self.require_grad = require_grad

        if not self.require_grad:
            for child in self.model.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, input_img):
        disp_ests = self.model(input_img)
        return disp_ests[0]


def CreateDistillWeight(path):
    model = DistillModel().cuda()
    original_model = torch.nn.DataParallel(DistillModel()).cuda()
    state_dict = torch.load(path)
    original_model.load_state_dict(state_dict["model"])

    pretrained_dict = original_model.state_dict()
    model_dict = model.state_dict()
    # 1. replace unnecessary keyname
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    new_weight = 0
    for key, value in model.state_dict().items():
        new_weight += value.float().mean()

    old_weight = 0
    for key, value in original_model.state_dict().items():
        old_weight += value.float().mean()

    if new_weight == old_weight:
        # save_ckpt
        torch.save({
            'model': model.state_dict(),
        }, './guo/distill_model.ckpt')
    else:
        print("Failed to create a newwork weight")
