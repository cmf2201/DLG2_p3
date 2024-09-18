import os
from contextlib import redirect_stdout

from adversarial_models import CreateDistillWeight

import warnings
warnings.simplefilter('ignore')

with redirect_stdout(open(os.devnull, 'w')):
    CreateDistillWeight('../repository/release-StereoUnsupFt-Mono-pt-CK.ckpt')
