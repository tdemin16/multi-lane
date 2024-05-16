from .cifar100 import get_args_parser as cifar100_parser
from .imr import get_args_parser as imr_parser
from .coco import get_args_parser as coco_parser
from .voc import get_args_parser as voc_parser

CONFIGS = {
    'cifar100': cifar100_parser,
    'imr': imr_parser,
    'coco': coco_parser,
    'voc': voc_parser,
}