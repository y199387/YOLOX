import argparse
from multiprocessing import dummy
import os
from loguru import logger

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from yolox.exp import get_exp

from bigdl.nano.pytorch import InferenceOptimizer

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    model.eval()

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model"])
    logger.info("loading checkpoint done.")

    input_sample = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    accelerated_models = []

    # onnx_model = InferenceOptimizer.trace(model,
    #                                       input_sample=input_sample,
    #                                       accelerator="onnxruntime")

    # ipex_model = InferenceOptimizer.trace(model,
    #                                       input_sample=input_sample,
    #                                       accelerator="jit",
    #                                       use_ipex=True)

    openvino_model = InferenceOptimizer.trace(model,
                                              input_sample=input_sample,
                                              accelerator="openvino")

    


    coco_evaluator = exp.get_evaluator(args.batch_size, False)
    coco_evaluator.per_class_AP = True
    coco_evaluator.per_class_AR = True

    *_, summary = coco_evaluator.evaluate(openvino_model, False)
    import pdb; pdb.set_trace()
    logger.info("\n" + "accelerated model results")
    logger.info("\n" + summary)

if __name__ == "__main__":
    main()