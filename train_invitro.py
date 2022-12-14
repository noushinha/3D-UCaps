import os
from argparse import ArgumentParser

import torch
import shutil

from datamodule.invitro import InvitroDataModule
from module.segcaps import SegCaps2D, SegCaps3D
from module.ucaps import UCaps3D
from module.unet import UNetModule
from monai.utils import set_determinism
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Call example
# python train3D.py --gpus 1 --model_name UCaps --num_workers 4 --max_epochs 20000
# --check_val_every_n_epoch 100 --log_dir=../logs --root_dir=/home/ubuntu/

if __name__ == "__main__":
    DSName = "invitro"
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/mnt/Data/Cryo-ET/3D-UCaps/data/" + str(DSName))
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Training options
    train_parser = parser.add_argument_group("Training config")
    train_parser.add_argument("--log_dir", type=str, default="/mnt/Data/Cryo-ET/3D-UCaps/logs")
    train_parser.add_argument("--model_name", type=str, default="ucaps", help="ucaps")
    train_parser.add_argument("--dataset", type=str, default="invitro", help="invitro")
    train_parser.add_argument("--train_patch_size", nargs="+", type=int, default=[64, 64, 64])
    train_parser.add_argument("--fold", type=int, default=0)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument("--num_samples", type=int, default=1,
                              help="Effective batch size: batch_size x num_samples")
    train_parser.add_argument("--balance_sampling", type=int, default=1)
    train_parser.add_argument("--use_class_weight", type=int, default=0)

    parser = Trainer.add_argparse_args(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "ucaps":
        parser, model_parser = UCaps3D.add_model_specific_args(parser)
    elif temp_args.model_name == "segcaps-2d":
        parser, model_parser = SegCaps2D.add_model_specific_args(parser)
    elif temp_args.model_name == "segcaps-3d":
        parser, model_parser = SegCaps3D.add_model_specific_args(parser)
    elif temp_args.model_name == "unet":
        parser, model_parser = UNetModule.add_model_specific_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    print(f"{args.model_name} config:")
    for a in model_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    print("Training config:")
    for a in train_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    # Improve reproducibility
    set_determinism(seed=0)
    seed_everything(0, workers=True)

    # Set up datamodule
    if args.dataset == "invitro":
        data_module = InvitroDataModule(
            **dict_args,
        )
    else:
        pass

    # initialise the LightningModule
    if args.use_class_weight:
        class_weight = torch.tensor(data_module.class_weight).float()
    else:
        class_weight = None

    if args.model_name == "ucaps":
        # checkpoint_path = '/mnt/Data/Cryo-ET/3D-UCaps/logs/ucaps_artificial_0/version_1/checkpoints/epoch=296-val_dice=0.9547.ckpt'
        checkpoint_path = '/mnt/Data/Cryo-ET/3D-UCaps/logs/ucaps_invitro_0/version_15/checkpoints/epoch=349-val_dice=0.7723.ckpt'
        net = UCaps3D.load_from_checkpoint(
            checkpoint_path,
            val_patch_size=[64, 64, 64],
            sw_batch_size=1,
            overlap=0.75,
        )
        substr = "reconstruct_branch"
        # print model summary
        from torchsummary import summary
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = net.to(device)
        print(summary(mymodel, (1, 64, 64, 64)))
        # for param in net.parameters():
        #     param.requires_grad = False
        # params = net.state_dict()
        # keys = list(params.keys())
        for name, param in net.named_parameters():
            if substr not in name:
                if param.requires_grad:
                    param.requires_grad = False
        # net = UCaps3D(**dict_args, class_weight=class_weight)
    elif args.model_name == "segcaps-3d":
        net = SegCaps3D(**dict_args, class_weight=class_weight)
    elif args.model_name == "segcaps-2d":
        net = SegCaps2D(**dict_args, class_weight=class_weight)
    elif args.model_name == "unet":
        net = UNetModule(**dict_args, class_weight=class_weight)

    # set up loggers and checkpoints
    if args.dataset == "iseg2017":
        tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=f"{args.model_name}_{args.dataset}", log_graph=True)
    else:
        tb_logger = TensorBoardLogger(save_dir=args.log_dir,
                                      name=f"{args.model_name}_{args.dataset}_{args.fold}", log_graph=True)
    checkpoint_callback = ModelCheckpoint(filename="{epoch}-{val_dice:.4f}", monitor="val_dice",
                                          save_top_k=2, mode="max", save_last=True)
    earlystopping_callback = EarlyStopping(monitor="val_dice", patience=50, mode="max")
    outpath = args.log_dir
    shutil.copyfile("module/ucaps.py", os.path.join(outpath, "ucaps_" + str(DSName) + ".txt"))
    shutil.copyfile("datamodule/invitro.py", os.path.join(outpath, "datamodule_" + str(DSName) + ".txt"))
    shutil.copyfile("train_" + str(DSName) + ".py", os.path.join(outpath, "train_" + str(DSName) + ".txt"))
    shutil.copyfile("data/" + str(DSName) + "/dataset.json", os.path.join(outpath, "json_" + str(DSName) + ".txt"))

    trainer = Trainer.from_argparse_args(
        args,
        # benchmark=True,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback],
        num_sanity_val_steps=1,
        terminate_on_nan=True,
        gpus=1, max_epochs=10
        # resume_from_checkpoint="/mnt/Data/Cryo-ET/3D-UCaps/logs/ucaps_artificial_0/version_1/checkpoints/epoch=296-val_dice=0.9547.ckpt"
    )

    trainer.fit(net, datamodule=data_module)
    print("Best model path ", checkpoint_callback.best_model_path)
    print("Best val mean dice ", checkpoint_callback.best_model_score.item())

