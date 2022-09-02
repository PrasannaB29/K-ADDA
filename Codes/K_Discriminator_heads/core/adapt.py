"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn

import params
from utils import make_variable


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, label_src), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            # images_src = make_variable(images_src)
            # images_tgt = make_variable(images_tgt)
            if torch.cuda.is_available():
                images_src = images_src.cuda()
                images_tgt = images_tgt.cuda()
            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            ###### EDIT #####
            # label_src = make_variable(torch.ones(feat_src.size(0)).long())
            # label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())

            ##### EDITED FOR JOINT CLASSIFYING DISTRIBUTION ######
            # label_src = torch.ones(feat_src.size(0)).long()
            label_src = label_src.long()
            label_tgt = torch.zeros(feat_tgt.size(0)).long()+10
            if torch.cuda.is_available():
                label_src = label_src.cuda()
                label_tgt = label_tgt.cuda()
                pred_concat = pred_concat.cuda()
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt) # has 11 values per example

            ##### EDITED FOR JOINT DISCRIMINATOR #####
            pred_tgt_exp = torch.exp(pred_tgt)
            pred_tgt_new = torch.zeros((pred_tgt_exp.size()[0], 2))
            pred_tgt_new[:, 0] = pred_tgt_exp[:, :-1].sum(dim=1)
            pred_tgt_new[:, 1] = pred_tgt_exp[:, -1]
            pred_tgt_new = torch.log(pred_tgt_new) # has 2 values per example to enable further usage

            # prepare fake labels
            #### EDITED FOR JOINT DISCRIMINATOR ####
            label_tgt = torch.zeros(feat_tgt.size(0)).long()
            if torch.cuda.is_available():
                pred_tgt_new = pred_tgt_new.cuda()
                label_tgt = label_tgt.cuda()
            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt_new, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            ####### EDIT #########
            # if ((step + 1) % params.log_step == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]:"
            #           "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
            #           .format(epoch + 1,
            #                   params.num_epochs,
            #                   step + 1,
            #                   len_data_loader,
            #                   loss_critic.data[0],
            #                   loss_tgt.data[0],
            #                   acc.data[0]))
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data,
                              loss_tgt.data,
                              acc.data))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder
