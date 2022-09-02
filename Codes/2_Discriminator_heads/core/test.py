"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    with torch.no_grad():
        for (images, labels) in data_loader:
            # images = make_variable(images, volatile=True)
            # labels = make_variable(labels).squeeze_()
            labels=labels.squeeze_()
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            preds = classifier(encoder(images))
            # if torch.cuda.is_available():
            #     preds = preds.cuda()
            loss += criterion(preds, labels).data

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum()

    acc = float(acc)
    loss /= float(len(data_loader))
    acc /= float(len(data_loader.dataset))

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
