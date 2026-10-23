import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(channel, pred=False):
    if pred:
        return nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
        )
    return nn.Sequential(
        nn.Conv2d(channel[0], channel[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(inplace=True),
    )


def att_layer(channel):
    return nn.Sequential(
        nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
        nn.BatchNorm2d(channel[2]),
        nn.Sigmoid(),
    )


class MTANSegNet(nn.Module):
    def __init__(self, n_tasks, class_nb):
        super().__init__()
        filter = [64, 128, 256, 512, 512]
        self.n_tasks = n_tasks
        self.class_nb = class_nb

        self.encoder_block = nn.ModuleList([conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(conv_layer([filter[i + 1], filter[i]]))

        self.conv_block_enc = nn.ModuleList([conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(conv_layer([filter[i + 1], filter[i + 1]]), conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(conv_layer([filter[i], filter[i]]), conv_layer([filter[i], filter[i]])))

        self.encoder_att = nn.ModuleList([nn.ModuleList([att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([conv_layer([filter[0], filter[0]])])
        for j in range(n_tasks):
            if j < n_tasks - 1:
                self.encoder_att.append(nn.ModuleList([att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))
        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = conv_layer([filter[0], class_nb], pred=True)
        self.pred_task2 = conv_layer([filter[0], 1], pred=True)
        if n_tasks == 3:
            self.pred_task3 = conv_layer([filter[0], 3], pred=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def shared_parameters(self):
        return (p for n, p in self.named_parameters() if "pred" not in n)

    def task_specific_parameters(self):
        return (p for n, p in self.named_parameters() if "pred" in n)

    def last_shared_parameters(self):
        return []

    def forward(self, x, return_representation=False):
        encoder, decoder, upsampled, indices = [], [], [], []
        h = x
        for i in range(5):
            e0 = self.encoder_block[i](h)
            e1 = self.conv_block_enc[i](e0)
            h, index = self.down_sampling(e1)
            encoder.append((e0, e1))
            indices.append(index)
        for i in range(5):
            up = self.up_sampling(h if i == 0 else decoder[i - 1][1], indices[-i - 1])
            d0 = self.decoder_block[-i - 1](up)
            decoder.append((d0, self.conv_block_dec[-i - 1](d0)))
            upsampled.append(up)

        representation = []
        for t in range(self.n_tasks):
            a = None
            for j in range(5):
                inputs = encoder[j][0] if j == 0 else torch.cat((encoder[j][0], a), dim=1)
                a = self.encoder_att[t][j](inputs) * encoder[j][1]
                a = F.max_pool2d(self.encoder_block_att[j](a), kernel_size=2, stride=2)
            for j in range(5):
                d = self.decoder_block_att[-j - 1](F.interpolate(a, scale_factor=2, mode="bilinear", align_corners=True))
                a = self.decoder_att[t][-j - 1](torch.cat((upsampled[j], d), dim=1)) * decoder[j][1]
            representation.append(a)

        predictions = [F.log_softmax(self.pred_task1(representation[0]), dim=1), self.pred_task2(representation[1])]
        if self.n_tasks == 3:
            normal = self.pred_task3(representation[2])
            predictions.append(normal / torch.norm(normal, p=2, dim=1, keepdim=True))
        if return_representation:
            return predictions, tuple(representation)
        return predictions


class MultiMNISTViT(nn.Module):
    def __init__(self, img_h=28, img_w=43, patch_size=4, emb_size=16, nhead=2, num_layers=3):
        super().__init__()
        self.emb_size = emb_size
        self.pad_h = (patch_size - img_h % patch_size) % patch_size
        self.pad_w = (patch_size - img_w % patch_size) % patch_size
        num_patches = ((img_h + self.pad_h) // patch_size) * ((img_w + self.pad_w) // patch_size)
        patch_dim = patch_size ** 2
        self.conv = nn.Conv2d(1, patch_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.patch_emb = nn.Linear(patch_dim, emb_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.rand(1, num_patches + 1, emb_size))
        layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=4 * emb_size, dropout=0.0, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head1 = nn.Linear(emb_size, 10)
        self.head2 = nn.Linear(emb_size, 10)

    def shared_parameters(self):
        return [*self.conv.parameters(), *self.patch_emb.parameters(), self.cls_token, self.pos_emb, *self.transformer_enc.parameters()]

    def task_specific_parameters(self):
        return [*self.head1.parameters(), *self.head2.parameters()]

    def forward(self, x, return_representation=False):
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="constant", value=0.0)
        x = self.patch_emb(self.conv(x).flatten(2).transpose(1, 2))
        x = torch.cat([self.cls_token.expand(x.size(0), 1, self.emb_size), x], dim=1) + self.pos_emb
        rep = self.transformer_enc(x)[:, 0, :]
        predictions = (self.head1(rep), self.head2(rep))
        return (predictions, rep) if return_representation else predictions
