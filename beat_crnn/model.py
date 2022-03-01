import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import beat_crnn.config as config


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_gru(rnn):
    """Initialize a GRU layer."""

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class AcousticModelCRnn(nn.Module):
    def __init__(self, classes_num, midfeat):
        super(AcousticModelCRnn, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=16)
        self.conv_block2 = ConvBlock(in_channels=16, out_channels=16)
        self.conv_block3 = ConvBlock(in_channels=16, out_channels=32)
        self.conv_block4 = ConvBlock(in_channels=32, out_channels=32)

        self.fc5 = nn.Linear(midfeat, 512, bias=False)
        self.bn5 = nn.BatchNorm1d(512)

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output


class Beat_CRNN(nn.Module):
    def __init__(self):
        super(Beat_CRNN, self).__init__()

        # fmin = 30
        # fmax = sample_rate // 2

        window = "hann"
        center = True
        pad_mode = "reflect"
        top_db = None
        mel_bins = 128
        midfeat = 256

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.WINDOW_SIZE,
            hop_length=config.HOP_SIZE,
            win_length=config.WINDOW_SIZE,
            window=window,
            center=True,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=config.SAMPLE_RATE,
            n_fft=config.WINDOW_SIZE,
            n_mels=mel_bins,
            top_db=top_db,
            freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.crnn = AcousticModelCRnn(1, midfeat)
        self.db_crnn = AcousticModelCRnn(1, midfeat)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Args:
          input: (batch_size, number_of_samples)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        beats_output = self.crnn(x)  # (batch_size, time_steps, 1)
        db_output = self.db_crnn(x)

        return beats_output, db_output
