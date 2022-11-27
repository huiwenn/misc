
import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import argparse
import joblib
from joblib import Parallel, delayed
import tensorflow as tf
from termcolor import cprint
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.baseline_utils as baseline_utils
from torch.utils.data import IterableDataset, Dataset

global save_dir

def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=512,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=8,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=12,
                        type=int,
                        help="Prediction Horizon")
                        
    parser.add_argument("--name",
                        default='lstm',
                        type=str,
                        help="model name")
    parser.add_argument(
        "--translation",
        action="store_true",
        help="translation normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--rotation",
        action="store_true",
        help="rotationally normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="rotationally normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--mis_metric",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512, #512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512, #512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=1000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()

use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 6, 12]

class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


class ModelUtils:
    """Utils for LSTM baselines."""
    def save_checkpoint(self, save_dir: str, state: Dict[str, Any]) -> None:
        """Save checkpoint file.

        Args:
            save_dir: Directory where model is to be saved
            state: State of the model

        """
        filename = "{}/LSTM_rollout{}.pth.tar".format(save_dir,
                                                      state["rollout_len"])
        torch.save(state, filename)

    def load_checkpoint(
            self,
            checkpoint_file: str,
            encoder: Any,
            decoder: Any,
            encoder_optimizer: Any,
            decoder_optimizer: Any,
    ) -> Tuple[int, int, float]:
        """Load the checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model

        Returns:
            epoch: epoch when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved

        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
            '''
            if use_cuda:
                from collections import OrderedDict

                encoder.module.load_state_dict(
                    checkpoint["encoder_state_dict"])
                decoder.module.load_state_dict(
                    checkpoint["decoder_state_dict"])
            else:
            '''
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])

            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return epoch, rollout_len, best_loss

    def my_collate_fn(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: Batch data

        Returns:
            input, output and helpers in the format expected by DataLoader

        """
        _input, output, helpers = [], [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
            #helpers.append(item[2])
        _input = torch.stack(_input)
        output = torch.stack(output)
        return [_input, output, helpers]

    def init_hidden(self, batch_size: int,
                    hidden_size: int) -> Tuple[Any, Any]:
        """Get initial hidden state for LSTM.

        Args:
            batch_size: Batch size
            hidden_size: Hidden size of LSTM

        Returns:
            Initial hidden states

        """
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device),
        )


class Logger(object):
    """Tensorboard logger class."""
    def __init__(self, log_dir: str, name: str = None):
        """Create a summary writer logging to log_dir.

        Args:
            log_dir: Directory where tensorboard logs are to be saved.
            name: Name of the sub-folder.

        """
        if name is None:
            name = "temp"
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            self.writer = tf.summary.create_file_writer(os.path.join(
                log_dir, name),
                filename_suffix=name)
        else:
            self.writer = tf.summary.create_file_writer(log_dir,
                                                        filename_suffix=name)

    def scalar_summary(self, tag: str, value: float, step: int):
        """Log a scalar variable.

        Args:
            tag: Tag for the variable being logged.
            value: Value of the variable.
            step: Iteration step number.

        """
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)


def train(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global global_step
    mse = nn.MSELoss()

    batches = 0
    for i, (_input, target, helpers) in enumerate(train_loader):
        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize losses
        loss = 0
        msel = 0
        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = torch.cat([encoder_input[:, :2], torch.zeros(batch_size,3,device=encoder_input.device)], dim=1)
        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden
        output_shape = list(target.shape)
        output_shape[-1] = 5
        decoder_outputs = torch.zeros(output_shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(rollout_len):

            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            output = Gaussian2d(decoder_output)

            decoder_outputs[:, di, :] = output

            # Update loss
            loss += criterion(output[:, :5], target[:, di, :2])
            msel += mse(output[:, :2], target[:, di, :2])
            # Use own predictions as inputs at next step
            decoder_input = output

        # Get average loss for pred_len
        loss = loss / rollout_len
        msel = msel / rollout_len

        # Backpropagate
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, loss:{loss}, mse:{msel}, Rollout:{rollout_len}")

            logger.scalar_summary(tag="Train/loss",
                                  value=loss.item(),
                                  step=epoch)

        global_step += 1
        
        # TODO change back
        #if i >=100:
        #    break


def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        logger: Logger,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        prev_loss: float,
        decrement_counter: int,
        rollout_len: int = 30,
) -> Tuple[float, int]:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        logger: Tensorboard logger
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon

    """
    args = parse_arguments()
    global best_loss
    total_loss = []
    ades = []
    fdes = np.array([])
    sample_de_all =[]
    nll = []
    mis = []
    es = []
    cov = []
    cov_1s = []
    cov_2s = []
    cov_3s = []
    for i, (_input, target, helpers) in enumerate(val_loader):
        # TODO: change back
        if i >= 100:
            break

        _input = _input.to(device)
        target = target.to(device)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize loss
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = torch.cat([encoder_input[:, :2], torch.zeros(batch_size,3,device=encoder_input.device)], dim=1)
        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden
        output_shape = list(target.shape)
        output_shape[-1] = 5
        decoder_outputs = torch.zeros(output_shape).to(device)

        # Decode hidden state in future trajectory
        de = []
        samplede = []
        miss = []
        covv = []
        nlls = []
        ess = []

        for di in range(rollout_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)

            output = Gaussian2d(decoder_output)

            decoder_outputs[:, di, :] = output

            # Update loss
            #print('output', output[:, :2])
            #print('target', target[:, di, :2])
            loss += criterion(output[:, :5], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = output
            de.append(torch.sqrt((decoder_output[:, 0] - target[:, di, 0])**2 +
                               (decoder_output[:, 1] - target[:, di, 1])**2).detach().cpu().numpy())
            
            covs = make_cov(output)
            p = torch.distributions.MultivariateNormal(decoder_output[:, :2], covs)
            sample = p.sample(sample_shape=(6,))
            gt_expand = target[:, di, :2].repeat((6, 1, 1))

            des = torch.sqrt((sample[:,:,0] - gt_expand[:,:,0])**2 +
                               (sample[:,:,1] - gt_expand[:,:,1])**2).detach().cpu().numpy()
            samplede.append(des)

            nlls.append(nll_loss_2(output[:, :5], target[:, di, :2]).detach().cpu().numpy())
            miss.append(quantile_loss(output[:, :5], target[:, di, :2]).detach().cpu().numpy())
            covv.append(get_coverage(output[:, :5], target[:, di, :2]).detach().cpu().numpy())
            # Use own predictions as inputs at next step
            es_ = torch.norm(output[:, :2] - target[:, di, :2], dim=-1)
            es_ += -1 *(torch.pow(output[:, 2],2) + torch.pow(output[:, 3],2))
            ess.append(es_.detach().cpu().numpy())

        # Get average loss for pred_len
        loss = loss / rollout_len
        total_loss.append(loss)
        ade = np.mean(np.array(de))
        ades.append(ade)
        fde = de[-1]
        sample_de_all.append(np.array(samplede))


        fdes = np.concatenate([fdes,fde])
        if rollout_len>=output_length:
            cov_1s = np.concatenate([cov_1s,covv[3]])
            cov_2s = np.concatenate([cov_2s,covv[7]])
            cov_3s = np.concatenate([cov_3s,covv[11]])
        nll.append(np.mean(nlls))
        mis.append(np.mean(miss))
        cov.append(np.mean(covv))
        es.append(np.mean(ess))



    # Save
    val_loss = sum(total_loss) / len(total_loss)
    ade = sum(ades)/len(ades)
    fde = np.mean(fdes)
    mrs = np.mean(mis)
    nll = np.mean(nll)
    cov = np.mean(cov)
    cov1s = np.mean(cov_1s)
    cov2s = np.mean(cov_2s)
    cov3s = np.mean(cov_3s)
    es = np.mean(es)

    sample_de_all = np.array(sample_de_all[:-1])
    minade = np.mean(np.min(sample_de_all, axis=2))
    minfde = np.mean(np.min(sample_de_all[:,-1,:,:],axis=1))
    cprint(
        f"Val -- Epoch:{epoch}, loss:{val_loss}, ade:{ade}, fde:{fde}, mis:{mrs}, es:{es} \
        cov:{cov}, 1s:{cov1s}, 2s:{cov2s}, 3s:{cov3s}, nll:{nll},  minade:{minade}, minfde:{minfde}, Rollout: {rollout_len}",
        color="green",
    )

    if val_loss <= best_loss:
        best_loss = val_loss
        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            },
        )

    logger.scalar_summary(tag="Val/loss", value=val_loss.item(), step=epoch)

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM Baselines."""
    def __init__(self, data_path: str, args, shuffle=True):
        """Initialize the Dataset.

        Args:
            data_dict: Dict containing all the data
            args: Arguments passed to the baseline code
            mode: train/val/test mode

        """
        '''
        with open(data_path, 'rb') as f:
            data_dict = pkl.load(f)

        # Get input
        wholetraj = np.concatenate([data_dict["input"],data_dict["output"]],axis=1)
        '''
        with open(data_path, 'rb') as f:
            wholetraj = np.load(f)
        
        print('normalizing')
        if args.rotation:
            normalized = baseline_utils.full_norm(wholetraj, args)
            new_path = data_path[:-4] + '_rotation.npy'
            with open(new_path, 'wb') as f:
                np.save(f, normalized)
        elif args.translation:
            normalized = baseline_utils.translation_norm(wholetraj) #.full_norm(wholetraj, args)#
            new_path = data_path[:-4] + '_transi.npy'
            with open(new_path, 'wb') as f:
                np.save(f, normalized)
        else:
            normalized = wholetraj

        self.input_data = normalized[:, :args.obs_len, :]
        self.output_data = normalized[:, args.obs_len:, :]
        self.data_size = self.input_data.shape[0]
        #print(self.input_data[:2])
        #print(self.output_data[:2])

    def __len__(self):
        """Get length of dataset.

        Returns:
            Length of dataset

        """
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[torch.FloatTensor, Any, Dict[str, np.ndarray]]:
        """Get the element at the given index.

        Args:
            idx: Query index

        Returns:
            A list containing input Tensor, Output Tensor (Empty if test) and viz helpers.

        """
        return (
            torch.FloatTensor(self.input_data[idx]),
            torch.FloatTensor(
                self.output_data[idx])
        )


def nll_loss_2(pred: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Negative log loss for single-variate gaussian, can probably be faster"""
    x_mean = pred[:, 0]
    y_mean = pred[:, 1]
    x_delta = x_mean - data[:, 0]
    y_delta = y_mean - data[:, 1]
    x_sigma = pred[:, 2]
    y_sigma = pred[:, 3]
    rho = pred[:, 4]

    root_det_epsilon = torch.pow(1-torch.pow(rho,2), 0.5) * x_sigma * y_sigma

    loss = torch.log(2*3.14159*root_det_epsilon) \
           + 0.5 * torch.pow(root_det_epsilon, -2) \
           * (torch.pow(x_sigma, 2) * torch.pow(y_delta, 2) \
              + torch.pow(y_sigma, 2) * torch.pow(x_delta, 2) \
              - 2 * rho * x_sigma * y_sigma * x_delta * y_delta)

    return torch.mean(loss)


def Gaussian2d(x: torch.Tensor) -> torch.Tensor :
    """Computes the parameters of a bivariate 2D Gaussian."""
    x_mean  = x[:, 0]
    y_mean  = x[:, 1]
    sigma_x = torch.exp(x[:, 2]) #not inverse, see if it works
    sigma_y = torch.exp(x[:, 3]) #not inverse
    rho     = torch.tanh(x[:, 4])
    return torch.stack([x_mean, y_mean, sigma_x, sigma_y, rho], dim=1)

def make_cov(x):
    sigma_x = x[:,2].pow(2).unsqueeze(-1)
    sigma_y = x[:,3].pow(2).unsqueeze(-1)
    off_diag = (x[:,4]*x[:,2]*x[:,3]).unsqueeze(-1)
    cov = torch.cat((sigma_x, off_diag, off_diag, sigma_y), 1)
    return cov.reshape(cov.shape[0], 2, 2)

def quantile_loss(pred: torch.Tensor, data: torch.Tensor, alpha=0.9):

    x_mean = pred[:, 0]
    y_mean = pred[:, 1]

    x_delta = x_mean - data[:, 0]
    y_delta = y_mean - data[:, 1]
    x_sigma = pred[:, 2]
    y_sigma = pred[:, 3]
    rho = pred[:, 4]

    ohr = torch.pow(1-torch.pow(rho, 2), 0.5)

    root_det_epsilon = ohr * x_sigma * y_sigma

    c_alpha = - 2 * np.log(1 - alpha)

    c_ =  (torch.pow(x_sigma, 2) * torch.pow(y_delta, 2) \
           + torch.pow(y_sigma, 2) * torch.pow(x_delta, 2) \
           - 2 * rho * x_sigma * y_sigma * x_delta * y_delta) * torch.pow(root_det_epsilon, -2)#c prime


    c_delta = c_ - c_alpha
    c_delta = torch.where(c_delta > 0, c_delta, torch.zeros_like(c_delta))

    mrs = root_det_epsilon * (c_alpha + c_delta/alpha)
    return torch.mean(mrs)

def get_coverage(pred: torch.Tensor, data: torch.Tensor, alpha=0.9):
    x_mean = pred[:, 0]
    y_mean = pred[:, 1]

    x_delta = x_mean - data[:, 0]
    y_delta = y_mean - data[:, 1]
    x_sigma = pred[:, 2]
    y_sigma = pred[:, 3]
    rho = pred[:, 4]

    ohr = torch.pow(1-torch.pow(rho, 2), 0.5)

    root_det_epsilon = ohr * x_sigma * y_sigma

    c_alpha = - 2 * np.log(1 - alpha)

    c_ = (torch.pow(x_sigma, 2) * torch.pow(y_delta, 2) \
           + torch.pow(y_sigma, 2) * torch.pow(x_delta, 2) \
           - 2 * rho * x_sigma * y_sigma * x_delta * y_delta) * torch.pow(root_det_epsilon, -2)#c prime

    c_delta = c_ - c_alpha
    cover = torch.where(c_delta > 0, torch.ones(c_.shape, device=c_.device), torch.zeros(c_.shape, device=c_.device))
    return cover

def main():
    """Main."""
    args = parse_arguments()
    global save_dir
    save_dir = 'checkpoints/' + args.name
    model_utils = ModelUtils()

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    os.makedirs(save_dir, exist_ok=True)
    # key for getting feature set
    # Get features

    baseline_key = "none"

    # Get model
    if args.mis_metric:
        criterion = quantile_loss
    else:
        criterion = nll_loss_2

    encoder = EncoderRNN(input_size=2)
    decoder = DecoderRNN(output_size=5)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        print('loading model')
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        if rollout_len<12:
            start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1
        else:
            start_rollout_idx = ROLLOUT_LENS.index(rollout_len)
            print("start rollout index", start_rollout_idx)

    else:
        start_epoch = 0
        start_rollout_idx = 0

    # Tensorboard logger
    log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

    # Get PyTorch Dataset
    if not args.test:
        print('loading train dataset')
        train_dataset = LSTMDataset(args.train_features, args)

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )

    print('loading val dataset')
    val_dataset = LSTMDataset(args.val_features, args)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=model_utils.my_collate_fn,
    )

    print("Training begins ...")

    decrement_counter = 0

    epoch = start_epoch
    global_start_time = time.time()
    for i in range(start_rollout_idx, len(ROLLOUT_LENS)):
        rollout_len = ROLLOUT_LENS[i]
        logger = Logger(log_dir, name="{}".format(rollout_len))
        best_loss = float("inf")
        prev_loss = best_loss
        rollout_epoch = 0
        if not args.test:

            while epoch < args.end_epoch:
                rollout_epoch+=1
                if rollout_epoch > 100:
                    break
                start = time.time()
                train(
                    train_loader,
                    epoch,
                    criterion,
                    logger,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    model_utils,
                    rollout_len,
                )
                end = time.time()

                print(
                    f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )

                epoch += 1

                #indent this out if test
                if epoch % 2 == 0:
                    start = time.time()
                    prev_loss, decrement_counter = validate(
                        val_loader,
                        epoch,
                        criterion,
                        logger,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        model_utils,
                        prev_loss,
                        decrement_counter,
                        rollout_len,
                    )
                    end = time.time()
                    print('decrement_counter',decrement_counter)
                    print(
                        f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                    )

                    # If val loss increased 3 times consecutively, go to next rollout length
                    if decrement_counter > 2 or args.test:
                        break
        else:
            start = time.time()
            prev_loss, decrement_counter = validate(
                val_loader,
                epoch,
                criterion,
                logger,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                model_utils,
                prev_loss,
                decrement_counter,
                rollout_len,
            )
            end = time.time()
            print('decrement_counter',decrement_counter)
            print(
                f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
            )

            # If val loss increased 3 times consecutively, go to next rollout length
            if decrement_counter > 2 or args.test:
                break

        '''
        start_time = time.time()

        temp_save_dir = tempfile.mkdtemp()

        test_size = data_dict["test_input"].shape[0]
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict, args)

        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=-2, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir)
            for i in range(0, test_size, args.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {args.traj_save_path}")
        '''

if __name__ == "__main__":
    main()




'''
def infer_absolute(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    args = parse_arguments()
    forecasted_trajectories = {}

    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(
            (batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def infer_helper(
        curr_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    args = parse_arguments()
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    print(f"#### LSTM+social inference at {start_idx} ####"
          ) if args.use_social else print(
        f"#### LSTM inference at {start_idx} ####")
    infer_absolute(
        curr_test_loader,
        encoder,
        decoder,
        start_idx,
        forecasted_save_dir,
        model_utils,
    )
'''
