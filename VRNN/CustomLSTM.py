################################################################################
# Dealing with long sequence with LSTM to avoid out of memory errors
##based on https://github.com/sdatkinson/neural-amp-modeler/blob/50c247a39d20ba5325ebdecc7541b06871c6105e/nam/models/recurrent.py#L113
################################################################################
import abc
import json
import wavio
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple, Union
def np_to_wav(
    x: np.ndarray,
    filename: Union[str, Path],
    rate: int = 48_000,
    sampwidth: int = 3,
    scale="none",
):
    wavio.write(
        str(filename),
        (np.clip(x, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(np.int32),
        rate,
        scale=scale,
        sampwidth=sampwidth,
    )


class InitializableFromConfig(object):
    @classmethod
    def init_from_config(cls, config):
        return cls(**cls.parse_config(config))

    @classmethod
    def parse_config(cls, config):
        return config

class Exportable(abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    def export(self, outdir: Path):
        """
        Interface for exporting.
        You should create at least a `config.json` containing the two fields:
        * "version" (str)
        * "architecture" (str)
        * "config": (dict w/ other necessary data like tensor shapes etc)
        :param outdir: Assumed to exist. Can be edited inside at will.
        """
        training = self.training
        self.eval()
        with open(Path(outdir, "config.json"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": self.__class__.__name__,
                    "config": self._export_config(),
                },
                fp,
                indent=4,
            )
        np.save(Path(outdir, "weights.npy"), self._export_weights())
        x, y = self._export_input_output()
        np.save(Path(outdir, "inputs.npy"), x)
        np.save(Path(outdir, "outputs.npy"), y)
        np_to_wav(x, Path(outdir, "input.wav"))
        np_to_wav(y, Path(outdir, "output.wav"))

        # And resume training state
        self.train(training)

    @abc.abstractmethod
    def export_cpp_header(self, filename: Path):
        """
        Export a .h file to compile into the plugin with the weights written right out
        as text
        """
        pass

    @abc.abstractmethod
    def _export_config(self):
        """
        Creates the JSON of the model's archtecture hyperparameters (number of layers,
        number of units, etc)
        :return: a JSON serializable object
        """
        pass

    @abc.abstractmethod
    def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an input and corresponding output signal to verify its behavior.
        They should be the same length, but the start of the output might have transient
        effects. Up to you to interpret.
        """
        pass

    @abc.abstractmethod
    def _export_weights(self) -> np.ndarray:
        """
        Flatten the weights out to a 1D array
        """
        pass

class _Base(nn.Module, InitializableFromConfig, Exportable):
    @abc.abstractproperty
    def pad_start_default(self) -> bool:
        pass

    @abc.abstractproperty
    def receptive_field(self) -> int:
        """
        Receptive field of the model
        """
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _forward(self, *args) -> torch.Tensor:
        """
        The true forward method.
        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

    def _export_input_output_args(self) -> Tuple[Any]:
        """
        Create any other args necessesary (e.g. params to eval at)
        """
        return ()

    def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
        args = self._export_input_output_args()
        rate = REQUIRED_RATE
        x = torch.cat(
            [
                torch.zeros((rate,)),
                0.5
                * torch.sin(
                    2.0 * math.pi * 220.0 * torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                torch.zeros((rate,)),
            ]
        )
        # Use pad start to ensure same length as requested by ._export_input_output()
        return (
            x.detach().cpu().numpy(),
            self(*args, x, pad_start=True).detach().cpu().numpy()
        )


class BaseNet(_Base):
    def forward(self, x: torch.Tensor, pad_start: Optional[bool] = None):
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None]
        if pad_start:
            x = torch.cat((torch.zeros((len(x), self.receptive_field - 1)), x), dim=1)
        y = self._forward(x)
        if scalar:
            y = y[0]
        return y

    @abc.abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The true forward method.
        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

class _L(nn.LSTM):
    """
    Tweaks to LSTM
    * Up the remembering
    """
    def reset_parameters(self) -> None:
        super().reset_parameters()
        # https://danijar.com/tips-for-training-recurrent-neural-networks/
        # forget += 1
        # ifgo
        value = 2.0
        idx_input = slice(0, self.hidden_size)
        idx_forget = slice(self.hidden_size, 2 * self.hidden_size)
        for layer in range(self.num_layers):
            for input in ("i", "h"):
                # Balance out the scale of the cell w/ a -=1
                getattr(self, f"bias_{input}h_l{layer}").data[
                    idx_input
                ] -= value
                getattr(self, f"bias_{input}h_l{layer}").data[
                    idx_forget
                ] += value



class LSTMCore(_L):
    def __init__(
        self,
        *args,
        train_burn_in: Optional[int] = None,
        train_truncate: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.batch_first:
            raise NotImplementedError("Need batch first")
        self._train_burn_in = train_burn_in
        self._train_truncate = train_truncate
        assert len(args) < 3, "Provide as kwargs"
        self._initial_cell = nn.Parameter(
            torch.zeros((self.num_layers, self.hidden_size))
        )
        self._initial_hidden = nn.Parameter(
            torch.zeros((self.num_layers, self.hidden_size))
        )

    def forward(self, x, hidden_state=None):
        """
        Same as nn.LSTM.forward except:
        * Learned inital state
        * truncated BPTT when .training
        """
        if x.ndim != 3:
            raise NotImplementedError("Need (B,L,D)")
        last_hidden_state = (
            self._initial_state(None if x.ndim == 2 else len(x))
            if hidden_state is None else hidden_state
        )
        if not self.training or self._train_truncate is None:
            output_features, hidden_state = super().forward(x, last_hidden_state)
        else:
            output_features_list = []
            h_state_list    = []
            c_state_list    = []
            if self._train_burn_in is not None:
                last_output_features, last_hidden_state = super().forward(
                    x[:, : self._train_burn_in, :], last_hidden_state
                )
                output_features_list.append(last_output_features.detach())
                h_state_list.append(last_hidden_state[0].detach())
                c_state_list.append(last_hidden_state[1].detach())

            burn_in_offset = 0 if self._train_burn_in is None else self._train_burn_in
            for i in range(burn_in_offset, x.shape[1], self._train_truncate):
                if i > burn_in_offset:
                    # Don't detach the burn-in state so that we can learn it.
                    last_hidden_state = tuple(z.detach() for z in last_hidden_state)
                last_output_features, last_hidden_state = super().forward(
                    x[:, i : i + self._train_truncate, :,],
                    last_hidden_state,
                )
                output_features_list.append(last_output_features)
                h_state_list.append(last_hidden_state[0])
                c_state_list.append(last_hidden_state[1])

            output_features = torch.cat(output_features_list, dim=1)

            h = torch.cat(h_state_list, dim=1)
            c = torch.cat(c_state_list, dim=1)
            hidden_state = (h,c)
        return output_features, hidden_state

    def _initial_state(self, n: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self._initial_hidden, self._initial_cell) if n is None else (
            torch.tile(self._initial_hidden[:, None], (1, n, 1)),
            torch.tile(self._initial_cell[:, None], (1, n, 1))
        )


class LSTM(BaseNet):
    """
    ABC for recurrent architectures
    """

    def __init__(
        self,
        hidden_size,
        train_burn_in: Optional[int] = None,
        train_truncate: Optional[int] = None,
        input_size: int = 1,
        **lstm_kwargs,
    ):
        """
        :param hidden_size: for LSTM
        :param train_burn_in: Detach calculations from first (this many) samples when
            training to burn in the hidden state.
        :param train_truncate: detach the hidden & cell states every this many steps
            during training so that backpropagation through time is faster + to simulate
            better starting states for h(t0)&c(t0) (instead of zeros)
            TODO recognition head to start the hidden state in a good place?
        :param input_size: Usually 1 (mono input). A catnet extending this might change
            it and provide the parametric inputs as additional input dimensions.
        """
        super().__init__()
        if "batch_first" in lstm_kwargs:
            raise ValueError("batch_first cannot be set.")
        self._input_size = input_size
        self._core = _L(
            self._input_size, hidden_size, batch_first=True, **lstm_kwargs
        )
        self._head = nn.Linear(hidden_size, 1)
        self._train_burn_in = train_burn_in
        self._train_truncate = train_truncate
        self._initial_cell = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._initial_hidden = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )

    @property
    def receptive_field(self) -> int:
        return 1

    @property
    def pad_start_default(self) -> bool:
        # I should simplify this...
        return True

    def export_cpp_header(self, filename: Path):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            LSTM.export(self, Path(tmpdir))  # Hacky...need to work w/ CatLSTM
            with open(Path(tmpdir, "config.json"), "r") as fp:
                _c = json.load(fp)
            version = _c["version"]
            config = _c["config"]
            s_parametric = self._export_cpp_header_parametric(config.get("parametric"))
            with open(filename, "w") as f:
                f.writelines(
                    (
                        "#pragma once\n",
                        "// Automatically-generated model file\n",
                        "#include <vector>\n",
                        '#include "json.hpp"\n',
                        '#include "lstm.h"\n',
                        f'#define PYTHON_MODEL_VERSION "{version}"\n',
                        f'const int NUM_LAYERS = {config["num_layers"]};\n',
                        f'const int INPUT_SIZE = {config["input_size"]};\n',
                        f'const int HIDDEN_SIZE = {config["hidden_size"]};\n',
                    )
                    + s_parametric
                    + (
                        "std::vector<float> PARAMS{"
                        + ", ".join(
                            [f"{w:.16f}f" for w in np.load(Path(tmpdir, "weights.npy"))]
                        )
                        + "};\n",
                    )
                )

    def _forward(self, x):
        """
        :param x: (B,L) or (B,L,D)
        :return: (B,L)
        """
        last_hidden_state = self._initial_state(len(x))
        if x.ndim==2:
            x = x[:, :, None]
        if not self.training or self._train_truncate is None:
            output_features = self._core(x, last_hidden_state)[0]
        else:
            output_features_list = []
            if self._train_burn_in is not None:
                last_output_features, last_hidden_state = self._core(
                    x[:, : self._train_burn_in, :], last_hidden_state
                )
                output_features_list.append(last_output_features.detach())
            burn_in_offset = 0 if self._train_burn_in is None else self._train_burn_in
            for i in range(burn_in_offset, x.shape[1], self._train_truncate):
                if i > burn_in_offset:
                    # Don't detach the burn-in state so that we can learn it.
                    last_hidden_state = tuple(z.detach() for z in last_hidden_state)
                last_output_features, last_hidden_state = self._core(
                    x[:, i : i + self._train_truncate, :,],
                    last_hidden_state,
                )
                output_features_list.append(last_output_features)
            output_features = torch.cat(output_features_list, dim=1)
        return self._head(output_features)[:, :, 0]

    def _export_cell_weights(
        self, i: int, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> np.ndarray:
        """
        * weight matrix (xh -> ifco)
        * bias vector
        * Initial hidden state
        * Initial cell state
        """

        tensors = [
            torch.cat(
                [
                    getattr(self._core, f"weight_ih_l{i}").data,
                    getattr(self._core, f"weight_hh_l{i}").data,
                ],
                dim=1,
            ),
            getattr(self._core, f"bias_ih_l{i}").data
            + getattr(self._core, f"bias_hh_l{i}").data,
            hidden_state,
            cell_state,
        ]
        return np.concatenate([z.detach().cpu().numpy().flatten() for z in tensors])

    def _export_config(self):
        return {
            "input_size": self._core.input_size,
            "hidden_size": self._core.hidden_size,
            "num_layers": self._core.num_layers,
        }

    def _export_cpp_header_parametric(self, config):
        # TODO refactor to merge w/ WaveNet implementation
        if config is not None:
            raise ValueError("Got non-None parametric config")
        return ("nlohmann::json PARAMETRIC {};\n",)

    def _export_weights(self):
        """
        * Loop over cells:
            * weight matrix (xh -> ifco)
            * bias vector
            * Initial hidden state
            * Initial cell state
        * Head weights
        * Head bias
        """
        return np.concatenate(
            [
                self._export_cell_weights(i, h, c)
                for i, (h, c) in enumerate(zip(*self._get_initial_state()))
            ]
            + [
                self._head.weight.data.detach().cpu().numpy().flatten(),
                self._head.bias.data.detach().cpu().numpy().flatten(),
            ]
        )

    def _get_initial_state(self, inputs=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience function to find a good hidden state to start the plugin at
        DX=input size
        L=num layers
        S=sequence length
        :param inputs: (1,S,DX)
        :return: (L,DH), (L,DH)
        """
        inputs = torch.zeros((1, 48_000, 1)) if inputs is None else inputs
        _, (h, c) = self._core(inputs)
        return h, c

    def _initial_state(self, n: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Literally what the forward pass starts with.
        Default is zeroes; this should be better since it can be learned.
        """
        return (self._initial_hidden, self._initial_cell) if n is None else (
            torch.tile(self._initial_hidden[:, None], (1, n, 1)),
            torch.tile(self._initial_cell[:, None], (1, n, 1))
        )
