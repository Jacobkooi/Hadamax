import flax.linen as nn
import jax.numpy as jnp
import jax
from jax import lax
from flax.linen.pooling import max_pool, avg_pool
from typing import Sequence


class NatureCNN(nn.Module):
    config: dict
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):


        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.config["ENCODER"] == "baseline":

            # First convolutional block
            # First block
            x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID",
                        kernel_init=nn.initializers.he_normal())(x)
            x = normalize(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID",
                        kernel_init=nn.initializers.he_normal())(x)
            x = normalize(x)
            x = nn.relu(x)
            x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID",
                        kernel_init=nn.initializers.he_normal())(x)
            x = normalize(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))

        elif self.config["ENCODER"] == "hadamax":

            ################## First block
            x1 = nn.Conv(32 , kernel_size=(8, 8), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x2 = nn.Conv(32 , kernel_size=(8, 8), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x1 = normalize(x1)  # Normalize before activation
            x2 = normalize(x2)  # Normalize before activatio
            x1 = nn.gelu(x1)  # Apply activation (tanh)
            x2 = nn.gelu(x2)  # Apply activation (tanh)
            x = x1 * x2  # Element-wise multiplication
            x = max_pool(x, window_shape=(4, 4), strides=(4, 4), padding="SAME")

            ################# Second block
            x1 = nn.Conv(64 , kernel_size=(4, 4), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x2 = nn.Conv(64 , kernel_size=(4, 4), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x1 = normalize(x1)  # Normalize before activation
            x2 = normalize(x2)  # Normalize before activation
            x1 = nn.gelu(x1)  # Apply activation (tanh)
            x2 = nn.gelu(x2)  # Apply activation (tanh)
            x = x1 * x2  # Element-wise multiplication
            x = max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

            ############### Third block
            x1 = nn.Conv(64 , kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x2 = nn.Conv(64 , kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                         kernel_init=nn.initializers.xavier_normal())(x)
            x1 = normalize(x1)  # Normalize before activation
            x2 = normalize(x2)  # Normalize before activation
            x1 = nn.gelu(x1)  # Apply activation (tanh)
            x2 = nn.gelu(x2)  # Apply activation (tanh)
            x = x1 * x2  # Element-wise multiplication
            x = max_pool(x, window_shape=(3, 3), strides=(1, 1), padding="SAME")

            x = x.reshape((x.shape[0], -1))  # Flatten for downstream processing

        else:
            raise ValueError("Invalid encoder architecture specified.")


        if self.config["ENCODER"] == "hadamax":
            x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
            x = normalize(x)
            x = nn.gelu(x)

        elif self.config["ENCODER"] == "baseline":
            x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
            x = normalize(x)
            x = nn.relu(x)

        else:
            raise ValueError("Invalid encoder architecture specified.")

        return x


class QNetwork(nn.Module):
    config: dict
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = NatureCNN(norm_type=self.norm_type, config=self.config)(x, train)
        x = nn.Dense(self.action_dim, name="action_dense")(x)
        return x


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, train):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x, train):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x, train)
        x = ResidualBlock(self.channels)(x, train)
        return x


class QNetwork_Impala(nn.Module):
    config: dict
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    channelss: Sequence[int] = (16, 32, 32, 64, 64)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        for channels in self.channelss:
            x = ConvSequence(channels)(x, train)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, name="Action_dense")(x)
        return x

