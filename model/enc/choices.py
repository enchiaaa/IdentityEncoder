from enum import Enum
from torch import nn


class TrainMode(Enum):
    # 定義訓練模式的枚舉類型，用於控制訓練過程中的行為

    # manipulate 模式：訓練分類器的模式，通常用於目標操作的任務
    manipulate = 'manipulate'

    # diffusion 模式：默認的擴散模型訓練模式，用於生成數據
    diffusion = 'diffusion'

    # latent_diffusion 模式：訓練潛在空間的擴散模型（如將 DDPM 適配到潛在空間）
    latent_diffusion = 'latentdiffusion'

    def is_manipulate(self):
        """
        判斷是否為 manipulate 模式
        Returns:
            bool: 如果模式為 manipulate，返回 True，否則返回 False
        """
        return self in [
            TrainMode.manipulate,
        ]

    def is_diffusion(self):
        """
        判斷是否為擴散模型相關模式（包括 diffusion 和 latent_diffusion）
        Returns:
            bool: 如果是擴散模式，返回 True，否則返回 False
        """
        return self in [
            TrainMode.diffusion,
            TrainMode.latent_diffusion,
        ]

    def is_autoenc(self):
        """
        判斷當前模式是否可能涉及自動編碼（autoencoding）
        Returns:
            bool: 如果可能涉及自動編碼（例如 diffusion 模式），返回 True，否則返回 False
        """
        return self in [
            TrainMode.diffusion,
        ]

    def is_latent_diffusion(self):
        """
        判斷是否為 latent_diffusion 模式
        Returns:
            bool: 如果模式為 latent_diffusion，返回 True，否則返回 False
        """
        return self in [
            TrainMode.latent_diffusion,
        ]

    def use_latent_net(self):
        """
        判斷是否需要使用潛在網路（latent net）
        Returns:
            bool: 如果模式為 latent_diffusion，返回 True，否則返回 False
        """
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        """
        判斷當前模式是否需要推理數據集中的潛在變量（latent variables）
        Returns:
            bool: 如果模式需要預先計算潛在變量並生成潛在數據集，返回 True，否則返回 False
        """
        # 在這些模式下會提前計算潛在變量
        # 數據集將包含所有預測的潛在變量
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ManipulateMode(Enum):
    """
    定義操作模式（ManipulateMode）枚舉，用於控制如何訓練分類器以進行目標操作。
    """

    # celebahq_all：在完整的 CelebA 屬性數據集上進行訓練
    celebahq_all = 'celebahq_all'

    # d2c_fewshot：使用 D2C 方法裁剪的 CelebA 小樣本數據進行訓練
    d2c_fewshot = 'd2cfewshot'

    # d2c_fewshot_allneg：使用 D2C 裁剪的 CelebA 小樣本數據，並僅包含負樣本
    d2c_fewshot_allneg = 'd2cfewshotallneg'

    def is_celeba_attr(self):
        """
        判斷是否基於 CelebA 屬性數據進行操作模式
        Returns:
            bool: 如果模式是基於 CelebA 的操作，返回 True，否則返回 False。
        """
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
        ]

    def is_single_class(self):
        """
        判斷是否為單一分類器模式（通常用於小樣本操作）
        Returns:
            bool: 如果是單一分類器模式，返回 True，否則返回 False。
        """
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot(self):
        """
        判斷是否為小樣本(few-shot)訓練模式
        Returns:
            bool: 如果是小樣本訓練模式，返回 True，否則返回 False。
        """
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot_allneg(self):
        """
        判斷是否為僅包含負樣本的小樣本模式
        Returns:
            bool: 如果是僅負樣本的小樣本模式，返回 True，否則返回 False。
        """
        return self in [
            ManipulateMode.d2c_fewshot_allneg,
        ]


class ModelType(Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
        ]

    def can_sample(self):
        return self in [ModelType.ddpm]


class ModelName(Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'


class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    """

    eps = 'eps'  # the model predicts epsilon


class ModelVarType(Enum):
    """
    定義了模型預測的方差類型
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = 'fixed_small' # 小的固定方差：在擴散過程中，這代表模型使用一個小的固定方差來進行噪聲的處理。
    # beta_t
    fixed_large = 'fixed_large' # 大的固定方差：這是模型使用一個較大的固定方差進行處理，通常用於更高階的擴散過程。


class LossType(Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'


class GenerativeType(Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(Enum):
    adam = 'adam'
    adamw = 'adamw'


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class ManipulateLossType(Enum):
    bce = 'bce'
    mse = 'mse'