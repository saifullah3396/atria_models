from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

from atria_models.core.diffusers_model import *  # noqa
from atria_models.core.local_model import *  # noqa
from atria_models.core.mmdet_model import *  # noqa
from atria_models.core.timm_model import *  # noqa
from atria_models.core.torchvision_model import *  # noqa
from atria_models.core.transformers_model import *  # noqa
from atria_models.models.layoutlmv3.modeling_layoutlmv3 import *  # noqa
from atria_models.pipelines.classification.image import *  # noqa
from atria_models.pipelines.classification.layout_token import *  # noqa
from atria_models.pipelines.classification.sequence import *  # noqa
from atria_models.pipelines.classification.token import *  # noqa
from atria_models.pipelines.mmdet.object_detection import *  # noqa
from atria_models.pipelines.qa.qa import *  # noqa

if __name__ == "__main__":
    import shutil

    config_path = Path(__file__).parent / "conf"
    if config_path.exists():
        shutil.rmtree(config_path)
    write_registry_to_yaml(
        str(Path(__file__).parent / "conf"), types=["model", "model_pipeline"]
    )
