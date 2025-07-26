from atria_models import MODEL


@MODEL.register(name="cifar10_models/resnet18")
def resnet18():
    from atria_models.models.cifar10_models.resnet import BasicBlock, ResNet

    return ResNet(BasicBlock, [2, 2, 2, 2])


@MODEL.register(name="cifar10_models/resnet34")
def resnet34():
    from atria_models.models.cifar10_models.resnet import BasicBlock, ResNet

    return ResNet(BasicBlock, [3, 4, 6, 3])


@MODEL.register(name="cifar10_models/resnet50")
def resnet50():
    from atria_models.models.cifar10_models.resnet import Bottleneck, ResNet

    return ResNet(Bottleneck, [3, 4, 6, 3])


@MODEL.register(name="cifar10_models/resnet101")
def resnet101():
    from atria_models.models.cifar10_models.resnet import Bottleneck, ResNet

    return ResNet(Bottleneck, [3, 4, 23, 3])


@MODEL.register(name="cifar10_models/resnet152")
def resnet152():
    from atria_models.models.cifar10_models.resnet import Bottleneck, ResNet

    return ResNet(Bottleneck, [3, 8, 36, 3])
