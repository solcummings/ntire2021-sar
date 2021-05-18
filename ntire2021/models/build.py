from models.mobilenet import MobileNet


def build(model_name: str, model_args: dict, **kwargs):
    model_dict = {
            'mobilenet': MobileNet,
    }
    return model_dict[model_name.lower()](**(model_args or {}))


