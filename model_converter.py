import io
import os

import torch


def convert_to_bytes(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    buffer = io.BytesIO()
    torch.save(model, buffer)

    return buffer


def save_bytes_to_file(file_path, buffer):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        buffer.seek(0)
        file.write(buffer.read())


def load_bytes_from_file(file_path):
    with open(file_path, 'rb') as file:
        model_bytes = file.read()

    return io.BytesIO(model_bytes)


def load_model_from_bytes(buffer):
    return torch.load(buffer, map_location=torch.device("cpu"))


# def load_ckpt_from_bytes(self, checkpoint_bytes):
#     buffer = io.BytesIO(checkpoint_bytes)
#     checkpoint_dict = torch.load(buffer, map_location=torch.device(self.device))
#     a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
#     print("Loaded checkpoint from bytes")
#     print('missing/unexpected keys:', a, b)

# def convert_model_to_bytes(model_path, save_path):
#     model = torch.load(model_path, map_location=torch.device("cpu"))
#
#     buffer = io.BytesIO()
#     torch.save(model, buffer)
#
#     with open(save_path, 'wb') as file:
#         buffer.seek(0)  # Rewind the buffer to the beginning
#         file.write(buffer.read())
#
#
# def load_ckpt_from_bytes(self, checkpoint_bytes):
#     buffer = io.BytesIO(checkpoint_bytes)
#     checkpoint_dict = torch.load(buffer, map_location=torch.device(self.device))
#     a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
#     print("Loaded checkpoint from bytes")
#     print('missing/unexpected keys:', a, b)
#
#
# def save_ckpt_as_bytes(self, checkpoint_dict):
#     buffer = io.BytesIO()
#     torch.save(checkpoint_dict, buffer)
#     return buffer.getvalue()
