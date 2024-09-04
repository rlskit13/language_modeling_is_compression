# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements a lossless compressor with language models (arithmetic coding)."""

from collections.abc import Iterator
import functools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os

from language_modeling_is_compression import arithmetic_coder
from language_modeling_is_compression import constants
from language_modeling_is_compression import transformer
from language_modeling_is_compression import utils

def _retrieve_model_params_HuggingFace(model_name: str = 'bert-base-uncased') -> dict:
    """Loads and returns the pretrained model, tokenizer, and config from Hugging Face.

    Args:
        model_name (str): The name or path of the pretrained model to load.

    Returns:
        dict: A dictionary containing the model, tokenizer, and config.

    Raises:
        FileNotFoundError: If the specified model is not found locally or online.
    """
    try:
        # Load the model, tokenizer, and config from Hugging Face
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "config": config
        }
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to load the model '{model_name}'. Ensure the model name is correct or that you have an internet connection."
        ) from exc


def _retrieve_model_params() -> dict:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  file_path = 'params.pth'
  if not os.path.exists(file_path):
      raise FileNotFoundError(
          'You must train a model first, the parameters file params.pth does not'
          ' exist yet.'
      )
  return torch.load(file_path)

def _retrieve_predict_fn(
    params: dict,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the prediction function for the trained model."""
    config = transformer.TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
    model = transformer.TransformerDecoder(config)
    model.load_state_dict(params)
    model.eval()

    def predict_fn(x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        with torch.no_grad():
            logits = model(x)
        return logits

    return predict_fn

def _retrieve_predict_fn_HuggingFace(
    params: dict,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the prediction function for the trained model."""
    #config = transformer.TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
    #model = transformer.TransformerDecoder(config)
    #model.load_state_dict(params)
    model = params['model']
    model.eval()

    def predict_fn(x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        with torch.no_grad():
            logits = model(x)
        return logits

    return predict_fn

def compress(
    data: bytes,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a pretrained model.

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `False`) since it has an O(n) runtime complexity, while
      the latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `True`) since this is what
      happens in the decoder (which iteratively reconstructs the sequence).

  Returns:
    The compressed data.
  """
  params = _retrieve_model_params()
  predict_fn = _retrieve_predict_fn(params)

 # Convert the `data` into a tensor of integers (representing the bytes).
  sequence_tensor = torch.tensor(np.frombuffer(data, dtype=np.uint8), dtype=torch.long)

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_tensor)):
      subsequence_probs = predict_fn(
          sequence_tensor[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
    log_probs = torch.vstack(log_probs)
  else:
    log_probs = predict_fn(sequence_tensor[None])[0, ...]
  probs = torch.exp(log_probs)

  output = list()
  encoder = arithmetic_coder.Encoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      output_fn=output.append,
  )
  for pdf, symbol in zip(probs, sequence_tensor):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
  encoder.terminate()

  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits

  return compressed_bytes

def compress_pretrained(
    data: bytes,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = False,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a pretrained model.

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `False`) since it has an O(n) runtime complexity, while
      the latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `True`) since this is what
      happens in the decoder (which iteratively reconstructs the sequence).

  Returns:
    The compressed data.
  """
  params = _retrieve_model_params_HuggingFace()
  predict_fn = _retrieve_predict_fn_HuggingFace(params)

  # Convert the `data` into a tensor of integers (representing the bytes).
  sequence_tensor = torch.tensor(np.frombuffer(data, dtype=np.uint8), dtype=torch.long)

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_tensor)):
      subsequence_probs = predict_fn(
          sequence_tensor[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
    log_probs = torch.vstack(log_probs)
  else:
    log_probs = predict_fn(sequence_tensor[None])[0, ...]
  probs = torch.exp(log_probs)

  output = list()
  encoder = arithmetic_coder.Encoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      output_fn=output.append,
  )
  for pdf, symbol in zip(probs, sequence_tensor):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
  encoder.terminate()

  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  if return_num_padded_bits:
    return compressed_bytes, num_padded_bits

  return compressed_bytes

def decompress(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  """
  params = _retrieve_model_params()
  predict_fn = _retrieve_predict_fn(params)

  data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

  # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
  # from the compressed input and returns `None` when the input is exhausted.
  def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
      return int(next(bit_sequence))
    except StopIteration:
      return None

  decoder = arithmetic_coder.Decoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      input_fn=_input_fn,
  )
  # We need a dummy token because the language model right-shifts the sequence
  # by one when computing the conditional probabilities. Concretely, at every
  # step, we need the `pdf` of the next token given all currently decompressed
  # tokens, but without a dummy token, the last `pdf` would be that of the last
  # already decompressed token. The value of the dummy token is irrelevant.
  sequence_array = torch.empty((1,), dtype=torch.uint8)
  probs = torch.exp(predict_fn(sequence_array.unsqueeze(0))[0, ...])

  for idx in range(uncompressed_length):
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[idx])
    )
    sequence_array =  torch.cat([sequence_array[:-1], torch.tensor([token], dtype=torch.uint8)])
    probs = torch.exp(predict_fn(sequence_array.unsqueeze(0))[0, ...])

  # Remove the dummy token and convert to bytes.
  return sequence_array[:-1].cpu().numpy().tobytes()


def decompress_pretrained(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  """
  params = _retrieve_model_params_HuggingFace()
  predict_fn = _retrieve_predict_fn_HuggingFace(params)

  data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

  # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
  # from the compressed input and returns `None` when the input is exhausted.
  def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
      return int(next(bit_sequence))
    except StopIteration:
      return None

  decoder = arithmetic_coder.Decoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      input_fn=_input_fn,
  )
  # We need a dummy token because the language model right-shifts the sequence
  # by one when computing the conditional probabilities. Concretely, at every
  # step, we need the `pdf` of the next token given all currently decompressed
  # tokens, but without a dummy token, the last `pdf` would be that of the last
  # already decompressed token. The value of the dummy token is irrelevant.
  sequence_array = torch.empty((1,), dtype=torch.uint8)
  probs = torch.exp(predict_fn(sequence_array.unsqueeze(0))[0, ...])

  for idx in range(uncompressed_length):
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[idx].cpu().numpy())
    )
    sequence_array = torch.cat([sequence_array[:-1], torch.tensor([token], dtype=torch.uint8)])
    probs = torch.exp(predict_fn(sequence_array.unsqueeze(0))[0, ...])

  # Remove the dummy token and convert to bytes.
  return sequence_array[:-1].cpu().numpy().tobytes()