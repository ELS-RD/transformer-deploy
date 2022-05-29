#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
This module is copy-pasted in generated Triton configuration folder to perform inference.
"""

import json
import os

# noinspection DuplicatedCode
from typing import Dict, List, Tuple, Union

import numpy as np
import torch


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer
from transformers.data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from transformers.utils import PaddingStrategy


# code adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py


class TritonPythonModel:
    tokenizer: PreTrainedTokenizer
    device: str
    # change defaults parameters here
    padding: str = None
    top_k: int = 1
    doc_stride: int = None
    max_answer_len: int = 15
    max_seq_len: int = None
    max_question_len: int = 64
    handle_impossible_answer: bool = False

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        current_path: str = os.path.join(args["model_repository"], args["model_version"])
        target_model = args["model_name"].replace("_inference", "_model")
        self.device = "cpu" if args["model_instance_kind"] == "CPU" else "cuda"

        def inference_triton(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Union[None, torch.Tensor] = None,
        ) -> np.ndarray:
            input_ids = input_ids.type(dtype=torch.int32)
            attention_mask = attention_mask.type(dtype=torch.int32)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.type(dtype=torch.int32)
            inputs = [
                pb_utils.Tensor.from_dlpack("input_ids", torch.to_dlpack(input_ids)),
                pb_utils.Tensor.from_dlpack("attention_mask", torch.to_dlpack(attention_mask)),
            ]
            if token_type_ids is not None:
                inputs.append(pb_utils.Tensor.from_dlpack("token_type_ids", torch.to_dlpack(token_type_ids)))
            inference_request = pb_utils.InferenceRequest(
                model_name=target_model,
                requested_output_names=["start_logits", "end_logits"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(inference_response.error().message())
            else:
                start_logits = pb_utils.get_output_tensor_by_name(inference_response, "start_logits")
                end_logits = pb_utils.get_output_tensor_by_name(inference_response, "end_logits")
                start_logits_tensor: torch.Tensor = torch.from_dlpack(start_logits.to_dlpack())
                end_logits_tensor: torch.Tensor = torch.from_dlpack(end_logits.to_dlpack())
                return start_logits_tensor.cpu().numpy(), end_logits_tensor.cpu().numpy()

        self.model = inference_triton
        self.tokenizer = AutoTokenizer.from_pretrained(current_path)
        self.config = AutoConfig.from_pretrained(current_path)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            question = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "QUESTION").as_numpy().tolist()
            ]
            context = [
                t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "CONTEXT").as_numpy().tolist()
            ]

            example = SquadExample(None, question[0], context[0], None, None, None)

            if self.max_seq_len is None:
                self.max_seq_len = min(self.tokenizer.model_max_length, 384)
            if self.doc_stride is None:
                self.doc_stride = min(self.max_seq_len // 2, 128)

            if not self.tokenizer.is_fast:
                features = squad_convert_examples_to_features(
                    examples=[example],
                    tokenizer=self.tokenizer,
                    max_seq_length=self.max_seq_len,
                    doc_stride=self.doc_stride,
                    max_query_length=self.max_question_len,
                    padding_strategy=PaddingStrategy.MAX_LENGTH,
                    is_training=False,
                    tqdm_enabled=False,
                )
            else:
                # Define the side we want to truncate / pad and the text/pair sorting
                question_first = self.tokenizer.padding_side == "right"

                encoded_inputs = self.tokenizer(
                    text=example.question_text if question_first else example.context_text,
                    text_pair=example.context_text if question_first else example.question_text,
                    padding="do_not_pad",
                    truncation="only_second" if question_first else "only_first",
                    max_length=self.max_seq_len,
                    stride=self.doc_stride,
                    return_tensors="np",
                    return_token_type_ids=True,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                )
                # When the input is too long, it's converted in a batch of inputs with overflowing tokens
                # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
                # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original
                # batch sample.
                # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
                # "num_span" is the number of output samples generated from the overflowing tokens.
                num_spans = len(encoded_inputs["input_ids"])

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
                p_mask = np.asarray(
                    [
                        [tok != 1 if question_first else 0 for tok in encoded_inputs.sequence_ids(span_id)]
                        for span_id in range(num_spans)
                    ]
                )

                features = []
                for span_idx in range(num_spans):
                    input_ids_span_idx = encoded_inputs["input_ids"][span_idx]
                    attention_mask_span_idx = (
                        encoded_inputs["attention_mask"][span_idx] if "attention_mask" in encoded_inputs else None
                    )
                    token_type_ids_span_idx = (
                        encoded_inputs["token_type_ids"][span_idx] if "token_type_ids" in encoded_inputs else None
                    )
                    # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
                    if self.tokenizer.cls_token_id is not None:
                        cls_indices = np.nonzero(np.array(input_ids_span_idx) == self.tokenizer.cls_token_id)[0]
                        for cls_index in cls_indices:
                            p_mask[span_idx][cls_index] = 0
                    submask = p_mask[span_idx]
                    if isinstance(submask, np.ndarray):
                        submask = submask.tolist()
                    features.append(
                        SquadFeatures(
                            input_ids=input_ids_span_idx,
                            attention_mask=attention_mask_span_idx,
                            token_type_ids=token_type_ids_span_idx,
                            p_mask=submask,
                            encoding=encoded_inputs[span_idx],
                            # We don't use the rest of the values - and actually
                            # for Fast tokenizer we could totally avoid using SquadFeatures and SquadExample
                            cls_index=None,
                            token_to_orig_map={},
                            example_index=0,
                            unique_id=0,
                            paragraph_len=0,
                            token_is_max_context=0,
                            tokens=[],
                            start_position=0,
                            end_position=0,
                            is_impossible=False,
                            qas_id=None,
                        )
                    )

            model_output_features = []
            for i, feature in enumerate(features):
                fw_args = {}
                others = {}
                model_input_names = self.tokenizer.model_input_names + ["p_mask"]

                for k, v in feature.__dict__.items():
                    if k in model_input_names:
                        tensor = torch.tensor(v)
                        if tensor.dtype == torch.int32:
                            tensor = tensor.long()
                        fw_args[k] = tensor.unsqueeze(0)
                    else:
                        others[k] = v

                model_inputs = {k: fw_args[k].to("cuda") for k in self.tokenizer.model_input_names}

                output_seq: Tuple[np.ndarray, np.ndarray] = self.model(**model_inputs)

                model_output_features.append(
                    {"start": output_seq[0], "end": output_seq[1], "example": example, **fw_args, **others}
                )

            output = self.postprocess(
                model_output_features, self.top_k, self.handle_impossible_answer, self.max_answer_len
            )

            tensor_output = [pb_utils.Tensor("output", np.array(json.dumps(output), dtype=np.object))]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses

    def postprocess(
        self,
        model_outputs,
        top_k=1,
        handle_impossible_answer=False,
        max_answer_len=15,
    ):
        min_null_score = 1000000  # large and positive
        answers = []
        for output in model_outputs:
            start_ = output["start"]
            end_ = output["end"]
            example = output["example"]

            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.abs(np.array(output["p_mask"]) - 1)

            if output.get("attention_mask", None) is not None:
                undesired_tokens = undesired_tokens & output["attention_mask"].numpy()

            # Generate mask
            undesired_tokens_mask = undesired_tokens == 0.0

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_ = np.where(undesired_tokens_mask, -10000.0, start_)
            end_ = np.where(undesired_tokens_mask, -10000.0, end_)

            # Normalize logits and spans to retrieve the answer
            start_ = np.exp(start_ - start_.max(axis=-1, keepdims=True))
            start_ = start_ / start_.sum()

            end_ = np.exp(end_ - end_.max(axis=-1, keepdims=True))
            end_ = end_ / end_.sum()

            if handle_impossible_answer:
                min_null_score = min(min_null_score, (start_[0, 0] * end_[0, 0]).item())

            # Mask CLS
            start_[0, 0] = end_[0, 0] = 0.0

            starts, ends, scores = self.decode(start_, end_, top_k, max_answer_len, undesired_tokens)
            if not self.tokenizer.is_fast:
                char_to_word = np.array(example.char_to_word_offset)

                # Convert the answer (tokens) back to the original text
                # Score: score from the model
                # Start: Index of the first character of the answer in the context string
                # End: Index of the character following the last character of the answer in the context string
                # Answer: Plain text of the answer
                for s, e, score in zip(starts, ends, scores):
                    token_to_orig_map = output["token_to_orig_map"]
                    answers.append(
                        {
                            "score": score.item(),
                            "start": np.where(char_to_word == token_to_orig_map[s])[0][0].item(),
                            "end": np.where(char_to_word == token_to_orig_map[e])[0][-1].item(),
                            "answer": " ".join(
                                example.doc_tokens[token_to_orig_map[s] : token_to_orig_map[e] + 1]  # noqa: E203
                            ),
                        }
                    )
            else:
                # Convert the answer (tokens) back to the original text
                # Score: score from the model
                # Start: Index of the first character of the answer in the context string
                # End: Index of the character following the last character of the answer in the context string
                # Answer: Plain text of the answer
                question_first = bool(self.tokenizer.padding_side == "right")
                enc = output["encoding"]

                # Encoding was *not* padded, input_ids *might*.
                # It doesn't make a difference unless we're padding on
                # the left hand side, since now we have different offsets
                # everywhere.
                if self.tokenizer.padding_side == "left":
                    offset = (output["input_ids"] == self.tokenizer.pad_token_id).numpy().sum()
                else:
                    offset = 0

                # Sometimes the max probability token is in the middle of a word so:
                # - we start by finding the right word containing the token with `token_to_word`
                # - then we convert this word in a character span with `word_to_chars`
                sequence_index = 1 if question_first else 0
                for s, e, score in zip(starts, ends, scores):
                    s = s - offset
                    e = e - offset
                    try:
                        start_word = enc.token_to_word(s)
                        end_word = enc.token_to_word(e)
                        start_index = enc.word_to_chars(start_word, sequence_index=sequence_index)[0]
                        end_index = enc.word_to_chars(end_word, sequence_index=sequence_index)[1]
                    except Exception:
                        # Some tokenizers don't really handle words. Keep to offsets then.
                        start_index = enc.offsets[s][0]
                        end_index = enc.offsets[e][1]

                    answers.append(
                        {
                            "score": score.item(),
                            "start": start_index,
                            "end": end_index,
                            "answer": example.context_text[start_index:end_index],
                        }
                    )

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:top_k]
        if len(answers) == 1:
            return answers[0]
        return answers

    def decode(
        self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int, undesired_tokens: np.ndarray
    ) -> Tuple:
        """
        Take the output of any `ModelForQuestionAnswering` and will generate probabilities for each span to be the
        actual answer.
        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.
        Args:
            start (`np.ndarray`): Individual start probabilities for each token.
            end (`np.ndarray`): Individual end probabilities for each token.
            topk (`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (`int`): Maximum size of the answer to extract from the model's output.
            undesired_tokens (`np.ndarray`): Mask determining tokens that can be part of the answer
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        starts, ends = np.unravel_index(idx_sort, candidates.shape)[1:]
        desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
        starts = starts[desired_spans]
        ends = ends[desired_spans]
        scores = candidates[0, starts, ends]

        return starts, ends, scores

    def span_to_answer(self, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.
        Args:
            text (`str`): The actual context to extract the answer from.
            start (`int`): The answer starting token index.
            end (`int`): The answer end token index.
        Returns:
            Dictionary like `{'answer': str, 'start': int, 'end': int}`
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                if token_idx == end:
                    char_end_idx = chars_idx + len(word)

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }
