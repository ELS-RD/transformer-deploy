from transformers import AutoTokenizer, TensorType
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, _):
        # hard coded path to simplify the code, IRL you serialize locally your tokenizer for stability
        self.tokenizer = AutoTokenizer.from_pretrained("philschmid/MiniLM-L6-H384-uncased-sst2")

    def execute(self, requests):
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [t.decode('UTF-8') for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens = self.tokenizer(text=query, return_token_type_ids=False, return_tensors=TensorType.NUMPY, max_length=128, truncation=True,)
            # communicate the tokenization results to Triton server
            input_ids = pb_utils.Tensor("INPUT_IDS", tokens['input_ids'])
            attention = pb_utils.Tensor("ATTENTION", tokens['attention_mask'])
            inference_response = pb_utils.InferenceResponse(output_tensors=[input_ids, attention])
            responses.append(inference_response)

        return responses
