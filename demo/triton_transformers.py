import argparse

import numpy as np
import tritonclient.http

from src.transformer_deploy.benchmarks.utils import print_timings, setup_logging, track_infer_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="require inference", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--length", required=True, help="sequence length", choices=(16, 128), type=int)
    parser.add_argument("--model", required=True, help="sequence length", choices=("onnx", "tensorrt"))
    args, _ = parser.parse_known_args()

    setup_logging()
    model_name = f"sts_{args.model}_inference"
    url = "127.0.0.1:8000"
    model_version = "1"
    batch_size = 1

    if args.length == 128:
        # from https://venturebeat.com/2021/08/25/how-hugging-face-is-tackling-bias-in-nlp/, text used in the HF demo
        text = """Today, Hugging Face has expanded to become a robust NLP startup, 
        known primarily for making open-source software such as Transformers and Datasets, 
        used for building NLP systems. “The software Hugging Face develops can be used for 
        classification, question answering, translation, and many other NLP tasks,” Rush said. 
        Hugging Face also hosts a range of pretrained NLP models, on GitHub, that practitioners can download 
        and apply for their problems, Rush added."""  # noqa: W291
    else:
        text = "This live event is great. I will sign-up for Infinity."

    triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
    assert triton_client.is_model_ready(
        model_name=model_name, model_version=model_version
    ), f"model {model_name} not yet ready"

    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

    query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
    model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)
    time_buffer = list()
    for _ in range(10000):
        query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
        _ = triton_client.infer(
            model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
        )
    for _ in range(1000):
        with track_infer_time(time_buffer):
            query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
            response = triton_client.infer(
                model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
            )

    print_timings(name="triton transformers", timings=time_buffer)
    print(response.as_numpy("output"))
