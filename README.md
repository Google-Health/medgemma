# MedGemma

MedGemma is a collection of [Gemma 3](https://ai.google.dev/gemma/docs/core)
variants that are trained for performance on medical text and image
comprehension. Developers can use MedGemma to accelerate building
healthcare-based AI applications. MedGemma comes in two variants: a 4B
multimodal version and a 27B text-only version.

MedGemma 4B utilizes a [SigLIP image encoder](https://arxiv.org/abs/2303.15343)
that has been specifically pre-trained on a variety of de-identified medical
data, including chest X-rays, dermatology images, ophthalmology images, and
histopathology slides. Its LLM component is trained on a diverse set of medical
data, including radiology images, histopathology patches, ophthalmology images,
dermatology images, and medical text.

MedGemma variants have been evaluated on a range of clinically relevant
benchmarks to illustrate their baseline performance. These include both open
benchmark datasets and curated datasets, with a focus on expert human
evaluations for tasks. Developers can fine tune MedGemma variants for improved
performance. Please read more about our work in our manuscript [link coming] and
consult our Intended Use Statement for more details.

## Get started

*   Read our
    [developer documentation](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
    to see the full range of next steps available, including learning more about
    the model through its
    [model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card).

*   Explore this repository, which contains [notebooks](./notebooks) for using
    the model.

*   Visit the model on
    [Hugging Face](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4) or
    [Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma).

## Contributing

We are open to bug reports, pull requests (PR), and other contributions. See
[CONTRIBUTING](CONTRIBUTING.md) and
[community guidelines](https://developers.google.com/health-ai-developer-foundations/community-guidelines)
for details.

## License

While the model is licensed under the
[Health AI Developer Foundations License](https://developers.google.com/health-ai-developer-foundations/terms),
everything in this repository is licensed under the Apache 2.0 license, see
[LICENSE](LICENSE).
