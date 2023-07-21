# RetNet

A minimal implementation of Retention Networks. [https://arxiv.org/pdf/2307.08621.pdf](https://arxiv.org/pdf/2307.08621.pdf)
Supports all types of forward implementations: `parallel`, `recurrent`, `chunkwise`

Check `play.ipynb` for minimal testing of parallel, recurrent, and chunkwise forward.

# TODO

## Complex Numbers (xpos)

- The authors mention xpos as $e^{in\theta}$, but it seems that it is not connected
to the xpos paper. Which is weird, since both are from microsoft research.

- RetNet uses complex numbers within (equation 5 of the paper), but complex numbers are
large (64bits) / non-quantizable / not many kernels are implmeneted on them.

- Moreover, when do we get back to real number, since we need to get LM prob in the end,
which must be a real number?

## Huggingface style

- make the implementation based on huggingface `PretrainedModel`.
