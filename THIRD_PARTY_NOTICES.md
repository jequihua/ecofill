# Third-party notices

ecofill's original code is released under the MIT License in [LICENSE](LICENSE).
This file covers code that ecofill ships but did not write, and the separate
terms of optional packages and pretrained model weights that ecofill can use
but never ships. It is a record of the terms, not legal advice.

## Code shipped inside the `ecofill` package

Two upstream code subsets are vendored unchanged. Their license files travel
with the source and are included in every built wheel.

### PyPOTS numerical cores (BSD-3-Clause)

- Location: `src/ecofill/_pypots/` (license file:
  [src/ecofill/_pypots/LICENSE](src/ecofill/_pypots/LICENSE)).
- Origin: a subset of the numerical modules of PyPOTS 1.5, the BRITS and SAITS
  model cores and the layers they need. The file bytes are unchanged; small
  package initializers were added.
- Used by: the optional `brits_pypots` and `saits_pypots` adapters. Running
  them requires the `deep` extra (PyTorch).

License text, reproduced exactly:

```text
Copyright (c) 2023-present, Wenjie Du
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### hesseflux gap-filling module (MIT)

- Location: `src/ecofill/_hesseflux_gapfill.py` (license file:
  [src/ecofill/_hesseflux_LICENSE.txt](src/ecofill/_hesseflux_LICENSE.txt)).
- Origin: `gapfill.py` from Matthias Cuntz's hesseflux, revision
  `53d1f9a75ba388d400e0cff65530f0da33720d8f`, copied unchanged.
- Used by: the `mds_hesseflux` method (`ecofill.mds`), which needs no extra.

License text, reproduced exactly:

```text
The hesseflux Python package is distributed under the MIT License.

Copyright (c) 2009-2022 Matthias Cuntz - mc (at) macu (dot) de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Optional packages (not shipped)

The optional extras only declare pinned dependencies. Installing an extra
downloads those packages from your package index under their own licenses. The
core install (`numpy`, `pandas`) is unaffected.

| Extra | Pinned packages | Used for |
|---|---|---|
| `research` | scipy 1.17.0, scikit-learn 1.8.0, statsmodels 0.14.6, h5py 3.16.0 | random-forest and boosting driver regression |
| `deep` | torch 2.14.0 | BRITS / SAITS adapters |
| `foundation` | timesfm 3.0.2, torch 2.14.0, huggingface-hub 1.33.0, safetensors 0.8.0 | TimesFM 2.5 and 3 forecast adapters |
| `tabpfn` | tabpfn 9.0.0, torch 2.14.0 | TabPFN 3.5 time-indexed backend |

Check each package's own license before redistributing an environment.

## Pretrained model weights (never shipped, never downloaded)

ecofill contains no model weights and never downloads any. The adapters accept
only a local file or directory that you obtained yourself, and they check its
exact SHA-256 before loading it. The weight terms below are separate from the
code licenses: installing an extra, the MIT license of ecofill, or an academic
purpose does not change them. Read the linked license text before you download
or use a checkpoint.

### TimesFM 2.5 (`TimesFM25Backend`)

- Checkpoint: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3),
  revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, file `model.safetensors`,
  SHA-256 `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`.
- Weight license: Apache-2.0, as declared by the model card at that revision
  and by the [TimesFM repository](https://github.com/google-research/timesfm)
  ("model weights up to version 2.5 remain Apache-2.0").
- Code: the `timesfm` package source is Apache-2.0.

### TimesFM 3.0 (`TimesFM3Backend`)

- Checkpoint: [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2),
  revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, file `model.safetensors`,
  SHA-256 `a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`.
- Weight license: TimesFM Non-Commercial License v1.0
  (`timesfm-non-commercial-license-v1.0`), the
  [LICENSE file at that revision](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE),
  SHA-256 `3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd`.
  It permits use only for non-commercial purposes (testing, evaluation or
  research not tied to commercial gain, production deployment or revenue) and
  does not permit distributing the model. The adapter checks that exact license
  file next to the weights and requires `purpose="non-commercial research"`.
  That argument is a reminder, not a license grant.
- Code: the `timesfm` package source is Apache-2.0, separate from these weights.

### TabPFN 3.5 (`TabPFNBackend`)

- Checkpoint: [Prior-Labs/tabpfn_3_5](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e),
  revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`, file
  `tabpfn-v3.5-20260909.safetensors` (876,027,932 bytes), SHA-256
  `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
- Weight license: TABPFN-3.5 License v1.0 (`tabpfn-3-5-license-v1.0`), the
  [LICENSE file at that revision](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e/LICENSE).
  It grants use only for non-commercial purposes; commercial or production use
  needs a separate license from Prior Labs, and redistribution carries
  attribution and license-copy conditions. ecofill checks only the checkpoint
  bytes, not your purpose.
- Code: the `tabpfn` 9.0.0 package source is Apache-2.0
  (tag v9.0.0, commit `c70b6ef0`; [repository](https://github.com/PriorLabs/TabPFN/tree/c70b6ef0488858d32244c52222abfc0c5be207d6)).

## Data

No measurements, private inputs or model outputs from real sites are included.
All examples and tests generate small synthetic series.
