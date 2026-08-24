# V1.3 LAN inference console

The V1.3 web console serves the confirmed learned typed-dispatch runtime on a
trusted local network. It exposes the final answer, complete execution trace,
and independent enable/disable controls for the GPT, math, and string towers.

## Start the console

From the repository root, use the Python environment that has PyTorch,
Transformers, and the local GPT-2 cache available:

```powershell
cd C:\CFTN\ctfn_text_build
D:\Applio-3.2.4\env\python.exe -m tools.serve_v1_3_inference --host 0.0.0.0 --port 7860 --device cuda
```

The launcher loads the frozen GPT-2 generalist, math and string specialists,
accepted V1.3 collaboration checkpoint, and learned dispatcher before it opens
the listening socket. Its defaults point at the confirmed artifacts on this
machine:

- `G:\ctfn-text\artifacts\v1_1_algorithmic_linear_equations`
- `G:\ctfn-text\artifacts\v1_2_conditional_bridge`
- `G:\ctfn-text\data\manifests\v1_3_multi_specialist`
- `G:\ctfn-text\config\v1_3_multi_specialist.yaml` (the sealed experiment
  config whose resolved-path hash matches the checkpoints)
- `G:\ctfn-text\artifacts\v1_3_multi_specialist\oracle_hard_answer_bus_recovery\oracle_hard_answer_bus_recovery.best.pth`
- `C:\CFTN\learned_dispatcher_v1_3_final_v2\learned_dispatcher.best.pth`

Every path has a corresponding command-line override. Run this to list them:

```powershell
D:\Applio-3.2.4\env\python.exe -m tools.serve_v1_3_inference --help
```

## Connect from the local network

The launcher prints the loopback and detected IPv4 addresses. On another
device connected to the same private network, open:

```text
http://<computer-ip>:7860/
```

If Windows Defender Firewall asks, allow the Python executable on **Private
networks only**. Do not configure router port forwarding. The service has no
authentication or TLS and must not be exposed directly to the public internet.
Use `--host 127.0.0.1` to make it accessible only on this computer.

## What the trace means

The console executes the architecture that passed V1.3 native confirmation:

1. The learned byte dispatcher predicts one finite intent and confidence.
2. A constrained compiler copies exact operands from the prompt into typed
   specialist calls. It never generates operands or reads oracle metadata.
3. Each enabled specialist receives its exact native request and returns a
   complete `<answer>...</answer>` payload.
4. Sequential results can become immutable inputs to later rounds.
5. A deterministic composer returns or joins the required payloads losslessly.
6. GPT executes for the registered pure-language path and as the fallback when
   the specialist dispatcher does not accept a general prompt.

The trace includes the dispatcher threshold, compiled plan, every round,
dependencies, exact requests, raw tower generations, extracted payloads,
composition state, timings, tower execution states, errors, and checkpoint
hashes. The legacy learned latent wake/halt route is shown as bypassed because
it is not part of the accepted native runtime; the learned typed dispatcher is
the active gate.

## Tower ablations

Every tower switch affects real execution:

- Disabling **GPT generalist** prevents pure-language fallback. It does not
  affect specialist plans, where GPT is not required.
- Disabling **Math tower** skips math calls. Parallel string work may still
  execute, but final composition remains incomplete when math is required.
- Disabling **String tower** behaves symmetrically.
- In a sequential plan, disabling an upstream tower also skips downstream calls
  whose typed dependencies are unavailable.

These failure cases return a normal trace with `skipped` or `incomplete`
states; the server does not silently substitute another tower.

## Scope and operation

The learned V1.3 specialist dispatcher deliberately fails closed outside its
demonstrated task grammar: one-variable linear equations, exact string
operations, the registered parallel/sequential combinations, and the archival
pure-language calibration task. Prompts outside that grammar fall back to the
frozen GPT generalist when it is enabled. The trace labels this route
`generalist_fallback_v1` and warns that no specialist plan was accepted.

The generalist is the frozen base GPT-2 used by the experiment, not an
instruction-tuned chat model. It can continue a prompt such as `Hello`, but its
open-world answers are not expected to have modern chat-model quality or
reliability. These fallback responses are not evidence that V1.3 generalized
beyond its measured task panels.

Inference is serialized through one runtime lock so concurrent browser clients
cannot interleave GPU model execution. Static files and API responses include
no-store and restrictive browser security headers. Stop the server with
`Ctrl+C` in its terminal.

The API endpoints are:

- `GET /api/health` — device, available towers, runtime mode, artifact paths,
  and hashes.
- `POST /api/infer` — prompt, tower selections, generation limits, response,
  and full trace.

Example request:

```json
{
  "prompt": "Reverse 'callosal'.",
  "towers": {"gpt": true, "math": true, "string": true},
  "generation": {
    "gpt_max_new_tokens": 32,
    "specialist_max_new_tokens": 96
  }
}
```
