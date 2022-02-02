# Socratic Tutor: Learning and Tutoring Symbolic Manipulation Domains

This project consists of (1) learning how to solve procedural educational domains,
and then (2) teaching people to do the same, using an expert domain model.

The learning part was published in the NeurIPS 2021 paper "Contrastive Reinforcement Learning of Symbolic Reasoning Domains":


```
@inproceedings{poesia2021contrastive,
  author = {Poesia, Gabriel and Dong, WenXin and Goodman, Noah},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  title = {Contrastive Reinforcement Learning of Symbolic Reasoning Domains},
  website = {https://arxiv.org/abs/2106.09146},
  year = {2021}
} 
```

The educational domains themselves are implemented in Rust
(a Racket implementation is available for historical purposes,
though not actively supported. All final experiments used Rust).

The learning algorithms are implemented in Python 3, using PyTorch. You can install
all Python dependencies using:

```
pip install -r requirements.txt
```

Finally, the human evaluation is done with a Web application, written using Next.js and React.
It is located under the `webapp` directory. To install all dependencies there, simply
use `npm install` on the `webapp` directory.

## Educational domains (Rust environments)

Our first goal is to learn how to automatically solve exercises in a new educational domain.
Concretely, a domain defines:

(1) What exercises exist in the domain (e.g. what are algebraic equations),
(2) How to generate new exercises in that domain (e.g. an equations generator),
(3) Given one state, what are the actions available at that state (e.g. in an equation, could be changing the order of some terms, applying an operation to both sides, etc),
(4) Finally, how to detect that an exercise was solved (e.g. in equations, `x = 4` is solved, while `2x = 8` is still not)

This all you need to implement a domain.
The environments have a fast Rust implementation in the `commoncore` directory,
which can be easily called from the Python learning agents thanks to [https://github.com/PyO3/pyo3](PyO3).
To set them up, follow the steps below:

* First, install a recent Rust compiler (1.50+). The easiest way to do it is with rustup. Simply
  visit [https://rustup.rs/](https://rustup.rs/) and follow the instructions there. To check your
  installation, run `rustc --version` in the command line: you should get something like
  `rustc 1.51.0 (2fd73fabe 2021-03-23)`.
* Then, compile the dynamic library:

```
$ cd /path/to/socratic-tutor/commoncore
$ cargo build --release
```

  This might take a few minutes. It should download all dependencies and then compile our library.
  If all goes well, you should find the library at `target/release/libcommoncore.so`
  (or the equivalent extension in your operating system, e.g. dylib for Mac).
* Finally, we only need to place that library in a location that we can import from Python.
  Simply create a symbolic link at the root of the project and named `commoncore.so`, that points
  to the compiled library:

```
user@machine:~/socratic-tutor$ ln -s commoncore/target/release/libcommoncore.so ./commoncore.so
```

And that's it! If you open a Python shell, you should be able to directly `import commoncore`.
Also, now that you set up the symlink, if you compile the Rust library again (e.g. after it was updated),
we'll automatically pick up the latest version from Python.

## Learning agents

Several learning algorithms are implemented to learn the domains.
They are all in `agent.py`, which is a file that also implements evaluation.

NOTE: If you don't have WandB, you should comment out all lines of code that use wandb in `agent.py` and `evaluation.py`.

To perform training and evaluation, we use `agent.py`. Run the following command:
```
python agent.py [-h] --config CONFIG [--learn] [--experiment] [--eval] [--eval-checkpoints] [--debug] [--range RANGE] [--gpu GPU]
```

- `--config`: Path to configuration file, or inline JSON. A template configuration file for the `--learn` mode is given in [`template_config.txt`](template_config.txt). (Note: The template currently does not comprehensively list all possible configurations.)
- `--learn`: Put an agent to learn from the environment.
- `--experiment`: Run a batch of experiments with multiple agents and environments.
- `--eval`: Evaluate a learned policy.
- `--eval-checkpoints`: Show the evolution of a learned policy during interaction.
- `--debug`: Enable debug messages.
- `--range RANGE`: Range of experiments to run. Format: 2-5 means range [2, 5).Used to split experiments across multiple machines. Default: all.
- `--gpu GPU`: Which GPU to use (e.g. `"cuda:0"`); defaults to CPU if none is specified.

`--learn` is used to run a single experiment (one agent on one domain), whereas `--experiment` is used to run a batch of experiments (e.g., multiple agents on multiple domains with multiple runs in each configuration). You almost surely want to use `--experiment` since it is more general, even if to perform a single run. Here is a complete config file to run `--experiment` with the NCE (ConPoLe) agent on the fractions domain (single run):

```json
{
  "experiment_id": "test",
  "domains": ["fractions"],
  "environment_backend": "Rust",
  "wandb_project": "test",
  "gpus": [0],
  "n_runs": 1,
  "agents": [
    {
      "type": "NCE",
      "name": "ConPoLe",
      "n_future_states": 1,
      "replay_buffer_size": 100000,
      "max_depth": 30,
      "beam_size": 10,
      "initial_depth": 8,
      "depth_step": 1,
      "optimize_every": 16,
      "n_gradient_steps": 128,
      "keep_optimizer": true,
      "step_every": 10000,
      "n_bootstrap_problems": 100,
      "q_function": {
        "type": "Bilinear",
        "char_emb_dim": 64,
        "hidden_dim": 256,
        "mlp": true,
        "lstm_layers": 2
      }
    }
  ],
  "eval_environment": {
    "evaluate_every": 100000,
    "eval_config": {
      "max_steps": 30,
      "n_problems": 200
    },
    "output_root": "output",
    "max_steps": 10000000,
    "print_every": 10000
  }
}
